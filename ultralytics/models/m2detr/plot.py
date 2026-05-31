# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Plot helpers for M2DETR training runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_results_csv(csv_path: str | Path) -> list[dict[str, str]]:
    """Read an Ultralytics results.csv file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    """Return a numeric series, using NaN for missing values."""
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def _present(y: np.ndarray) -> bool:
    """Return whether a series has at least one finite value."""
    return bool(y.size and np.isfinite(y).any())


def _plot_line(ax, x: np.ndarray, y: np.ndarray, label: str, **kwargs) -> bool:
    """Plot a finite series if available."""
    if not _present(y):
        return False
    ax.plot(x, y, marker=".", linewidth=2, markersize=7, label=label, **kwargs)
    return True


def _highlight_best(ax, x: np.ndarray, y: np.ndarray, mode: str, label: str) -> None:
    """Mark the best finite point on a curve."""
    finite = np.isfinite(y)
    if not finite.any():
        return
    idxs = np.flatnonzero(finite)
    best_pos = np.nanargmin(y[finite]) if mode == "min" else np.nanargmax(y[finite])
    idx = idxs[best_pos]
    ax.scatter(x[idx], y[idx], s=120, c="#0057b8", zorder=10)
    ax.annotate(
        f"{label}: {y[idx]:.4f} (e{int(x[idx])})",
        xy=(x[idx], y[idx]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )


def _save(fig, path: Path, on_plot=None) -> None:
    """Save a figure and notify the Ultralytics plot registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    if on_plot:
        on_plot(path)


def plot_m2detr_results(csv_path: str | Path, save_dir: str | Path, on_plot=None) -> list[Path]:
    """Create M2DETR-specific training plots from Ultralytics results.csv.

    Saves:
        - figures/m2detr_training.png: loss, detection metrics, image-classification metrics.
        - figures/m2detr_losses.png: individual M2DETR train/val loss components.
    """
    rows = _read_results_csv(csv_path)
    if not rows:
        return []

    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    figures_dir = save_dir / "figures"
    x = _series(rows, "epoch")
    if not _present(x):
        x = np.arange(1, len(rows) + 1, dtype=float)

    train_loss_keys = ["train/giou_loss", "train/cls_loss", "train/l1_loss", "train/image_cls_loss"]
    val_loss_keys = ["val/giou_loss", "val/cls_loss", "val/l1_loss", "val/image_cls_loss"]
    train_losses = [_series(rows, k) for k in train_loss_keys]
    val_losses = [_series(rows, k) for k in val_loss_keys]
    train_total = np.nansum(np.vstack(train_losses), axis=0)
    val_total = np.nansum(np.vstack(val_losses), axis=0)

    saved = []
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    fig.patch.set_facecolor("#f7f7f7")
    for ax in axes:
        ax.set_facecolor("#f7f7f7")

    _plot_line(axes[0], x, train_total, "Train total loss", color="#c1121f")
    has_val = _plot_line(axes[0], x, val_total, "Val total loss", color="#1b7f3a")
    if has_val:
        _highlight_best(axes[0], x, val_total, "min", "Best val loss")
    axes[0].set_title("M2DETR Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    metric_specs = [
        ("metrics/precision(B)", "Precision", "#006ba4"),
        ("metrics/recall(B)", "Recall", "#ff800e"),
        ("metrics/mAP50(B)", "mAP50", "#595959"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#5f9ed1"),
    ]
    for key, label, color in metric_specs:
        _plot_line(axes[1], x, _series(rows, key), label, color=color)
    map50 = _series(rows, "metrics/mAP50(B)")
    _highlight_best(axes[1], x, map50, "max", "Best mAP50")
    axes[1].set_title("Detection Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(bottom=0)

    _plot_line(axes[2], x, _series(rows, "train/image_cls_loss"), "Train image cls loss", color="#c1121f")
    _plot_line(axes[2], x, _series(rows, "val/image_cls_loss"), "Val image cls loss", color="#1b7f3a")
    image_acc = _series(rows, "metrics/image_cls_acc")
    if _plot_line(axes[2], x, image_acc, "Val image cls acc", color="#006ba4"):
        _highlight_best(axes[2], x, image_acc, "max", "Best image acc")
    axes[2].set_title("Image Classification")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss / Accuracy")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    path = figures_dir / "m2detr_training.png"
    _save(fig, path, on_plot)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), tight_layout=True)
    axes = axes.ravel()
    loss_specs = [
        ("giou_loss", "GIoU Loss"),
        ("cls_loss", "Detection Class Loss"),
        ("l1_loss", "BBox L1 Loss"),
        ("image_cls_loss", "Image Class Loss"),
    ]
    for ax, (name, title) in zip(axes, loss_specs):
        _plot_line(ax, x, _series(rows, f"train/{name}"), f"train/{name}", color="#c1121f")
        _plot_line(ax, x, _series(rows, f"val/{name}"), f"val/{name}", color="#1b7f3a")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.45)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    path = figures_dir / "m2detr_losses.png"
    _save(fig, path, on_plot)
    plt.close(fig)
    saved.append(path)

    return saved
