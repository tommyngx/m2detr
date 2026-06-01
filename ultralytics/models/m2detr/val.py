# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validator for M2DETR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.metrics import DetMetrics

from .data import build_m2detr_csv_dataset, check_m2detr_csv_dataset, is_m2detr_csv_dataset


class M2DETRMetrics(DetMetrics):
    """Detection metrics with a compact repr for notebooks and train return values."""

    def __str__(self) -> str:
        """Return a concise summary instead of dumping all metric attributes and curves."""
        try:
            p, r, map50, map5095 = self.mean_results()
            acc = getattr(self, "image_cls_acc", 0.0)
            auc = getattr(self, "image_cls_auc", 0.0)
            return (
                "M2DETRMetrics("
                f"Acc={acc:.5g}, AUC={auc:.5g}, precision={p:.5g}, recall={r:.5g}, "
                f"mAP50={map50:.5g}, mAP50-95={map5095:.5g}"
                ")"
            )
        except Exception:
            return "M2DETRMetrics()"

    __repr__ = __str__


class M2DETRValidator(RTDETRValidator):
    """RT-DETR validator with optional image-level classification metrics."""

    def __call__(self, trainer=None, model=None):
        """Run validation, allowing standalone validation on M2DETR CSV YAMLs."""
        if trainer is None and is_m2detr_csv_dataset(self.args.data):
            data = check_m2detr_csv_dataset(self.args.data, save_dir=self.save_dir)
            cfg = data["m2detr_csv_config"]
            checked_yaml = {
                "path": str(data["path"]),
                "train": data["train"],
                "val": data["val"],
                "train_csv": str(cfg["train_csv"]),
                "val_csv": str(cfg["val_csv"]),
                "m2detr_csv": True,
                "channels": data["channels"],
                "names": data["names"],
                "cls_names": data["cls_names"],
                "image_col": cfg.get("image_col"),
                "image_id_col": cfg.get("image_id_col"),
                "image_cls_col": cfg.get("image_cls_col"),
                "box_cls_col": cfg.get("box_cls_col"),
                "bbox_cols": cfg.get("bbox_cols"),
                "bbox_format": cfg.get("bbox_format"),
                "split_col": cfg.get("split_col"),
                "train_split": cfg.get("train_split"),
                "val_split": cfg.get("val_split"),
                "width_col": cfg.get("width_col"),
                "height_col": cfg.get("height_col"),
                "validate_bbox": cfg.get("validate_bbox"),
                "load_image_dims": cfg.get("load_image_dims"),
                "min_area": cfg.get("min_area"),
            }
            checked_yaml = {k: v for k, v in checked_yaml.items() if v is not None}
            checked_yaml_path = Path(self.save_dir) / "m2detr_csv_checked.yaml"
            YAML.save(checked_yaml_path, checked_yaml)
            self.args.data = str(checked_yaml_path)
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build RT-DETR or CSV-backed M2DETR validation dataset."""
        if self.data and is_m2detr_csv_dataset(self.data):
            if "_m2detr_csv_frames" not in self.data:
                self.data = check_m2detr_csv_dataset(self.data, save_dir=self.save_dir)
            return build_m2detr_csv_dataset(
                self.data,
                imgsz=self.args.imgsz,
                mode=mode,
                augment=False,
                prefix=f"{mode}: ",
                batch_size=batch,
            )
        return super().build_dataset(img_path, mode=mode, batch=batch)

    @staticmethod
    def _get_image_cls(batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        """Fetch image-level labels from common batch keys."""
        bs = int(batch["img"].shape[0])
        for key in ("im_cls", "image_cls", "cls_label", "image_label", "labels", "classification"):
            if key not in batch:
                continue
            labels = batch[key]
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            elif isinstance(labels, (list, tuple)) and all(torch.is_tensor(x) for x in labels):
                labels = torch.stack([x.reshape(()) if x.numel() == 1 else x.reshape(-1)[0] for x in labels]).to(device)
            else:
                labels = torch.as_tensor(labels, device=device)
            labels = labels.view(-1)
            if labels.numel() == bs:
                return labels.long()
        return None

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize detection and image classification metric state."""
        super().init_metrics(model)
        self.metrics.__class__ = M2DETRMetrics
        self._last_cls_logits = None
        self.image_cls_true = []
        self.image_cls_pred = []
        self.image_cls_conf = []
        self.image_cls_probs = []

    def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]):
        """Store image classification logits, then run RT-DETR postprocess."""
        self._last_cls_logits = None
        if isinstance(preds, (list, tuple)) and len(preds) > 1 and isinstance(preds[1], dict):
            self._last_cls_logits = preds[1].get("cls_logits")
        return super().postprocess(preds)

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update detection metrics and optional image-level classification metrics."""
        super().update_metrics(preds, batch)
        if self._last_cls_logits is None:
            return
        labels = self._get_image_cls(batch, self._last_cls_logits.device)
        if labels is None:
            return
        probs = self._last_cls_logits.softmax(-1)
        conf, pred = probs.max(-1)
        self.image_cls_true.extend(labels.detach().cpu().tolist())
        self.image_cls_pred.extend(pred.detach().cpu().tolist())
        self.image_cls_conf.extend(conf.detach().cpu().tolist())
        self.image_cls_probs.extend(probs.detach().cpu().tolist())

    def get_stats(self) -> dict[str, Any]:
        """Return detection stats plus optional image-level classification stats."""
        stats = super().get_stats()
        if not self.image_cls_true:
            return stats
        y_true = np.asarray(self.image_cls_true, dtype=int)
        y_pred = np.asarray(self.image_cls_pred, dtype=int)
        y_prob = np.asarray(self.image_cls_probs, dtype=float)
        labels = np.union1d(y_true, y_pred)
        acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
        auc = _classification_auc(y_true, y_prob)
        precisions, recalls, f1s = [], [], []
        for cls in labels:
            tp = float(((y_pred == cls) & (y_true == cls)).sum())
            fp = float(((y_pred == cls) & (y_true != cls)).sum())
            fn = float(((y_pred != cls) & (y_true == cls)).sum())
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        stats.update(
            {
                "metrics/image_cls_acc": acc,
                "metrics/image_cls_auc": auc,
                "metrics/image_cls_precision": float(np.mean(precisions)) if precisions else 0.0,
                "metrics/image_cls_recall": float(np.mean(recalls)) if recalls else 0.0,
                "metrics/image_cls_f1": float(np.mean(f1s)) if f1s else 0.0,
            }
        )
        self.metrics.image_cls_acc = acc
        self.metrics.image_cls_auc = auc
        self.metrics.image_cls_precision = stats["metrics/image_cls_precision"]
        self.metrics.image_cls_recall = stats["metrics/image_cls_recall"]
        self.metrics.image_cls_f1 = stats["metrics/image_cls_f1"]
        return stats

    def get_desc(self) -> str:
        """Return a compact multitask validation table header."""
        return ("%22s" + "%11s" * 4) % ("Class", "Acc", "AUC", "mAP50", "mAP50-95")

    def print_results(self) -> None:
        """Print compact multitask metrics during training and fuller details for standalone val/test."""
        p, r, map50, map5095 = self.metrics.mean_results()
        acc = getattr(self.metrics, "image_cls_acc", 0.0)
        auc = getattr(self.metrics, "image_cls_auc", 0.0)
        pf = "%22s" + "%11.3g" * 4
        LOGGER.info(pf % ("all", acc, auc, map50, map5095))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute detection metrics without labels")
        if not self.training:
            LOGGER.info(
                "Details: images=%s, instances=%s, Box(P=%.3g, R=%.3g), cls(P=%.3g, R=%.3g, F1=%.3g)"
                % (
                    self.seen,
                    int(self.metrics.nt_per_class.sum()),
                    p,
                    r,
                    getattr(self.metrics, "image_cls_precision", 0.0),
                    getattr(self.metrics, "image_cls_recall", 0.0),
                    getattr(self.metrics, "image_cls_f1", 0.0),
                )
            )

    def finalize_metrics(self) -> None:
        """Finalize detection metrics and save image classification confusion matrix if available."""
        super().finalize_metrics()
        if self.args.plots and self.image_cls_true:
            self._plot_image_cls_confusion_matrix()

    def _plot_image_cls_confusion_matrix(self) -> None:
        """Save an image classification confusion matrix."""
        import matplotlib.pyplot as plt

        y_true = np.asarray(self.image_cls_true, dtype=int)
        y_pred = np.asarray(self.image_cls_pred, dtype=int)
        labels = np.union1d(y_true, y_pred)
        if not labels.size:
            return
        label_to_i = {label: i for i, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[label_to_i[true], label_to_i[pred]] += 1

        fig, ax = plt.subplots(1, 1, figsize=(8, 7), tight_layout=True)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title("Image Classification Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)), labels=[str(x) for x in labels])
        ax.set_yticks(np.arange(len(labels)), labels=[str(x) for x in labels])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        path = Path(self.save_dir) / "image_cls_confusion_matrix.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.on_plot(path, {"type": "image_cls_confusion_matrix", "matrix": cm.tolist()})


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC for binary labels using rank statistics."""
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(y_score, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = ranks[y_true].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _classification_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute binary or macro one-vs-rest AUC from class probabilities."""
    if y_true.size == 0 or y_prob.ndim != 2 or y_prob.shape[0] != y_true.size:
        return 0.0
    if y_prob.shape[1] == 2:
        return _binary_auc((y_true == 1).astype(int), y_prob[:, 1])
    aucs = []
    for cls in np.unique(y_true):
        if cls < 0 or cls >= y_prob.shape[1]:
            continue
        aucs.append(_binary_auc((y_true == cls).astype(int), y_prob[:, cls]))
    return float(np.mean(aucs)) if aucs else 0.0
