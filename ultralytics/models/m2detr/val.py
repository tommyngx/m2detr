# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validator for M2DETR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import YAML

from .data import build_m2detr_csv_dataset, check_m2detr_csv_dataset, is_m2detr_csv_dataset


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
        self._last_cls_logits = None
        self.image_cls_true = []
        self.image_cls_pred = []
        self.image_cls_conf = []

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

    def get_stats(self) -> dict[str, Any]:
        """Return detection stats plus optional image-level classification stats."""
        stats = super().get_stats()
        if not self.image_cls_true:
            return stats
        y_true = np.asarray(self.image_cls_true, dtype=int)
        y_pred = np.asarray(self.image_cls_pred, dtype=int)
        labels = np.union1d(y_true, y_pred)
        acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
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
                "metrics/image_cls_precision": float(np.mean(precisions)) if precisions else 0.0,
                "metrics/image_cls_recall": float(np.mean(recalls)) if recalls else 0.0,
                "metrics/image_cls_f1": float(np.mean(f1s)) if f1s else 0.0,
            }
        )
        return stats

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
