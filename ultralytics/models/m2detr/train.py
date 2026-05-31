# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Trainer for M2DETR."""

from copy import copy

import numpy as np

from ultralytics.data import build_dataloader
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import M2DETRDetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, LOCAL_RANK, RANK, colorstr
from ultralytics.utils.plotting import plot_labels
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first

from .data import build_m2detr_csv_dataset, check_m2detr_csv_dataset, is_m2detr_csv_dataset
from .plot import plot_m2detr_results
from .val import M2DETRValidator


class M2DETRTrainer(RTDETRTrainer):
    """RT-DETR trainer that builds M2DETRDetectionModel and reports image classification loss."""

    def get_dataset(self):
        """Load YOLO-format datasets or the M2DETR CSV metadata format."""
        if is_m2detr_csv_dataset(self.args.data):
            data = check_m2detr_csv_dataset(self.args.data, save_dir=self.save_dir)
            if self.args.single_cls:
                LOGGER.info("Overriding detection class names with single class.")
                data["names"] = {0: "item"}
                data["nc"] = 1
            return data
        return super().get_dataset()

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build RT-DETR or CSV-backed M2DETR dataset."""
        if self.data.get("m2detr_csv"):
            return build_m2detr_csv_dataset(
                self.data,
                imgsz=self.args.imgsz,
                mode=mode,
                augment=mode == "train",
                prefix=colorstr(f"{mode}: "),
                batch_size=batch,
            )
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader, preserving CSV dataset collation when used."""
        if not self.data.get("m2detr_csv"):
            return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=mode == "train",
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return an initialized M2DETR model."""
        model = M2DETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach detection and image-class metadata to the model."""
        super().set_model_attributes()
        if hasattr(self.model, "cls_names"):
            self.model.cls_names = self.data.get("cls_names", self.model.cls_names)
        if hasattr(self.model, "cls_nc"):
            data_cls_nc = self.data.get("cls_nc")
            if data_cls_nc and int(data_cls_nc) != int(self.model.cls_nc):
                LOGGER.warning(
                    f"M2DETR data cls_nc={data_cls_nc} differs from model cls_nc={self.model.cls_nc}; "
                    "update the model YAML cls_nc if you need a non-binary image classification head."
                )

    def get_validator(self):
        """Return the standard RT-DETR validator while tracking the extra training loss."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "image_cls_loss"
        return M2DETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def plot_training_labels(self):
        """Plot detection labels for CSV and YOLO datasets without failing on negative-only splits."""
        if not self.data.get("m2detr_csv"):
            return super().plot_training_labels()

        boxes = [lb["bboxes"] for lb in self.train_loader.dataset.labels if len(lb["bboxes"])]
        cls = [lb["cls"] for lb in self.train_loader.dataset.labels if len(lb["cls"])]
        if not boxes:
            LOGGER.info("M2DETR CSV train split has no detection boxes to plot.")
            return
        plot_labels(
            np.concatenate(boxes, 0),
            np.concatenate(cls, 0).squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Save Ultralytics plots plus M2DETR-specific research plots."""
        super().plot_metrics()
        try:
            plot_m2detr_results(self.csv, self.save_dir, on_plot=self.on_plot)
        except Exception as e:
            LOGGER.warning(f"Failed to create M2DETR plots: {e}")

    def final_eval(self):
        """Run final evaluation without sending CSV metadata through the YOLO dataset checker."""
        if not self.data.get("m2detr_csv"):
            return super().final_eval()

        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model} with M2DETR CSV dataloader...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False
            best_model, _ = load_checkpoint(model, device=self.device)
            ema_model = self.ema.ema
            self.ema.ema = best_model
            try:
                self.metrics = self.validator(trainer=self)
                self.metrics.pop("fitness", None)
            finally:
                self.ema.ema = ema_model
            self.epoch += 1
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1
