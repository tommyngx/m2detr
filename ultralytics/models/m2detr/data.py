# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CSV dataset utilities for M2DETR.

This module mirrors the notebook/zdetr metadata flow: read image-level CSV rows,
group multiple bounding boxes per image, keep negative images, print useful
dataset stats, then expose an Ultralytics-compatible detection batch.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ultralytics.data.utils import DATASETS_DIR, IMG_FORMATS
from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import LOGGER, YAML, colorstr
from ultralytics.utils.checks import check_yaml

CSV_DATA_KEYS = ("csv", "csv_path", "metadata", "metadata_csv", "train_csv", "val_csv")


def is_m2detr_csv_dataset(dataset: str | Path | dict[str, Any]) -> bool:
    """Return True when a dataset YAML/dict declares CSV metadata inputs."""
    if isinstance(dataset, dict):
        return bool(dataset.get("m2detr_csv") or dataset.get("format") == "m2detr_csv" or any(k in dataset for k in CSV_DATA_KEYS))

    dataset = str(dataset)
    if not dataset.endswith((".yaml", ".yml")):
        return False
    file = check_yaml(dataset, hard=False)
    if not file:
        return False
    try:
        data = YAML.load(file)
    except Exception:
        return False
    return is_m2detr_csv_dataset(data)


def check_m2detr_csv_dataset(dataset: str | Path | dict[str, Any], save_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate an M2DETR CSV dataset config."""
    if isinstance(dataset, dict):
        data = dataset.copy()
        yaml_file = data.get("yaml_file")
    else:
        yaml_file = check_yaml(str(dataset))
        data = YAML.load(yaml_file, append_filename=True)

    yaml_parent = Path(yaml_file).parent if yaml_file else Path.cwd()
    path = _resolve_dataset_root(data.get("path"), yaml_parent)
    data["path"] = path
    data["channels"] = int(data.get("channels", 3))
    data["names"] = _normalize_names(data.get("names"), data.get("nc"), default={0: "lesion"})
    data["nc"] = len(data["names"])

    train_csv = _resolve_config_path(data.get("train_csv"), path)
    val_csv = _resolve_config_path(data.get("val_csv"), path)
    csv_path = _resolve_config_path(data.get("csv") or data.get("csv_path") or data.get("metadata_csv") or data.get("metadata"), path)
    if train_csv is None and csv_path is None:
        csv_path = next((path / x for x in ("metadata2.csv", "metadata.csv") if (path / x).exists()), None)
    if train_csv is None and csv_path is None:
        raise ValueError("M2DETR CSV dataset requires one of: csv, metadata, metadata_csv, train_csv.")

    if train_csv is None:
        train_csv = csv_path
    if val_csv is None:
        val_csv = csv_path or train_csv

    data["train"] = str(train_csv)
    data["val"] = str(val_csv)
    data["m2detr_csv"] = True
    data["m2detr_csv_config"] = _csv_config(data, path, train_csv, val_csv)

    frames, stats, cls_names = prepare_m2detr_csv_frames(data["m2detr_csv_config"], save_dir=save_dir)
    data["_m2detr_csv_frames"] = frames
    data["_m2detr_csv_stats"] = stats
    data["cls_names"] = _normalize_names(data.get("cls_names") or cls_names, data.get("cls_nc"), default={0: "negative", 1: "positive"})
    data["cls_nc"] = len(data["cls_names"])

    if len(frames["train"]) == 0:
        raise ValueError("M2DETR CSV train split is empty.")
    if len(frames["val"]) == 0:
        raise ValueError("M2DETR CSV val/test split is empty.")

    return data


def prepare_m2detr_csv_frames(config: dict[str, Any], save_dir: str | Path | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[int, str]]:
    """Read CSV metadata, group bboxes per image, and report dataset statistics."""
    train_raw = pd.read_csv(config["train_csv"])
    val_raw = pd.read_csv(config["val_csv"]) if config["val_csv"] != config["train_csv"] else train_raw.copy()

    if config["train_csv"] == config["val_csv"]:
        train_raw, val_raw = _split_single_csv(train_raw, config)
    else:
        train_raw = train_raw.copy()
        val_raw = val_raw.copy()

    combined = pd.concat([train_raw.assign(_m2_split="train"), val_raw.assign(_m2_split="val")], ignore_index=True)
    columns = _resolve_columns(combined, config)
    image_cls_map, cls_names = _build_image_cls_mapping(combined, columns["image_cls_col"], config.get("cls_names"))

    train_df, train_meta = _group_csv_dataframe(train_raw, columns, config, image_cls_map, split_name="train")
    val_df, val_meta = _group_csv_dataframe(val_raw, columns, config, image_cls_map, split_name="val")
    frames = {"train": train_df, "val": val_df}
    stats = _build_stats(train_raw, val_raw, train_df, val_df, columns, config, train_meta, val_meta)
    _log_stats(stats)
    if save_dir:
        _save_stats(frames, stats, save_dir)
    return frames, stats, cls_names


def build_m2detr_csv_dataset(
    data: dict[str, Any],
    imgsz: int | tuple[int, int],
    mode: str = "train",
    augment: bool = False,
    prefix: str = "",
    batch_size: int | None = None,
) -> "M2CSVDetectionDataset":
    """Build a train/val M2DETR CSV dataset from a checked data dict."""
    if "_m2detr_csv_frames" not in data:
        data = check_m2detr_csv_dataset(data)

    df = data["_m2detr_csv_frames"]["train" if mode == "train" else "val"]
    fraction = float(data.get("m2detr_csv_config", {}).get("fraction", 1.0) or 1.0)
    if mode == "train" and 0 < fraction < 1:
        df = df.sample(frac=fraction, random_state=0).reset_index(drop=True)

    return M2CSVDetectionDataset(
        df=df,
        root=data["path"],
        imgsz=imgsz,
        mode=mode,
        augment=augment,
        prefix=prefix,
        names=data["names"],
        cls_names=data.get("cls_names"),
        batch_size=batch_size,
        min_area=float(data.get("m2detr_csv_config", {}).get("min_area", 1.0)),
    )


class M2CSVDetectionDataset(Dataset):
    """Ultralytics-compatible M2DETR dataset backed by grouped CSV metadata."""

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        imgsz: int | tuple[int, int],
        mode: str = "train",
        augment: bool = False,
        prefix: str = "",
        names: dict[int, str] | None = None,
        cls_names: dict[int, str] | None = None,
        batch_size: int | None = None,
        min_area: float = 1.0,
    ):
        """Initialize dataset."""
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.imgsz = _imgsz_tuple(imgsz)
        self.mode = mode
        self.augment = augment
        self.prefix = prefix
        self.names = names or {0: "lesion"}
        self.cls_names = cls_names or {0: "negative", 1: "positive"}
        self.batch_size = batch_size or 1
        self.min_area = min_area
        self.rect = False
        self.labels = self._build_labels()
        LOGGER.info(f"{prefix}M2DETR CSV dataset: {len(self.df)} images, {sum(len(x['cls']) for x in self.labels)} boxes")

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one Ultralytics-style training sample."""
        row = self.df.iloc[idx]
        im_file = str(row["im_file"])
        image, ori_shape = self._load_image(im_file)
        ori_h, ori_w = ori_shape
        image, bboxes_ltwh = self._augment(image, row.get("bbox_list", []), ori_w)

        target_h, target_w = self.imgsz
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img = torch.from_numpy(np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1)))

        bboxes, cls = self._format_boxes(bboxes_ltwh, row.get("box_cls_list", []), ori_w, ori_h)
        image_cls = int(row.get("image_cls", -1))
        return {
            "img": img,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros((len(bboxes),), dtype=torch.float32),
            "im_file": im_file,
            "ori_shape": ori_shape,
            "ratio_pad": ((target_h / max(ori_h, 1), target_w / max(ori_w, 1)), (0.0, 0.0)),
            "image_cls": torch.tensor(image_cls, dtype=torch.long),
            "image_id": str(row.get("image_id", Path(im_file).stem)),
        }

    def _load_image(self, im_file: str) -> tuple[Image.Image, tuple[int, int]]:
        """Load RGB image, falling back to a black image for missing files."""
        try:
            image = Image.open(im_file).convert("RGB")
            w, h = image.size
            return image, (h, w)
        except Exception as e:
            LOGGER.warning(f"{self.prefix}Failed to read image '{im_file}': {e}")
            h, w = self.imgsz
            return Image.new("RGB", (w, h)), (h, w)

    def _augment(self, image: Image.Image, bboxes_ltwh: list[list[float]], width: int) -> tuple[Image.Image, list[list[float]]]:
        """Apply lightweight CSV-loader augmentation."""
        if self.augment and np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped = []
            for x, y, w, h in _as_box_list(bboxes_ltwh):
                flipped.append([max(0.0, width - x - w), y, w, h])
            return image, flipped
        return image, _as_box_list(bboxes_ltwh)

    def _format_boxes(
        self,
        bboxes_ltwh: list[list[float]],
        box_cls_list: list[int],
        img_w: int,
        img_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert source ltwh pixel boxes to normalized center xywh tensors."""
        bboxes, cls = [], []
        for i, box in enumerate(_as_box_list(bboxes_ltwh)):
            clipped = _clip_ltwh(*box, img_w=img_w, img_h=img_h, min_area=self.min_area)
            if clipped is None:
                continue
            x, y, w, h = clipped
            bboxes.append([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h])
            cls.append(int(box_cls_list[i]) if i < len(box_cls_list) else 0)
        if not bboxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0, 1), dtype=torch.float32)
        return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(cls, dtype=torch.float32).view(-1, 1)

    def _build_labels(self) -> list[dict[str, Any]]:
        """Build label cache used by Ultralytics stats and auto-batch helpers."""
        labels = []
        for _, row in self.df.iterrows():
            img_w = int(row.get("img_width") or 0)
            img_h = int(row.get("img_height") or 0)
            bboxes, cls = [], []
            if img_w > 0 and img_h > 0:
                for i, box in enumerate(_as_box_list(row.get("bbox_list", []))):
                    clipped = _clip_ltwh(*box, img_w=img_w, img_h=img_h, min_area=1.0)
                    if clipped is None:
                        continue
                    x, y, w, h = clipped
                    bboxes.append([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h])
                    box_cls = row.get("box_cls_list", [])
                    cls.append(int(box_cls[i]) if i < len(box_cls) else 0)
            labels.append(
                {
                    "im_file": row["im_file"],
                    "shape": (img_h, img_w),
                    "cls": np.asarray(cls, dtype=np.float32).reshape(-1, 1),
                    "bboxes": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
                }
            )
        return labels

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate CSV dataset samples into the batch format expected by RT-DETR."""
        new_batch = {
            "img": torch.stack([x["img"] for x in batch], 0),
            "cls": torch.cat([x["cls"] for x in batch], 0),
            "bboxes": torch.cat([x["bboxes"] for x in batch], 0),
            "batch_idx": torch.cat(
                [torch.full((len(x["bboxes"]),), i, dtype=torch.float32) for i, x in enumerate(batch)], 0
            ),
            "image_cls": torch.stack([x["image_cls"] for x in batch], 0),
            "im_file": [x["im_file"] for x in batch],
            "ori_shape": [x["ori_shape"] for x in batch],
            "ratio_pad": [x["ratio_pad"] for x in batch],
            "image_id": [x["image_id"] for x in batch],
        }
        return new_batch


def _resolve_dataset_root(path: str | Path | None, yaml_parent: Path) -> Path:
    """Resolve dataset root relative to YAML first, then Ultralytics datasets dir."""
    if path is None:
        return yaml_parent.resolve()
    root = Path(path).expanduser()
    if root.is_absolute():
        return root.resolve()
    yaml_relative = (yaml_parent / root).resolve()
    if yaml_relative.exists():
        return yaml_relative
    return (DATASETS_DIR / root).resolve()


def _resolve_config_path(path: str | Path | None, root: Path) -> Path | None:
    """Resolve config file path relative to dataset root."""
    if path in {None, ""}:
        return None
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _csv_config(data: dict[str, Any], root: Path, train_csv: Path, val_csv: Path) -> dict[str, Any]:
    """Extract CSV loader settings from dataset YAML."""
    image_root = _resolve_config_path(data.get("image_root") or data.get("images") or data.get("images_dir"), root)
    return {
        "root": root,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "image_root": image_root or root,
        "image_col": data.get("image_col"),
        "image_id_col": data.get("image_id_col"),
        "image_cls_col": data.get("image_cls_col") or data.get("target_column"),
        "box_cls_col": data.get("box_cls_col"),
        "bbox_cols": data.get("bbox_cols"),
        "bbox_format": str(data.get("bbox_format", "")).lower() or None,
        "split_col": data.get("split_col", "split"),
        "train_split": data.get("train_split", "train"),
        "val_split": data.get("val_split", data.get("test_split", ["val", "valid", "validation", "test"])),
        "width_col": data.get("width_col"),
        "height_col": data.get("height_col"),
        "validate_bbox": bool(data.get("validate_bbox", True)),
        "load_image_dims": bool(data.get("load_image_dims", True)),
        "min_area": float(data.get("min_area", 1.0)),
        "names": data.get("names"),
        "cls_names": data.get("cls_names"),
        "fraction": data.get("fraction", 1.0),
    }


def _split_single_csv(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one metadata CSV into train and val/test frames."""
    split_col = _find_col(df, config.get("split_col"), ("split", "fold", "subset", "stage"))
    if split_col:
        split = df[split_col].astype(str).str.lower()
        train_values = _as_lower_set(config.get("train_split", "train"))
        val_values = _as_lower_set(config.get("val_split", ["val", "valid", "validation", "test"]))
        train_df = df[split.isin(train_values)].copy()
        val_df = df[split.isin(val_values)].copy()
        if len(train_df) and len(val_df):
            return train_df, val_df
        LOGGER.warning("M2DETR CSV split column found but train/val values were incomplete; using full CSV for both splits.")
    else:
        LOGGER.warning("M2DETR CSV has no split column; using full CSV for both train and val.")
    return df.copy(), df.copy()


def _resolve_columns(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve all relevant column names from a CSV dataframe."""
    image_col = _find_col(df, config.get("image_col"), ("link", "image_path", "path", "file", "file_name", "filename", "img_path"), required=True)
    image_id_col = _find_col(df, config.get("image_id_col"), ("image_id", "img_id", "id", "uid")) or image_col
    image_cls_col = _find_col(df, config.get("image_cls_col"), ("cancer", "label", "classification", "target", "image_cls", "image_label"))
    box_cls_col = _find_col(df, config.get("box_cls_col"), ("box_cls", "bbox_cls", "class_id", "category_id", "category", "lesion_class"))
    width_col = _find_col(df, config.get("width_col"), ("img_width", "image_width", "original_width", "width_px", "w_img"))
    height_col = _find_col(df, config.get("height_col"), ("img_height", "image_height", "original_height", "height_px", "h_img"))
    bbox_cols, bbox_format = _resolve_bbox_columns(df, config)
    patient_col = _find_col(df, None, ("patient_id", "patient", "case_id", "study_id"))
    return {
        "image_col": image_col,
        "image_id_col": image_id_col,
        "image_cls_col": image_cls_col,
        "box_cls_col": box_cls_col,
        "width_col": width_col,
        "height_col": height_col,
        "bbox_cols": bbox_cols,
        "bbox_format": bbox_format,
        "patient_col": patient_col,
    }


def _resolve_bbox_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str] | None, str]:
    """Resolve bbox columns and source format."""
    if config.get("bbox_cols"):
        bbox_cols = [_find_col(df, c, (c,), required=True) for c in config["bbox_cols"]]
        bbox_format = (config.get("bbox_format") or "").lower()
        if not bbox_format:
            bbox_format = "xyxy" if any(c.lower() in {"x2", "y2", "xmax", "ymax"} for c in bbox_cols) else "xywh"
        return bbox_cols, bbox_format

    candidates = [
        (("x", "y", "width", "height"), "xywh"),
        (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
        (("x_min", "y_min", "x_max", "y_max"), "xyxy"),
        (("x1", "y1", "x2", "y2"), "xyxy"),
    ]
    for names, fmt in candidates:
        cols = [_find_col(df, None, (name,)) for name in names]
        if all(cols):
            return cols, (config.get("bbox_format") or fmt).lower()
    return None, "xywh"


def _group_csv_dataframe(
    df: pd.DataFrame,
    columns: dict[str, Any],
    config: dict[str, Any],
    image_cls_map: dict[str, int],
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Group one CSV split to one row per image."""
    groups: dict[str, dict[str, Any]] = {}
    meta = {"candidate_bbox_rows": 0, "invalid_bbox_rows": 0, "missing_image_dims": 0}
    dim_cache: dict[str, tuple[int, int]] = {}

    for _, row in df.iterrows():
        image_rel = str(row[columns["image_col"]])
        im_file = _resolve_image_path(image_rel, config["image_root"])
        image_id = _clean_value(row[columns["image_id_col"]], default=Path(image_rel).stem)
        key = str(image_id)
        img_w, img_h = _row_image_size(row, columns, im_file, config, dim_cache)
        if img_w <= 0 or img_h <= 0:
            meta["missing_image_dims"] += 1

        if key not in groups:
            groups[key] = {
                "image_id": key,
                "im_file": str(im_file),
                "link": image_rel,
                "split": split_name,
                "image_cls": _map_image_cls(row, columns["image_cls_col"], image_cls_map),
                "img_width": img_w,
                "img_height": img_h,
                "patient_id": _clean_value(row[columns["patient_col"]]) if columns["patient_col"] else "",
                "bbox_list": [],
                "box_cls_list": [],
            }
        elif groups[key]["img_width"] <= 0 and img_w > 0:
            groups[key]["img_width"] = img_w
            groups[key]["img_height"] = img_h

        box = _parse_bbox(row, columns)
        if box is None:
            continue
        meta["candidate_bbox_rows"] += 1
        if config.get("validate_bbox", True):
            box = _clip_ltwh(*box, img_w=img_w, img_h=img_h, min_area=float(config.get("min_area", 1.0)))
        if box is None:
            meta["invalid_bbox_rows"] += 1
            continue
        groups[key]["bbox_list"].append([float(x) for x in box])
        groups[key]["box_cls_list"].append(_map_box_cls(row, columns["box_cls_col"], config.get("names")))

    grouped = pd.DataFrame(groups.values())
    if grouped.empty:
        grouped = pd.DataFrame(
            columns=["image_id", "im_file", "link", "split", "image_cls", "img_width", "img_height", "patient_id", "bbox_list", "box_cls_list"]
        )
    grouped["num_bboxes"] = grouped["bbox_list"].apply(len)
    return grouped.reset_index(drop=True), meta


def _find_col(df: pd.DataFrame, configured: str | None, candidates: tuple[str, ...], required: bool = False) -> str | None:
    """Find a CSV column by configured name or case-insensitive candidates."""
    lower_to_col = {str(c).lower(): c for c in df.columns}
    if configured:
        if configured in df.columns:
            return configured
        if str(configured).lower() in lower_to_col:
            return lower_to_col[str(configured).lower()]
        if required:
            raise ValueError(f"Configured column '{configured}' was not found in CSV.")
        return None
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_to_col:
            return lower_to_col[name.lower()]
    if required:
        raise ValueError(f"None of the required columns were found: {candidates}")
    return None


def _resolve_image_path(image_rel: str, image_root: Path) -> Path:
    """Resolve an image path from CSV."""
    p = Path(image_rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidate = (image_root / p).resolve()
    if candidate.suffix.lower().lstrip(".") in IMG_FORMATS or candidate.exists():
        return candidate
    return candidate


def _row_image_size(
    row: pd.Series,
    columns: dict[str, Any],
    im_file: Path,
    config: dict[str, Any],
    dim_cache: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """Get image width/height from CSV or by loading the image header."""
    img_w = _to_int(row[columns["width_col"]]) if columns.get("width_col") else 0
    img_h = _to_int(row[columns["height_col"]]) if columns.get("height_col") else 0
    if img_w > 0 and img_h > 0:
        return img_w, img_h
    if not config.get("load_image_dims", True):
        return 0, 0
    key = str(im_file)
    if key not in dim_cache:
        try:
            with Image.open(im_file) as image:
                dim_cache[key] = image.size
        except Exception:
            dim_cache[key] = (0, 0)
    return dim_cache[key]


def _parse_bbox(row: pd.Series, columns: dict[str, Any]) -> list[float] | None:
    """Parse a source bbox row into ltwh pixel format."""
    bbox_cols = columns["bbox_cols"]
    if not bbox_cols:
        return None
    values = [_to_float(row[c]) for c in bbox_cols]
    if any(v is None for v in values):
        return None
    if all(abs(v) < 1e-9 for v in values):
        return None
    if columns["bbox_format"] in {"xyxy", "x1y1x2y2"}:
        x1, y1, x2, y2 = values
        x, y = min(x1, x2), min(y1, y2)
        return [x, y, abs(x2 - x1), abs(y2 - y1)]
    if columns["bbox_format"] in {"ltwh", "xywh"}:
        x, y, w, h = values
        return [x, y, w, h]
    raise ValueError(f"Unsupported bbox_format='{columns['bbox_format']}'. Use xywh, ltwh, or xyxy.")


def _clip_ltwh(x: float, y: float, w: float, h: float, img_w: int, img_h: int, min_area: float = 1.0) -> list[float] | None:
    """Validate and clip ltwh box to image bounds."""
    vals = np.asarray([x, y, w, h], dtype=np.float64)
    if not np.isfinite(vals).all():
        return None
    x, y, w, h = vals.tolist()
    x, y, w, h = max(0.0, x), max(0.0, y), max(0.0, w), max(0.0, h)
    if img_w > 0 and img_h > 0:
        if x >= img_w or y >= img_h:
            return None
        w = min(w, img_w - x)
        h = min(h, img_h - y)
    if w <= 0 or h <= 0 or w * h < min_area:
        return None
    return [x, y, w, h]


def _build_image_cls_mapping(df: pd.DataFrame, cls_col: str | None, configured_names: Any) -> tuple[dict[str, int], dict[int, str]]:
    """Create stable image-class mapping for numeric or string labels."""
    names = _normalize_names(configured_names, None, default=None) if configured_names is not None else None
    if cls_col is None:
        return {}, names or {0: "negative", 1: "positive"}

    values = [_clean_value(v) for v in df[cls_col].dropna().tolist()]
    values = [v for v in values if v != ""]
    if not values:
        return {}, names or {0: "negative", 1: "positive"}
    if all(_to_float(v) is not None for v in values):
        numeric = sorted({int(float(v)) for v in values})
        if names:
            return {}, names
        return {}, {i: str(i) for i in range(max(numeric) + 1)}

    if names:
        name_to_idx = {str(v): int(k) for k, v in names.items()}
        return name_to_idx, names
    unique = sorted({str(v) for v in values})
    return {name: i for i, name in enumerate(unique)}, {i: name for i, name in enumerate(unique)}


def _map_image_cls(row: pd.Series, cls_col: str | None, cls_map: dict[str, int]) -> int:
    """Map a row image class to an int label."""
    if cls_col is None:
        return 0
    value = _clean_value(row[cls_col])
    if value == "":
        return -1
    if value in cls_map:
        return int(cls_map[value])
    numeric = _to_float(value)
    return int(numeric) if numeric is not None else -1


def _map_box_cls(row: pd.Series, box_cls_col: str | None, names: Any) -> int:
    """Map detection class to int label."""
    if box_cls_col is None:
        return 0
    value = _clean_value(row[box_cls_col])
    numeric = _to_float(value)
    if numeric is not None:
        return int(numeric)
    names = _normalize_names(names, None, default={0: "lesion"})
    name_to_idx = {str(v): int(k) for k, v in names.items()}
    return name_to_idx.get(value, 0)


def _normalize_names(names: Any, nc: int | None = None, default: dict[int, str] | None = None) -> dict[int, str]:
    """Normalize class names to Ultralytics dict format."""
    if names is None:
        if default is not None:
            names = default
        elif nc is not None:
            names = {i: f"class_{i}" for i in range(int(nc))}
        else:
            names = {0: "class_0"}
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    return check_class_names(names)


def _build_stats(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: dict[str, Any],
    config: dict[str, Any],
    train_meta: dict[str, int],
    val_meta: dict[str, int],
) -> dict[str, Any]:
    """Build human-readable CSV preprocessing stats."""
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    label_counts = {
        "train": _counter_to_dict(Counter(train_df["image_cls"].tolist())),
        "val": _counter_to_dict(Counter(val_df["image_cls"].tolist())),
    }
    patient_counts = {}
    if "patient_id" in all_df.columns and all_df["patient_id"].astype(str).str.len().sum():
        patient_counts = {
            "train": int(train_df["patient_id"].replace("", np.nan).dropna().nunique()),
            "val": int(val_df["patient_id"].replace("", np.nan).dropna().nunique()),
            "total": int(all_df["patient_id"].replace("", np.nan).dropna().nunique()),
        }
    return {
        "csv": {"train": str(config["train_csv"]), "val": str(config["val_csv"])},
        "columns": columns,
        "raw_rows": {"train": int(len(train_raw)), "val": int(len(val_raw)), "total": int(len(train_raw) + len(val_raw))},
        "images": {"train": int(len(train_df)), "val": int(len(val_df)), "total": int(len(all_df))},
        "patients": patient_counts,
        "image_label_counts": label_counts,
        "bboxes": {
            "train_total": int(train_df["num_bboxes"].sum()) if len(train_df) else 0,
            "val_total": int(val_df["num_bboxes"].sum()) if len(val_df) else 0,
            "candidate_rows": int(train_meta["candidate_bbox_rows"] + val_meta["candidate_bbox_rows"]),
            "invalid_rows": int(train_meta["invalid_bbox_rows"] + val_meta["invalid_bbox_rows"]),
            "images_without_bbox": int((all_df["num_bboxes"] == 0).sum()) if len(all_df) else 0,
            "images_with_multi_bbox": int((all_df["num_bboxes"] > 1).sum()) if len(all_df) else 0,
            "max_per_image": int(all_df["num_bboxes"].max()) if len(all_df) else 0,
            "min_area": float(config.get("min_area", 1.0)),
        },
        "missing_image_dims": int(train_meta["missing_image_dims"] + val_meta["missing_image_dims"]),
    }


def _log_stats(stats: dict[str, Any]) -> None:
    """Print concise dataset stats to the Ultralytics logger."""
    images = stats["images"]
    raw_rows = stats["raw_rows"]
    bboxes = stats["bboxes"]
    LOGGER.info(colorstr("M2DETR CSV preprocessing:"))
    LOGGER.info(f"  Raw rows: train {raw_rows['train']}, val {raw_rows['val']}, total {raw_rows['total']}")
    LOGGER.info(f"  Images: train {images['train']}, val {images['val']}, total {images['total']}")
    if stats["patients"]:
        p = stats["patients"]
        LOGGER.info(f"  Unique patients: train {p['train']}, val {p['val']}, total {p['total']}")
    LOGGER.info(
        f"  BBoxes: train {bboxes['train_total']}, val {bboxes['val_total']}, invalid rows {bboxes['invalid_rows']}, "
        f"no-box images {bboxes['images_without_bbox']}, multi-box images {bboxes['images_with_multi_bbox']}, max/image {bboxes['max_per_image']}"
    )
    if stats["missing_image_dims"]:
        LOGGER.warning(f"  Missing image dimensions for {stats['missing_image_dims']} row(s).")
    LOGGER.info(f"  Image label counts: train {stats['image_label_counts']['train']} | val {stats['image_label_counts']['val']}")


def _save_stats(frames: dict[str, pd.DataFrame], stats: dict[str, Any], save_dir: str | Path) -> None:
    """Save stats JSON and grouped CSV files for inspection."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    serializable_stats = json.loads(json.dumps(stats, default=str))
    (save_dir / "m2detr_csv_stats.json").write_text(json.dumps(serializable_stats, indent=2), encoding="utf-8")
    for split, df in frames.items():
        view = df.copy()
        for col in ("bbox_list", "box_cls_list"):
            if col in view:
                view[col] = view[col].apply(json.dumps)
        view.to_csv(save_dir / f"m2detr_csv_{split}_images.csv", index=False)


def _imgsz_tuple(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize image size to (height, width)."""
    if isinstance(imgsz, (tuple, list)):
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _as_box_list(value: Any) -> list[list[float]]:
    """Return a clean list of ltwh bboxes."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [list(map(float, box)) for box in value if isinstance(box, (list, tuple, np.ndarray)) and len(box) == 4]


def _as_lower_set(value: Any) -> set[str]:
    """Normalize string/list split values to lowercase set."""
    if isinstance(value, (list, tuple, set)):
        return {str(v).lower() for v in value}
    return {str(value).lower()}


def _clean_value(value: Any, default: str = "") -> str:
    """Convert a pandas value to clean string."""
    if pd.isna(value):
        return default
    return str(value)


def _to_float(value: Any) -> float | None:
    """Safely parse a scalar as float."""
    try:
        if pd.isna(value):
            return None
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    """Safely parse a scalar as positive int."""
    value = _to_float(value)
    return int(value) if value is not None and value > 0 else 0


def _counter_to_dict(counter: Counter) -> dict[int, int]:
    """Convert Counter keys to ints where possible."""
    result = {}
    for k, v in counter.items():
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = str(k)
        result[k] = int(v)
    return dict(sorted(result.items(), key=lambda x: str(x[0])))
