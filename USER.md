# M2DETR trong Ultralytics

M2DETR là nhánh multitask cho mammography: một head detection kiểu RT-DETR và một head image-level classification. Repo này tích hợp thêm module `m2detr` vào Ultralytics để tận dụng trainer, optimizer, logging, metric và checkpoint có sẵn.

## 1. Cài đặt

Chạy từ root repo:

```bash
cd /Users/francistommy/Desktop/BugHunter/Project/m2detr
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Nếu dùng backbone timm mới hơn, kiểm tra `timm` đã có:

```bash
python - <<'PY'
import timm
print(timm.__version__)
PY
```

## 2. Model YAML có sẵn

Các model nằm trong `ultralytics/cfg/models/m2detr/`.

| Model | Mục đích |
| --- | --- |
| `m2detr.yaml` | Default compact, timm ResNet50 backbone |
| `m2detr-l.yaml` | So sánh với RT-DETR-L backbone/head |
| `m2detr-x.yaml` | So sánh với RT-DETR-X backbone/head |
| `m2detr-convnextv2.yaml` | timm ConvNeXtV2 tiny |
| `m2detr-efficientnetv2.yaml` | timm EfficientNetV2 |
| `m2detr-dinov2.yaml` | timm DINOv2 small |
| `m2detr-dinov3.yaml` | timm DINOv3 ConvNeXt small |

Mặc định các YAML đang dùng:

```yaml
nc: 1       # số class detection, ví dụ lesion
cls_nc: 2  # số class image classification, ví dụ negative/positive
```

Nếu dataset classification không phải binary, sửa `cls_nc` trong model YAML tương ứng để khớp với `cls_names` trong data YAML.

## 3. Chuẩn bị CSV

Loader mới đọc CSV giống flow trong `notebook/mammo2detr/zdetr`: một dòng là một annotation, nhiều bbox thì lặp lại cùng `image_id`, ảnh âm tính vẫn được giữ với bbox rỗng/NaN/0.

Ví dụ CSV:

```csv
image_id,link,cancer,split,x,y,width,height,img_width,img_height,patient_id
img_001,images/img_001.png,0,train,,,,,1024,1024,p001
img_002,images/img_002.png,1,train,120,240,80,96,1024,1024,p002
img_002,images/img_002.png,1,train,400,300,60,70,1024,1024,p002
img_003,images/img_003.png,1,test,180,200,90,110,1024,1024,p003
```

Cột mặc định:

| Cột | Ý nghĩa |
| --- | --- |
| `link` | đường dẫn ảnh, tương đối theo `path` hoặc `image_root` |
| `image_id` | ID ảnh để group nhiều bbox |
| `cancer` | label classification cấp ảnh |
| `split` | `train`, `val`, `test`... |
| `x,y,width,height` | bbox dạng top-left + width/height |
| `img_width,img_height` | kích thước ảnh, nên có để preprocessing nhanh |
| `patient_id` | optional, dùng để in stats unique patients |

Loader cũng hỗ trợ bbox dạng `xyxy` qua `bbox_cols: [xmin, ymin, xmax, ymax]` và `bbox_format: xyxy`.

## 4. Data YAML

Template có sẵn tại `ultralytics/cfg/datasets/m2detr-csv.yaml`. Một data YAML tối thiểu:

```yaml
path: /Users/francistommy/Desktop/BugHunter/Datasets/mammo

csv: metadata.csv
split_col: split
train_split: train
val_split: [val, valid, validation, test]

image_col: link
image_id_col: image_id
image_cls_col: cancer
bbox_cols: [x, y, width, height]
bbox_format: xywh
width_col: img_width
height_col: img_height

names:
  0: lesion

cls_names:
  0: negative
  1: positive

validate_bbox: true
load_image_dims: true
min_area: 1
```

Nếu train/val là hai CSV riêng:

```yaml
path: /Users/francistommy/Desktop/BugHunter/Datasets/mammo
train_csv: train_metadata.csv
val_csv: val_metadata.csv
```

Nếu không khai báo `csv`, loader sẽ tự tìm `metadata2.csv`, rồi `metadata.csv` trong `path`.

## 5. Kiểm tra data và stats

Chạy trước khi train dài:

```bash
python - <<'PY'
from ultralytics.models.m2detr.data import check_m2detr_csv_dataset

data = check_m2detr_csv_dataset(
    "/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
    save_dir="runs/m2detr/check_data",
)
print(data["_m2detr_csv_stats"])
PY
```

Output quan trọng:

```text
M2DETR CSV preprocessing:
  Raw rows: train ..., val ..., total ...
  Images: train ..., val ..., total ...
  Unique patients: train ..., val ..., total ...
  BBoxes: train ..., val ..., invalid rows ..., no-box images ..., multi-box images ..., max/image ...
  Image label counts: train {...} | val {...}
```

Các file được lưu thêm để soi data:

```text
runs/m2detr/check_data/m2detr_csv_stats.json
runs/m2detr/check_data/m2detr_csv_train_images.csv
runs/m2detr/check_data/m2detr_csv_val_images.csv
```

## 6. Smoke test train 1 epoch

Nên chạy nhỏ trước để kiểm tra toàn bộ pipeline:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/m2detr/m2detr.yaml")
model.train(
    data="/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
    imgsz=256,
    epochs=1,
    batch=1,
    device="cpu",
    workers=0,
    amp=False,
    plots=True,
    project="runs/m2detr",
    name="smoke_m2detr",
    exist_ok=True,
)
PY
```

Nếu smoke test chạy qua train và final validation, chuyển sang GPU.

## 7. Train chính

Ví dụ ResNet50 default:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/m2detr/m2detr.yaml")
model.train(
    data="/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
    imgsz=640,
    epochs=100,
    batch=8,
    device=0,
    workers=8,
    amp=True,
    plots=True,
    project="runs/m2detr",
    name="m2detr_resnet50",
)
PY
```

So sánh RT-DETR-L/X:

```bash
python - <<'PY'
from ultralytics import YOLO

for cfg, name in [
    ("ultralytics/cfg/models/m2detr/m2detr-l.yaml", "m2detr_l"),
    ("ultralytics/cfg/models/m2detr/m2detr-x.yaml", "m2detr_x"),
]:
    YOLO(cfg).train(
        data="/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
        imgsz=640,
        epochs=100,
        batch=8,
        device=0,
        workers=8,
        amp=True,
        plots=True,
        project="runs/m2detr",
        name=name,
    )
PY
```

So sánh timm backbone:

```bash
python - <<'PY'
from ultralytics import YOLO

configs = {
    "convnextv2": "ultralytics/cfg/models/m2detr/m2detr-convnextv2.yaml",
    "efficientnetv2": "ultralytics/cfg/models/m2detr/m2detr-efficientnetv2.yaml",
    "dinov2": "ultralytics/cfg/models/m2detr/m2detr-dinov2.yaml",
    "dinov3": "ultralytics/cfg/models/m2detr/m2detr-dinov3.yaml",
}

for name, cfg in configs.items():
    YOLO(cfg).train(
        data="/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
        imgsz=640,
        epochs=100,
        batch=8,
        device=0,
        workers=8,
        amp=True,
        plots=True,
        project="runs/m2detr",
        name=f"m2detr_{name}",
    )
PY
```

## 8. Validation lại checkpoint

Dùng `M2DETR` trực tiếp để chắc chắn checkpoint dùng đúng wrapper:

```bash
python - <<'PY'
from ultralytics import M2DETR

model = M2DETR("runs/m2detr/m2detr_resnet50/weights/best.pt")
metrics = model.val(
    data="/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml",
    imgsz=640,
    batch=8,
    device=0,
    workers=8,
    plots=True,
    project="runs/m2detr",
    name="val_m2detr_resnet50",
)
print(metrics.results_dict)
PY
```

## 9. Output cần xem

Trong mỗi run:

```text
runs/m2detr/<run_name>/results.csv
runs/m2detr/<run_name>/weights/best.pt
runs/m2detr/<run_name>/weights/last.pt
runs/m2detr/<run_name>/m2detr_csv_stats.json
runs/m2detr/<run_name>/m2detr_csv_train_images.csv
runs/m2detr/<run_name>/m2detr_csv_val_images.csv
```

Nếu `plots=True`, M2DETR lưu thêm plot riêng, không gộp vào plotting gốc:

```text
runs/m2detr/<run_name>/figures/m2detr_training.png
runs/m2detr/<run_name>/figures/m2detr_losses.png
runs/m2detr/<run_name>/image_cls_confusion_matrix.png
```

Metrics classification image-level sẽ xuất hiện dạng:

```text
metrics/image_cls_acc
metrics/image_cls_precision
metrics/image_cls_recall
metrics/image_cls_f1
```

## 10. CLI thay thế

Python API là đường khuyến nghị. Nếu muốn dùng CLI:

```bash
yolo detect train \
  model=ultralytics/cfg/models/m2detr/m2detr.yaml \
  data=/Users/francistommy/Desktop/BugHunter/Datasets/mammo/m2detr-data.yaml \
  imgsz=640 epochs=100 batch=8 device=0 workers=8 plots=True \
  project=runs/m2detr name=m2detr_resnet50
```

## 11. Lỗi thường gặp

`train split is empty` hoặc `val split is empty`:
kiểm tra `split_col`, `train_split`, `val_split` có khớp giá trị trong CSV không.

Stats báo nhiều `invalid rows`:
kiểm tra bbox có đúng format không. Nếu dùng `[xmin, ymin, xmax, ymax]` phải đặt `bbox_format: xyxy`.

Preprocessing chậm:
thêm `img_width` và `img_height` vào CSV. Nếu thiếu, loader sẽ mở từng ảnh để lấy kích thước.

Warning `data cls_nc differs from model cls_nc`:
sửa `cls_nc` trong model YAML cho khớp số class trong `cls_names`.

Timm báo không có model:
cập nhật `timm`, hoặc dùng YAML backbone khác đã chạy được trong môi trường hiện tại.

Matplotlib/font cache warning trên macOS:

```bash
export MPLCONFIGDIR=/private/tmp/mpl
```
