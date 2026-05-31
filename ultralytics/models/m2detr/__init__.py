# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .model import M2DETR
from .train import M2DETRTrainer
from .val import M2DETRValidator

__all__ = "M2DETR", "M2DETRTrainer", "M2DETRValidator"
