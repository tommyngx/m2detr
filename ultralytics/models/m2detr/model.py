# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""M2DETR model interface."""

from ultralytics.engine.model import Model
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.nn.tasks import M2DETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .train import M2DETRTrainer
from .val import M2DETRValidator


class M2DETR(Model):
    """Multitask mammography DETR: RT-DETR detection plus image classification."""

    def __init__(self, model: str = "m2detr.yaml") -> None:
        """Initialize M2DETR from YAML or weights."""
        assert TORCH_1_11, "M2DETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """Map M2DETR task components."""
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": M2DETRValidator,
                "trainer": M2DETRTrainer,
                "model": M2DETRDetectionModel,
            }
        }
