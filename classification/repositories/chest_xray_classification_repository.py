from pathlib import Path

from classification.repositories.yolo_classification_model_repository import (
    YoloClassificationModelRepository,
)


CHEST_XRAY_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "weights" / "yolo26x_chest_xray-cls.pt"
)


class ChestXrayClassificationRepository(YoloClassificationModelRepository):
    def __init__(self):
        super().__init__(CHEST_XRAY_MODEL_PATH)
