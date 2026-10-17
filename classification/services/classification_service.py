from classification.repositories.yolo_classification_model_repository import (
    YoloClassificationModelRepository,
)
from services.base_vision_service import BaseVisionService


class ClassificationService(BaseVisionService):
    def __init__(self):
        super().__init__(YoloClassificationModelRepository())
