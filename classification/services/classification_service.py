from classification.repositories.yolo_classification_model_repository import (
    ClassificationModel,
    YoloClassificationModelRepository,
)
from services.base_vision_service import BaseVisionService


class ClassificationService(BaseVisionService):
    def __init__(self, model: ClassificationModel = "yolo26x-cls"):
        super().__init__(YoloClassificationModelRepository(model))
