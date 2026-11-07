from classification.repositories.chest_xray_classification_repository import (
    ChestXrayClassificationRepository,
)
from services.base_vision_service import BaseVisionService


class ChestXrayClassificationService(BaseVisionService):
    def __init__(self):
        super().__init__(ChestXrayClassificationRepository())
