from segmentation.repositories.yolo_segmentation_model_repository import (
    SegmentationModel,
    YoloSegmentationModelRepository,
)
from services.base_vision_service import BaseVisionService
from services.tracking_service import TrackingService


class SegmentationService(BaseVisionService):
    def __init__(self, model: SegmentationModel = "yolo26x-seg"):
        if model != "yolo26x-seg":
            raise ValueError(f"Unsupported model for SegmentationService: {model}")
        super().__init__(YoloSegmentationModelRepository(model=model))
        self.tracking_service = TrackingService(self.model_repository)

    def predict_frame_bytes(
        self, frame_bytes, conf=0.5, tracking=False, tracker="bytetrack", classes=None
    ):
        if tracking:
            return self.tracking_service.track_frame_bytes(
                frame_bytes, conf, tracker, classes
            )
        return super().predict_frame_bytes(frame_bytes, conf, classes)
