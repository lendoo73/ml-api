from object_detection.repositories.face_mask_detection_repository import (
    FaceMaskDetectionRepository,
)
from object_detection.services.object_tracking_service import ObjectTrackingService
from services.base_vision_service import BaseVisionService


class FaceMaskDetectionService(BaseVisionService):
    def __init__(self):
        repository = FaceMaskDetectionRepository()
        super().__init__(repository)
        self.tracking_service = ObjectTrackingService(repository)

    def predict_frame_bytes(
        self, frame_bytes: bytes, conf=0.5, tracking=False, tracker="bytetrack", classes=None
    ):
        if tracking:
            return self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker, classes)
        return super().predict_frame_bytes(frame_bytes, conf, classes)
