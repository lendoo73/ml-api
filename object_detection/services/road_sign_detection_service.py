from object_detection.repositories.road_sign_detection_repository import (
    RoadSignDetectionRepository,
)
from services.base_vision_service import BaseVisionService
from object_detection.services.object_tracking_service import ObjectTrackingService


class RoadSignDetectionService(BaseVisionService):
    def __init__(self):
        repository = RoadSignDetectionRepository()
        super().__init__(repository)
        self.tracking_service = ObjectTrackingService(repository)

    def predict_frame_bytes(self, frame_bytes: bytes, conf=0.5, tracking=False, tracker="bytetrack"):
        if tracking:
            return self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker)
        return super().predict_frame_bytes(frame_bytes, conf)
