from pose_estimation.repositories.yolo_pose_model_repository import YoloPoseModelRepository
from services.base_vision_service import BaseVisionService
from services.tracking_service import TrackingService
from services.tracking_service import TrackingService


class PoseEstimationService(BaseVisionService):
    def __init__(self):
        super().__init__(YoloPoseModelRepository())
        self.tracking_service = TrackingService(self.model_repository)

    def predict_frame_bytes(self, frame_bytes, conf=0.5, tracking=False, tracker="bytetrack"):
        return self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker) if tracking else super().predict_frame_bytes(frame_bytes, conf)
