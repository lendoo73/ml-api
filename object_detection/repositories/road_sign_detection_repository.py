from pathlib import Path

from object_detection.repositories.yolo_model_repository import YoloModelRepository


ROAD_SIGN_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "weights" / "yolo26x_road_sign-detect.pt"
)


class RoadSignDetectionRepository(YoloModelRepository):
    def __init__(self):
        super().__init__(ROAD_SIGN_MODEL_PATH)
