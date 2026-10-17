from pathlib import Path

from object_detection.repositories.yolo_model_repository import YoloModelRepository


FACE_MASK_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "weights" / "yolo26x_face_mask_detect.pt"
)


class FaceMaskDetectionRepository(YoloModelRepository):
    def __init__(self):
        super().__init__(FACE_MASK_MODEL_PATH)
