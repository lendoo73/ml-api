from types import SimpleNamespace
from oriented_bounding_box.repositories.yolo_oriented_bounding_box_model_repository import (
    YoloOrientedBoundingBoxModelRepository,
)


class Tensor:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def tolist(self): return self.value


def test_obb_result_conversion_uses_xywhr_and_corners():
    result = SimpleNamespace(
        names={0: "car"},
        obb=SimpleNamespace(
            xywhr=Tensor([[10, 20, 30, 40, 0.25]]),
            xyxyxyxy=Tensor([[[0, 0], [20, 0], [20, 40], [0, 40]]]),
            cls=Tensor([0]), conf=Tensor([0.9]),
        ),
    )
    detection = YoloOrientedBoundingBoxModelRepository._detections_from_result(result)[0]
    assert detection.class_name == "car"
    assert detection.obb.angle == 0.25
    assert detection.obb.corners[2] == [20.0, 40.0]
