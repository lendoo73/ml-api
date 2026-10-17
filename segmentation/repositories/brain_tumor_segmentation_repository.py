from pathlib import Path

from segmentation.models.entities.brain_tumor_segmentation import BrainTumorSegmentation
from segmentation.repositories.yolo_segmentation_model_repository import (
    YoloSegmentationModelRepository,
)


BRAIN_TUMOR_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "weights" / "yolo26x-brain_tumor-seg.pt"
)


class BrainTumorSegmentationRepository(YoloSegmentationModelRepository):
    def __init__(self):
        super().__init__(str(BRAIN_TUMOR_MODEL_PATH))

    @staticmethod
    def _segmentations_from_result(result) -> list[BrainTumorSegmentation]:
        if result.masks is None:
            return []
        return [
            BrainTumorSegmentation(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
                polygon=polygon.tolist(),
            )
            for polygon, class_id, confidence in zip(
                result.masks.xy, result.boxes.cls, result.boxes.conf
            )
        ]
