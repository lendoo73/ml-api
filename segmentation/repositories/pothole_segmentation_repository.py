from pathlib import Path

from segmentation.repositories.yolo_segmentation_model_repository import (
    YoloSegmentationModelRepository,
)


class PotholeSegmentationRepository(YoloSegmentationModelRepository):
    def __init__(self):
        path = Path(__file__).resolve().parent.parent / "weights" / "yolo26x_pothole_segmentation.pt"
        super().__init__(str(path))

    def predict(self, source, conf=0.5, **options):
        return self._segmentations_from_result(self._predict_result(source, conf, **options))

    def _predict_result(self, source, conf=0.5, **options):
        return self.model.predict(source=source, conf=conf, verbose=False, **options)[0]

