from segmentation.repositories.pothole_segmentation_repository import PotholeSegmentationRepository
from segmentation.services.segmentation_service import SegmentationService
from segmentation.repositories.yolo_segmentation_model_repository import SegmentationModel
from services.tracking_service import TrackingService


class PotholeSegmentationService(SegmentationService):
    def __init__(self, model: SegmentationModel = "pothole"):
        if model != "pothole":
            raise ValueError(f"Unsupported model for PotholeSegmentationService: {model}")
        super().__init__()
        self.model_repository = PotholeSegmentationRepository(model=model)
        self.tracking_service = TrackingService(self.model_repository)

    def predict_frame(self, frame, output="json", plot_options=None, conf=0.5, **options):
        if output == "image":
            result = self.model_repository._predict_result(frame, conf, **options)
            import cv2
            import numpy as np
            annotated = result.plot(pil=True, **(plot_options or {}))
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Could not encode annotated image")
            return encoded.tobytes()
        return self.model_repository.predict(frame, conf, **options)

    async def predict_image(self, source, output="json", plot_options=None, conf=0.5, **options):
        from fastapi.concurrency import run_in_threadpool
        from utils.casting import decode_upload_file_to_ndarray
        image = await decode_upload_file_to_ndarray(source)
        return await run_in_threadpool(self.predict_frame, image, output, plot_options, conf, **options)

    async def predict_url(self, source_url, output="json", plot_options=None, conf=0.5, **options):
        from fastapi.concurrency import run_in_threadpool
        from utils.image import download_image_to_ndarray
        image = await run_in_threadpool(download_image_to_ndarray, source_url)
        return await run_in_threadpool(self.predict_frame, image, output, plot_options, conf, **options)
