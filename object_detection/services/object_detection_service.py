import os

from fastapi.concurrency import run_in_threadpool

from object_detection.repositories.yolo_model_repository import YoloModelRepository
from object_detection.services.object_tracking_service import ObjectTrackingService
from services.base_vision_service import BaseVisionService
from utils.video import download_video_to_temp, save_uploadfile_to_temp
from exceptions import InvalidVideoSourceError


class ObjectDetectionService(BaseVisionService):
    def __init__(self):
        repository = YoloModelRepository()
        super().__init__(repository)
        self.tracking_service = ObjectTrackingService(repository)

    def predict_frame_bytes(
        self, frame_bytes: bytes, conf: float = 0.5, tracking=False,
        tracker="bytetrack", classes: list[int] | None = None,
    ):
        if tracking:
            return self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker, classes)
        return super().predict_frame_bytes(frame_bytes, conf, classes)

    async def predict_video_path(
        self, video_path, conf=0.5, progress_callback=None, plot_options=None,
        tracking=False, tracker="bytetrack", classes: list[int] | None = None,
    ):
        if tracking:
            return await run_in_threadpool(
                self.tracking_service.track_video,
                video_path,
                conf,
                tracker,
                progress_callback,
                classes,
            )
        return await super().predict_video_path(
            video_path, conf, progress_callback, plot_options, classes=classes
        )

    async def predict_annotated_video(
        self, source, plot_options=None, conf=0.5, progress_callback=None,
        tracking=False, tracker="bytetrack", classes: list[int] | None = None,
    ):
        if not tracking:
            return await super().predict_annotated_video(
                source, plot_options, conf, progress_callback, classes=classes
            )

        video_path = await save_uploadfile_to_temp(source)
        try:
            return await run_in_threadpool(
                self.tracking_service.track_annotated_video,
                video_path,
                plot_options,
                conf,
                tracker,
                progress_callback,
                classes,
            )
        finally:
            os.remove(video_path)

    async def predict_video_url(
        self, source_url: str, output: str = "json", plot_options=None, conf=0.5,
        progress_callback=None, tracking=False, tracker="bytetrack", classes: list[int] | None = None,
    ):
        if not tracking:
            return await super().predict_video_url(
                source_url, output, plot_options, conf, progress_callback, classes=classes
            )

        video_path = await run_in_threadpool(download_video_to_temp, source_url)
        try:
            try:
                if output == "video":
                    return await run_in_threadpool(
                        self.tracking_service.track_annotated_video,
                        video_path,
                        plot_options,
                        conf,
                        tracker,
                        progress_callback,
                        classes,
                    )
                return await run_in_threadpool(
                    self.tracking_service.track_video,
                    video_path,
                    conf,
                    tracker,
                    progress_callback,
                    classes,
                )
            except ValueError as error:
                raise InvalidVideoSourceError(
                    "Video URL does not point to a readable video file."
                ) from error
        finally:
            os.remove(video_path)
