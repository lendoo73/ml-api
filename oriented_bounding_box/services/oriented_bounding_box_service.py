import os

from fastapi.concurrency import run_in_threadpool

from services.base_vision_service import BaseVisionService
from utils.video import download_video_to_temp, save_uploadfile_to_temp
from oriented_bounding_box.repositories.yolo_oriented_bounding_box_model_repository import (
    YoloOrientedBoundingBoxModelRepository,
)


class OrientedBoundingBoxService(BaseVisionService):
    def __init__(self):
        super().__init__(YoloOrientedBoundingBoxModelRepository())

    def predict_frame_bytes(self, frame_bytes, conf=0.5, tracking=False, tracker="bytetrack"):
        if tracking:
            import cv2
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image frame")
            return self.model_repository.track(frame, conf, tracker)
        return super().predict_frame_bytes(frame_bytes, conf)

    async def predict_video_path(self, video_path, conf=0.5, progress_callback=None,
                                 tracking=False, tracker="bytetrack"):
        if tracking:
            return await run_in_threadpool(
                self.model_repository.track_video, video_path, conf, tracker, progress_callback
            )
        return await super().predict_video_path(video_path, conf, progress_callback)

    async def predict_annotated_video(self, source, plot_options=None, conf=0.5,
                                      progress_callback=None, tracking=False, tracker="bytetrack"):
        video_path = await save_uploadfile_to_temp(source)
        try:
            method = self.model_repository.track_annotated_video if tracking else self.model_repository.predict_annotated_video
            return await run_in_threadpool(
                method, video_path, plot_options=plot_options, conf=conf,
                tracker=tracker, progress_callback=progress_callback,
            ) if tracking else await run_in_threadpool(
                method, video_path, plot_options=plot_options, conf=conf,
                progress_callback=progress_callback,
            )
        finally:
            os.remove(video_path)

    async def predict_video_url(self, source_url, output="json", plot_options=None, conf=0.5,
                                progress_callback=None, tracking=False, tracker="bytetrack"):
        if not tracking:
            return await super().predict_video_url(source_url, output, plot_options, conf, progress_callback)
        video_path = await run_in_threadpool(download_video_to_temp, source_url)
        try:
            if output == "video":
                return await run_in_threadpool(
                    self.model_repository.track_annotated_video, video_path,
                    plot_options=plot_options, conf=conf, tracker=tracker,
                    progress_callback=progress_callback,
                )
            return await self.predict_video_path(video_path, conf, progress_callback, True, tracker)
        finally:
            os.remove(video_path)
