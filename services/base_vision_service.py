import os

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool

from exceptions import InvalidVideoSourceError
from media_search.services.media_search_service import MediaSearchService
from utils.casting import decode_upload_file_to_ndarray
from utils.image import download_image_to_ndarray
from utils.video import download_video_to_temp, save_uploadfile_to_temp


class BaseVisionService:
    def __init__(self, model_repository):
        self.model_repository = model_repository
        self.media_search_service = MediaSearchService()

    async def predict_image(
        self,
        source,
        output: str = "json",
        plot_options: dict | None = None,
        conf: float = 0.5,
        classes: list[int] | None = None,
    ):
        image = await decode_upload_file_to_ndarray(source)
        if classes is None:
            return await run_in_threadpool(self.predict_frame, image, output, plot_options, conf)
        return await run_in_threadpool(
            self.predict_frame, image, output, plot_options, conf, classes
        )

    def predict_frame(
        self,
        frame,
        output: str = "json",
        plot_options: dict | None = None,
        conf: float = 0.5,
        classes: list[int] | None = None,
    ):
        if classes is None:
            if output == "image":
                return self.model_repository.predict_annotated_image(frame, plot_options, conf)
            return self.model_repository.predict(frame, conf)
        if output == "image":
            return self.model_repository.predict_annotated_image(frame, plot_options, conf, classes)
        return self.model_repository.predict(frame, conf, classes)

    def predict_frame_bytes(self, frame_bytes: bytes, conf: float = 0.5, classes=None):
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image frame")
        return self.predict_frame(frame, conf=conf, classes=classes)

    def predict_tracked_frame_bytes(self, frame_bytes: bytes, conf=0.5, tracker="bytetrack"):
        return self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker)

    async def predict_url(
        self,
        source_url: str,
        output: str = "json",
        plot_options: dict | None = None,
        conf: float = 0.5,
        classes: list[int] | None = None,
    ):
        image = await run_in_threadpool(download_image_to_ndarray, source_url)
        if classes is None:
            return await run_in_threadpool(self.predict_frame, image, output, plot_options, conf)
        return await run_in_threadpool(
            self.predict_frame, image, output, plot_options, conf, classes
        )

    async def predict_search(
        self,
        query: str,
        output: str = "json",
        plot_options: dict | None = None,
        conf: float = 0.5,
        classes: list[int] | None = None,
    ):
        search_result = await run_in_threadpool(
            self.media_search_service.search_images,
            query=query,
            per_page=1,
            randomize=True,
        )
        if not search_result.items:
            raise ValueError(f"No image found for query: {query}")
        return await self.predict_url(
            search_result.items[0].source_url,
            output,
            plot_options,
            conf,
            classes,
        )

    async def predict_video(self, source, conf=0.5, progress_callback=None):
        video_path = await save_uploadfile_to_temp(source)
        try:
            return await self.predict_video_path(video_path, conf, progress_callback)
        finally:
            os.remove(video_path)

    async def predict_video_path(
        self, video_path, conf=0.5, progress_callback=None, plot_options=None,
        tracking=False, tracker="bytetrack", classes=None,
    ):
        if tracking:
            return await self.predict_tracked_video_path(
                video_path, conf, progress_callback, tracker, classes
            )
        if classes is None:
            return await run_in_threadpool(
                self.model_repository.predict_video, video_path, conf=conf,
                progress_callback=progress_callback,
            )
        return await run_in_threadpool(
            self.model_repository.predict_video, video_path, conf=conf, classes=classes,
            progress_callback=progress_callback,
        )

    async def predict_tracked_video_path(
        self, video_path, conf=0.5, progress_callback=None, tracker="bytetrack", classes=None
    ):
        return await run_in_threadpool(
            self.tracking_service.track_video,
            video_path,
            conf,
            tracker,
            progress_callback,
            classes,
        )

    async def predict_annotated_video(
        self, source, plot_options=None, conf=0.5, progress_callback=None,
        tracking=False, tracker="bytetrack", classes=None,
    ):
        if tracking:
            return await self.predict_tracked_annotated_video(
                source, plot_options, conf, progress_callback, tracker, classes
            )
        video_path = await save_uploadfile_to_temp(source)
        try:
            if classes is None:
                return await run_in_threadpool(
                    self.model_repository.predict_annotated_video, video_path,
                    plot_options=plot_options, conf=conf,
                    progress_callback=progress_callback,
                )
            return await run_in_threadpool(
                self.model_repository.predict_annotated_video, video_path,
                plot_options=plot_options, conf=conf, classes=classes,
                progress_callback=progress_callback,
            )
        finally:
            os.remove(video_path)

    async def predict_tracked_annotated_video(
        self, source, plot_options=None, conf=0.5, progress_callback=None,
        tracker="bytetrack", classes=None,
    ):
        video_path = await save_uploadfile_to_temp(source)
        try:
            return await run_in_threadpool(
                self.tracking_service.track_annotated_video,
                video_path, plot_options, conf, tracker, progress_callback, classes,
            )
        finally:
            os.remove(video_path)

    async def predict_video_url(
        self, source_url: str, output: str = "json", plot_options=None, conf=0.5,
        progress_callback=None, tracking=False, tracker="bytetrack", classes=None,
    ):
        if tracking:
            return await self.predict_tracked_video_url(
                source_url, output, plot_options, conf, progress_callback, tracker, classes
            )
        video_path = await run_in_threadpool(download_video_to_temp, source_url)
        try:
            try:
                if classes is None:
                    if output == "video":
                        return await run_in_threadpool(
                            self.model_repository.predict_annotated_video, video_path,
                            plot_options=plot_options, conf=conf,
                            progress_callback=progress_callback,
                        )
                    return await self.predict_video_path(video_path, conf, progress_callback)
                if output == "video":
                    return await run_in_threadpool(
                        self.model_repository.predict_annotated_video, video_path,
                        plot_options=plot_options, conf=conf, classes=classes,
                        progress_callback=progress_callback,
                    )
                return await self.predict_video_path(video_path, conf, progress_callback, classes=classes)
            except ValueError as error:
                raise InvalidVideoSourceError(
                    "Video URL does not point to a readable video file."
                ) from error
        finally:
            os.remove(video_path)

    async def predict_tracked_video_url(
        self, source_url, output="json", plot_options=None, conf=0.5,
        progress_callback=None, tracker="bytetrack", classes=None,
    ):
        video_path = await run_in_threadpool(download_video_to_temp, source_url)
        try:
            if output == "video":
                return await run_in_threadpool(
                    self.tracking_service.track_annotated_video,
                    video_path, plot_options, conf, tracker, progress_callback, classes,
                )
            return await self.predict_tracked_video_path(
                video_path, conf, progress_callback, tracker, classes
            )
        finally:
            os.remove(video_path)
