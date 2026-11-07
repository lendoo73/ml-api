import cv2
import numpy as np


class TrackingService:
    def __init__(self, model_repository):
        self.model_repository = model_repository

    def track_frame_bytes(self, frame_bytes: bytes, conf: float, tracker: str, classes=None):
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image frame")
        if classes is None:
            return self.model_repository.track_frame(frame, conf=conf, tracker=tracker)
        return self.model_repository.track_frame(
            frame, conf=conf, tracker=tracker, classes=classes
        )

    def track_video(self, source, conf: float, tracker: str, progress_callback=None, classes=None):
        options = {"classes": classes} if classes is not None else {}
        return self.model_repository.track_video(
            source, conf=conf, tracker=tracker, progress_callback=progress_callback, **options
        )

    def track_annotated_video(
        self, source, plot_options: dict | None, conf: float, tracker: str, progress_callback=None,
        classes=None,
    ):
        options = {"classes": classes} if classes is not None else {}
        return self.model_repository.track_annotated_video(
            source, plot_options=plot_options, conf=conf, tracker=tracker,
            progress_callback=progress_callback,
            **options,
        )

    def reset_tracking(self):
        self.model_repository.reset_tracking()
