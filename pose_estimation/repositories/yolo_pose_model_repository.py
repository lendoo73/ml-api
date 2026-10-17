import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pose_estimation.models.entities.pose import Pose


class YoloPoseModelRepository:
    def __init__(self, model_path: str | None = None):
        default_model_path = Path(__file__).resolve().parent.parent / "weights" / "yolo26x-pose.pt"
        self._model_path = model_path or os.getenv("YOLO_POSE_MODEL", str(default_model_path))
        self._thread_models = threading.local()

    @property
    def model(self):
        model = getattr(self._thread_models, "model", None)
        if model is None:
            model = YOLO(self._model_path)
            self._thread_models.model = model
        return model

    def predict(self, source, conf: float = 0.5) -> list[Pose]:
        return self._poses_from_result(self._predict_result(source, conf))

    def predict_annotated_image(
        self, source, plot_options: dict | None = None, conf: float = 0.5
    ) -> bytes:
        result = self._predict_result(source, conf)
        annotated_image = result.plot(pil=True, **(plot_options or {}))
        success, encoded_image = cv2.imencode(
            ".jpg", cv2.cvtColor(np.asarray(annotated_image), cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RuntimeError("Could not encode annotated image")
        return encoded_image.tobytes()

    def predict_video(self, source, conf=0.5, progress_callback=None) -> list[dict]:
        total_frames = self._probe_frame_count(source)
        frames = []
        for frame_index, result in enumerate(
            self.model.predict(source=source, conf=conf, stream=True, verbose=False)
        ):
            if progress_callback:
                progress_callback(frame_index + 1, total_frames)
            frames.append({
                "frame_index": frame_index,
                "poses": [pose.model_dump() for pose in self._poses_from_result(result)],
            })
        return frames

    def predict_annotated_video(self, source, plot_options=None, conf=0.5, progress_callback=None) -> str:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError("Could not open video source")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(prefix="pose_estimation_", suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                annotated_frame = self._predict_result(frame, conf).plot(
                    pil=True, **(plot_options or {})
                )
                writer.write(cv2.cvtColor(np.asarray(annotated_frame), cv2.COLOR_RGB2BGR))
                frame_index += 1
                if progress_callback:
                    progress_callback(frame_index, total_frames)
        finally:
            capture.release()
            writer.release()
        return output_path

    def track_frame(self, source, conf=0.5, tracker="bytetrack"):
        result = self.model.track(source=source, conf=conf, tracker=f"{tracker}.yaml", persist=True, verbose=False)[0]
        return self._tracked_poses_from_result(result)

    def track_video(self, source, conf=0.5, tracker="bytetrack", progress_callback=None):
        total_frames = self._probe_frame_count(source)
        frames = []
        for index, result in enumerate(self.model.track(source=source, conf=conf, tracker=f"{tracker}.yaml", stream=True, persist=True, verbose=False)):
            if progress_callback:
                progress_callback(index + 1, total_frames)
            frames.append({
                "frame_index": index,
                "poses": [item.model_dump() for item in self._tracked_poses_from_result(result)],
            })
        self.reset_tracking()
        return frames

    def track_annotated_video(self, source, plot_options=None, conf=0.5, tracker="bytetrack", progress_callback=None):
        return self.predict_annotated_video(source, plot_options, conf, progress_callback)

    def reset_tracking(self):
        if getattr(self._thread_models, "model", None) is not None:
            self._thread_models.model.predictor = None

    def _predict_result(self, source, conf: float = 0.5):
        return self.model.predict(source=source, conf=conf, verbose=False)[0]

    @staticmethod
    def _poses_from_result(result) -> list[Pose]:
        if result.keypoints is None:
            return []
        keypoints = result.keypoints.data.tolist()
        return [
            Pose(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
                keypoints=points,
            )
            for points, class_id, confidence in zip(
                keypoints, result.boxes.cls, result.boxes.conf
            )
        ]

    @classmethod
    def _tracked_poses_from_result(cls, result) -> list[Pose]:
        poses = cls._poses_from_result(result)
        if result.boxes.id is None:
            return poses
        track_ids = result.boxes.id.int().cpu().tolist()
        return [
            pose.model_copy(update={"track_id": track_id})
            for pose, track_id in zip(poses, track_ids)
        ]

    @staticmethod
    def _probe_frame_count(source: str):
        capture = cv2.VideoCapture(source)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        finally:
            capture.release()
