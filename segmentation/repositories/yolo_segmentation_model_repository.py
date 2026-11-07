import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import threading
from typing import Literal
from ultralytics import YOLO

from segmentation.models.entities.segmentation import Segmentation


SegmentationModel = Literal["yolo26x-seg", "brain-tumor", "pothole"]
MODEL_WEIGHT_PATHS = {
    "yolo26x-seg": os.getenv(
        "YOLO_SEGMENTATION_MODEL",
        str(Path(__file__).resolve().parent.parent / "weights" / "yolo26x-seg.pt"),
    ),
    "brain-tumor": str(
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x-brain_tumor-seg.pt"
    ),
    "pothole": str(
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x_pothole_segmentation.pt"
    ),
}


class YoloSegmentationModelRepository:
    def __init__(
        self,
        model: SegmentationModel = "yolo26x-seg",
        model_path: str | None = None,
    ):
        self._model_path = model_path or MODEL_WEIGHT_PATHS[model]
        self._thread_models = threading.local()

    @property
    def model(self):
        model = getattr(self._thread_models, "model", None)
        if model is None:
            model = YOLO(self._model_path)
            self._thread_models.model = model
        return model

    def predict(
        self, source, conf: float = 0.5, classes: list[int] | None = None
    ) -> list[Segmentation]:
        return self._segmentations_from_result(self._predict_result(source, conf, classes))

    def predict_annotated_image(
        self,
        source,
        plot_options: dict | None = None,
        conf: float = 0.5,
        classes: list[int] | None = None,
    ) -> bytes:
        result = self._predict_result(source, conf, classes)
        annotated_image = result.plot(pil=True, **(plot_options or {}))
        success, encoded_image = cv2.imencode(
            ".jpg", cv2.cvtColor(np.asarray(annotated_image), cv2.COLOR_RGB2BGR)
        )
        if not success:
            raise RuntimeError("Could not encode annotated image")
        return encoded_image.tobytes()

    def predict_video(
        self, source, conf=0.5, progress_callback=None, classes: list[int] | None = None
    ) -> list[dict]:
        total_frames = self._probe_frame_count(source)
        frames = []
        options = {"classes": classes} if classes is not None else {}
        for frame_index, result in enumerate(
            self.model.predict(
                source=source, conf=conf, stream=True, verbose=False, **options
            )
        ):
            if progress_callback:
                progress_callback(frame_index + 1, total_frames)
            frames.append({
                "frame_index": frame_index,
                "segmentations": [
                    item.model_dump() for item in self._segmentations_from_result(result)
                ],
            })
        return frames

    def predict_annotated_video(
        self, source, plot_options=None, conf=0.5, progress_callback=None,
        classes: list[int] | None = None,
    ) -> str:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError("Could not open video source")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(prefix="segmentation_", suffix=".mp4", delete=False)
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
                annotated_frame = self._predict_result(frame, conf, classes).plot(
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

    def track_frame(self, source, conf=0.5, tracker="bytetrack", classes=None, **model_options):
        self._ensure_tracking_state()
        options = model_options.copy()
        if classes is not None:
            options["classes"] = classes
        result = self.model.track(
            source=source,
            conf=conf,
            tracker=f"{tracker}.yaml",
            persist=True,
            verbose=False,
            **options,
        )[0]
        return self._tracked_segmentations_from_result(result)

    def track_video(
        self, source, conf=0.5, tracker="bytetrack", progress_callback=None, classes=None
    ):
        total_frames = self._probe_frame_count(source)
        frames = []
        options = {"classes": classes} if classes is not None else {}
        self.reset_tracking()
        try:
            for index, result in enumerate(
                self.model.track(
                    source=source,
                    conf=conf,
                    tracker=f"{tracker}.yaml",
                    stream=True,
                    persist=True,
                    verbose=False,
                    **options,
                )
            ):
                if progress_callback:
                    progress_callback(index + 1, total_frames)
                frames.append({
                    "frame_index": index,
                    "segmentations": [
                        item.model_dump()
                        for item in self._tracked_segmentations_from_result(result)
                    ],
                })
        finally:
            self.reset_tracking()
        return frames

    def track_annotated_video(
        self, source, plot_options=None, conf=0.5, tracker="bytetrack",
        progress_callback=None, classes=None,
    ):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError("Could not open video source")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(prefix="segmentation_", suffix=".mp4", delete=False)
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
                result = self.model.track(
                    source=frame, conf=conf, tracker=f"{tracker}.yaml",
                    persist=True, verbose=False,
                    **({"classes": classes} if classes is not None else {}),
                )[0]
                tracked_segmentations = self._tracked_segmentations_from_result(result)
                annotated_frame = np.asarray(result.plot(pil=True, **(plot_options or {})))
                for segmentation in tracked_segmentations:
                    if segmentation.trajectory:
                        points = np.asarray(segmentation.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(255, 0, 0),
                            thickness=3,
                        )
                writer.write(cv2.cvtColor(np.asarray(annotated_frame), cv2.COLOR_RGB2BGR))
                frame_index += 1
                if progress_callback:
                    progress_callback(frame_index, total_frames)
        finally:
            capture.release()
            writer.release()
            self.reset_tracking()
        return output_path

    def reset_tracking(self):
        if getattr(self._thread_models, "model", None) is not None:
            self._thread_models.model.predictor = None
        self._thread_models.trajectories = defaultdict(list)

    def _ensure_tracking_state(self):
        if not hasattr(self._thread_models, "trajectories"):
            self._thread_models.trajectories = defaultdict(list)

    def _tracked_segmentations_from_result(self, result) -> list[Segmentation]:
        segmentations = self._segmentations_from_result(result)
        boxes = result.boxes
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(boxes)
        )
        tracked = []
        for item, coordinates, track_id in zip(segmentations, boxes.xyxy, track_ids):
            trajectory = None
            if track_id is not None:
                x_min, y_min, x_max, y_max = coordinates.tolist()
                center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                trajectory = self._thread_models.trajectories[track_id]
                trajectory.append(center)
                del trajectory[:-30]
            tracked.append(item.model_copy(update={
                "track_id": track_id,
                "trajectory": trajectory,
            }))
        return tracked

    def _predict_result(self, source, conf: float = 0.5, classes=None):
        return self.model.predict(
            source=source,
            conf=conf,
            verbose=False,
            **({"classes": classes} if classes is not None else {}),
        )[0]

    def resolve_class_ids(self, classes: list[int | str] | None) -> list[int] | None:
        if classes is None:
            return None

        model_names = {int(class_id): str(name) for class_id, name in self.model.names.items()}
        name_to_id = {name.casefold(): class_id for class_id, name in model_names.items()}
        resolved = []
        for value in classes:
            class_value = str(value).strip()
            if class_value.lstrip("-").isdigit():
                class_id = int(class_value)
                if class_id not in model_names:
                    raise ValueError(f"Unknown class ID: {class_value}.")
            else:
                class_id = name_to_id.get(class_value.casefold())
                if class_id is None:
                    raise ValueError(f"Unknown class name: {class_value}.")
            if class_id not in resolved:
                resolved.append(class_id)
        return resolved

    @staticmethod
    def _probe_frame_count(source: str):
        capture = cv2.VideoCapture(source)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        finally:
            capture.release()

    @staticmethod
    def _segmentations_from_result(result) -> list[Segmentation]:
        if result.masks is None:
            return []
        return [
            Segmentation(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
                polygon=polygon.tolist(),
            )
            for polygon, class_id, confidence in zip(
                result.masks.xy, result.boxes.cls, result.boxes.conf
            )
        ]
