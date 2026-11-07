import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Literal
import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from object_detection.models.entities.detection import Detection


DetectionModel = Literal["yolo26x", "face-mask-detect", "road-sign-detect", "knife"]
MODEL_WEIGHT_PATHS = {
    "yolo26x": Path(__file__).resolve().parent.parent / "weights" / "yolo26x.pt",
    "face-mask-detect": (
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x_face_mask_detect.pt"
    ),
    "road-sign-detect": (
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x_road_sign-detect.pt"
    ),
    "knife": Path(__file__).resolve().parent.parent / "weights" / "yolo26s-knife.pt",
}


class YoloModelRepository:
    def __init__(self, model_path: str | None = None, *, model: DetectionModel = "yolo26x"):
        self._model_path = model_path or str(MODEL_WEIGHT_PATHS[model])
        self._thread_models = threading.local()

    @property
    def model(self):
        model = getattr(self._thread_models, "model", None)
        if model is None:
            model = YOLO(self._model_path)
            self._thread_models.model = model
        return model

    def predict(
        self, source, conf: float = 0.5, classes: list[int | str] | None = None
    ) -> list[Detection]:
        result = (
            self._predict_result(source, conf)
            if classes is None
            else self._predict_result(source, conf, classes)
        )
        return [
            Detection(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
                bbox=self._to_bounding_box(coordinates),
            )
            for coordinates, class_id, confidence in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf,
            )
        ]

    def predict_annotated_image(
        self, source, plot_options: dict | None = None, conf: float = 0.5,
        classes: list[int | str] | None = None,
    ) -> bytes:
        result = (
            self._predict_result(source, conf)
            if classes is None
            else self._predict_result(source, conf, classes)
        )
        options = dict(plot_options or {})
        if options.get("line_width") is not None:
            options["line_width"] = max(1, round(options["line_width"]))
        annotated_image = self._plot_result(result, options)
        annotated_image = cv2.cvtColor(np.asarray(annotated_image), cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".jpg", annotated_image)
        if not success:
            raise RuntimeError("Could not encode annotated image")
        return encoded_image.tobytes()

    def _predict_result(self, source, conf: float = 0.5, classes: list[int | str] | None = None):
        classes = self.resolve_class_ids(classes)
        options = {"classes": classes} if classes is not None else {}
        return self.model.predict(source=source, conf=conf, verbose=False, **options)[0]

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

    def track_frame(
        self, source, conf: float, tracker: str, trajectory_length: int = 30,
        classes: list[int | str] | None = None,
    ) -> list[Detection]:
        classes = self.resolve_class_ids(classes)
        self._ensure_tracking_state()
        result = self.model.track(
            source=source,
            conf=conf,
            tracker=self._tracker_config(tracker),
            persist=True,
            verbose=False,
            **({"classes": classes} if classes is not None else {}),
        )[0]
        return self._tracked_detections(result, trajectory_length)

    def track_video(
        self, source, conf: float, tracker: str, trajectory_length: int = 30,
        progress_callback=None, classes: list[int | str] | None = None,
    ) -> list[dict]:
        classes = self.resolve_class_ids(classes)
        fps = self._probe_fps(source)
        total_frames = self._probe_frame_count(source)
        frames = []
        self.reset_tracking()
        try:
            for frame_index, result in enumerate(
                self.model.track(
                    source=source,
                    conf=conf,
                    tracker=self._tracker_config(tracker),
                    persist=True,
                    stream=True,
                    verbose=False,
                    **({"classes": classes} if classes is not None else {}),
                )
            ):
                if progress_callback:
                    progress_callback(frame_index + 1, total_frames)
                frames.append({
                    "frame_index": frame_index,
                    "timestamp_ms": int(frame_index * 1000 / fps) if fps else None,
                    "detections": [
                        detection.model_dump()
                        for detection in self._tracked_detections(result, trajectory_length)
                    ],
                })
        finally:
            self.reset_tracking()
        return frames

    def track_annotated_video(
        self,
        source,
        plot_options=None,
        conf=0.5,
        tracker="bytetrack",
        trajectory_length=30,
        progress_callback=None,
        classes: list[int | str] | None = None,
    ) -> str:
        classes = self.resolve_class_ids(classes)
        self.reset_tracking()
        try:
            return self._process_annotated_video(
                source,
                plot_options,
                conf,
                progress_callback,
                lambda frame: self.model.track(
                    source=frame,
                    conf=conf,
                    tracker=self._tracker_config(tracker),
                    persist=True,
                    verbose=False,
                    **({"classes": classes} if classes is not None else {}),
                )[0],
                trajectory_length,
            )
        finally:
            self.reset_tracking()

    def reset_tracking(self):
        model = getattr(self._thread_models, "model", None)
        if model is not None:
            model.predictor = None
        self._thread_models.trajectories = defaultdict(list)

    def _ensure_tracking_state(self):
        if not hasattr(self._thread_models, "trajectories"):
            self._thread_models.trajectories = defaultdict(list)

    @staticmethod
    def _tracker_config(tracker: str) -> str:
        configurations = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
        try:
            return configurations[tracker]
        except KeyError as error:
            raise ValueError(f"Unsupported tracker: {tracker}") from error

    @staticmethod
    def _to_bounding_box(coordinates):
        from object_detection.models.entities.bounding_box import BoundingBox

        return BoundingBox.from_xyxy(coordinates.tolist())

    def predict_video(
        self, source, conf=0.5, progress_callback=None, classes: list[int | str] | None = None
    ) -> list[dict]:
        classes = self.resolve_class_ids(classes)
        fps = self._probe_fps(source)
        total_frames = self._probe_frame_count(source)
        frames = []
        for frame_index, result in enumerate(
            self.model.predict(
                source=source, conf=conf, stream=True, verbose=False,
                **({"classes": classes} if classes is not None else {}),
            )
        ):
            if progress_callback:
                progress_callback(frame_index + 1, total_frames)
            frames.append({
                "frame_index": frame_index,
                "timestamp_ms": int(frame_index * 1000 / fps) if fps else None,
                "detections": [d.model_dump() for d in self._detections_from_result(result)],
            })
        return frames

    def predict_annotated_video(
        self, source, plot_options=None, conf=0.5, progress_callback=None,
        classes: list[int | str] | None = None,
    ) -> str:
        return self._process_annotated_video(
            source,
            plot_options,
            conf,
            progress_callback,
            lambda frame: self._predict_result(frame, conf, classes),
        )

    def _process_annotated_video(
        self, source, plot_options, conf, progress_callback, result_factory, trajectory_length=None
    ) -> str:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError("Could not open video source")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(
            prefix="object_detection_",
            suffix=".mp4",
            delete=False,
        )
        output_path = output_file.name
        output_file.close()
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if not writer.isOpened():
            capture.release()
            os.remove(output_path)
            raise RuntimeError("Could not initialize video writer")

        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                result = result_factory(frame)
                annotated_frame = self._plot_result(result, plot_options or {})
                if trajectory_length is not None:
                    for detection in self._tracked_detections(result, trajectory_length):
                        if detection.trajectory:
                            points = np.asarray(detection.trajectory, dtype=np.int32).reshape((-1, 1, 2))
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

        return output_path

    @staticmethod
    def _plot_result(result, options):
        txt_color = options.get("txt_color")
        if txt_color is None:
            return result.plot(pil=True, **options)
        annotated_image = result.plot(pil=True, **{**options, "labels": False})
        annotator = Annotator(
            annotated_image,
            line_width=options.get("line_width"),
            font_size=options.get("font_size"),
            font=options.get("font", "Arial.ttf"),
            pil=True,
            example=result.names,
        )
        # Ultralytics otherwise replaces custom text colors for predefined box colors.
        annotator.dark_colors = set()
        annotator.light_colors = set()
        boxes = getattr(result, "obb", None)
        if boxes is None:
            boxes = getattr(result, "boxes", None)
        if boxes is not None and options.get("boxes", True) and options.get("labels", True):
            for index, box in enumerate(reversed(boxes)):
                class_id = int(box.cls.item())
                confidence = float(box.conf.item()) if options.get("conf", True) else None
                label = result.names[class_id]
                if confidence is not None:
                    label = f"{label} {confidence:.2f}"
                coordinates = box.xyxy.squeeze().tolist()
                annotator.box_label(
                    coordinates,
                    label,
                    color=colors(class_id, True),
                    txt_color=txt_color,
                )
        return annotator.result()

    def _detections_from_result(self, result) -> list[Detection]:
        return [
            Detection(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
                bbox=self._to_bounding_box(coordinates),
            )
            for coordinates, class_id, confidence in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            )
        ]

    def _tracked_detections(self, result, trajectory_length: int) -> list[Detection]:
        self._ensure_tracking_state()
        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        detections = []
        for coordinates, class_id, confidence, track_id in zip(
            boxes.xyxy, boxes.cls, boxes.conf, track_ids
        ):
            trajectory = None
            if track_id is not None:
                bbox = self._to_bounding_box(coordinates)
                center = ((bbox.x_min + bbox.x_max) // 2, (bbox.y_min + bbox.y_max) // 2)
                trajectory = self._thread_models.trajectories[track_id]
                trajectory.append(center)
                del trajectory[:-trajectory_length]
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=result.names[int(class_id)],
                    confidence=float(confidence),
                    bbox=self._to_bounding_box(coordinates),
                    track_id=track_id,
                    trajectory=trajectory,
                )
            )
        return detections

    @staticmethod
    def _probe_fps(source: str):
        capture = cv2.VideoCapture(source)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            return float(fps) if fps > 0 else None
        finally:
            capture.release()

    @staticmethod
    def _probe_frame_count(source: str):
        capture = cv2.VideoCapture(source)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        finally:
            capture.release()
