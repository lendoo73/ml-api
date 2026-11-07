import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ultralytics import YOLO

from classification.models.entities.classification import Classification


default_model_path = Path(__file__).resolve().parent.parent / "weights" / "yolo26x-cls.pt"
ClassificationModel = Literal["yolo26x-cls", "chest-xray", "weather"]
MODEL_WEIGHT_PATHS = {
    "yolo26x-cls": default_model_path,
    "chest-xray": (
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x_chest_xray-cls.pt"
    ),
    "weather": (
        Path(__file__).resolve().parent.parent / "weights" / "yolo26x_weather_cls.pt"
    ),
}

class YoloClassificationModelRepository:
    def __init__(
        self,
        model: ClassificationModel = "yolo26x-cls",
        model_path: str | Path | None = None,
    ):
        self._model_path = str(model_path or MODEL_WEIGHT_PATHS[model])
        self._thread_models = threading.local()

    @property
    def model(self):
        model = getattr(self._thread_models, "model", None)
        if model is None:
            model = YOLO(self._model_path)
            self._thread_models.model = model
        return model

    def predict(self, source, conf: float = 0.5) -> list[Classification]:
        return self._classifications_from_result(
            self._predict_result(source, conf), conf
        )

    def predict_annotated_image(
        self, source, plot_options: dict | None = None, conf: float = 0.5
    ) -> bytes:
        result = self._predict_result(source, conf)
        annotated_image = result.plot(pil=True, **(plot_options or {}))
        annotated_image = cv2.cvtColor(np.asarray(annotated_image), cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".jpg", annotated_image)
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
                "classifications": [
                    classification.model_dump()
                    for classification in self._classifications_from_result(result)
                ],
            })
        return frames

    def predict_annotated_video(self, source, plot_options=None, conf=0.5, progress_callback=None) -> str:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError("Could not open video source")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(
            prefix="classification_",
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

    def _predict_result(self, source, conf: float = 0.5):
        return self.model.predict(source=source, conf=conf, verbose=False)[0]

    @staticmethod
    def _probe_frame_count(source: str):
        capture = cv2.VideoCapture(source)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        finally:
            capture.release()

    @staticmethod
    def _classifications_from_result(
        result, conf: float = 0.5
    ) -> list[Classification]:
        probabilities = result.probs
        class_ids = probabilities.top5
        confidences = probabilities.top5conf
        return [
            Classification(
                class_id=int(class_id),
                class_name=result.names[int(class_id)],
                confidence=float(confidence),
            )
            for class_id, confidence in zip(class_ids, confidences)
            if float(confidence) >= conf
        ]
