import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Literal, Type

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from classification.services.classification_service import ClassificationService
from classification.services.chest_xray_classification_service import (
    ChestXrayClassificationService,
)
from classification.services.weather_classification_service import WeatherClassificationService
from object_detection.services.object_detection_service import ObjectDetectionService
from oriented_bounding_box.services.car_parking_lot_service import CarParkingLotService
from oriented_bounding_box.services.oriented_bounding_box_service import (
    OrientedBoundingBoxService,
)
from pose_estimation.services.pose_angle_service import (
    PoseAngleService,
)
from pose_estimation.services.pose_estimation_service import PoseEstimationService
from segmentation.services.brain_tumor_segmentation_service import (
    BrainTumorSegmentationService,
)
from segmentation.services.pothole_segmentation_service import PotholeSegmentationService
from segmentation.services.segmentation_service import SegmentationService


class ModelId(str, Enum):
    OBJECT_DETECTION = "object-detection"
    ROAD_SIGN_DETECTION = "road-sign-detection"
    FACE_MASK_DETECTION = "face-mask-detection"
    POSE_ESTIMATION = "pose-estimation"
    SEGMENTATION = "segmentation"
    POTHOLE_SEGMENTATION = "pothole-segmentation"
    BRAIN_TUMOR_SEGMENTATION = "brain-tumor-segmentation"
    CLASSIFICATION = "classification"
    CHEST_XRAY_CLASSIFICATION = "chest-xray-classification"
    WEATHER_CLASSIFICATION = "weather-classification"
    ORIENTED_BOUNDING_BOX = "oriented-bounding-box"
    CAR_PARKING_LOT = "car-parking-lot"


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectionOptions(Options):
    conf: float = Field(0.5, ge=0, le=1)
    classes: list[int | str] | None = None
    tracking: bool = False
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"


class PoseOptions(Options):
    conf: float = Field(0.5, ge=0, le=1)
    tracking: bool = False
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"
    angles: list[list[int | str]] | None = None


class SegmentationOptions(DetectionOptions):
    pass


class PotholeOptions(Options):
    conf: float = Field(0.5, ge=0, le=1)
    imgsz: int = Field(640, gt=0)
    classes: list[int] | None = None
    max_det: int = Field(300, gt=0)
    retina_masks: bool = False
    tracking: bool = False
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"


class ClassificationOptions(Options):
    conf: float = Field(0.5, ge=0, le=1)


class OrientedBoxOptions(ClassificationOptions):
    tracking: bool = False
    tracker: Literal["bytetrack", "botsort"] = "bytetrack"


@dataclass(frozen=True)
class ModelSpec:
    service_class: Type
    options_class: Type[Options]
    task: str
    service_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_SPECS = {
    ModelId.OBJECT_DETECTION.value: ModelSpec(
        ObjectDetectionService, DetectionOptions, "detection", {"model": "yolo26x"}
    ),
    ModelId.ROAD_SIGN_DETECTION.value: ModelSpec(
        ObjectDetectionService, DetectionOptions, "detection", {"model": "road-sign-detect"}
    ),
    ModelId.FACE_MASK_DETECTION.value: ModelSpec(
        ObjectDetectionService, DetectionOptions, "detection", {"model": "face-mask-detect"}
    ),
    ModelId.POSE_ESTIMATION.value: ModelSpec(
        PoseEstimationService, PoseOptions, "pose"
    ),
    ModelId.SEGMENTATION.value: ModelSpec(
        SegmentationService, SegmentationOptions, "segmentation"
    ),
    ModelId.POTHOLE_SEGMENTATION.value: ModelSpec(
        PotholeSegmentationService, PotholeOptions, "pothole"
    ),
    ModelId.BRAIN_TUMOR_SEGMENTATION.value: ModelSpec(
        BrainTumorSegmentationService, SegmentationOptions, "segmentation"
    ),
    ModelId.CLASSIFICATION.value: ModelSpec(
        ClassificationService, ClassificationOptions, "standard"
    ),
    ModelId.CHEST_XRAY_CLASSIFICATION.value: ModelSpec(
        ChestXrayClassificationService, ClassificationOptions, "standard"
    ),
    ModelId.WEATHER_CLASSIFICATION.value: ModelSpec(
        WeatherClassificationService, ClassificationOptions, "standard"
    ),
    ModelId.ORIENTED_BOUNDING_BOX.value: ModelSpec(
        OrientedBoundingBoxService, OrientedBoxOptions, "oriented"
    ),
    ModelId.CAR_PARKING_LOT.value: ModelSpec(
        CarParkingLotService, OrientedBoxOptions, "oriented"
    ),
}


@lru_cache(maxsize=None)
def _cached_service(model_id: str):
    spec = MODEL_SPECS[model_id]
    return spec.service_class(**spec.service_kwargs)


@lru_cache(maxsize=None)
def _cached_executor(model_id: str):
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"multi-{model_id}")


def parse_configuration(
    models: list[str], model_options: dict[str, dict[str, Any]] | None
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not models:
        raise ValueError("models must contain at least one supported model.")
    if len(set(models)) != len(models):
        raise ValueError("models must not contain duplicates.")

    raw_options = model_options or {}
    unknown_models = set(models) - MODEL_SPECS.keys()
    if unknown_models:
        raise ValueError(f"Unsupported model identifier(s): {', '.join(sorted(unknown_models))}.")
    unselected_options = set(raw_options) - set(models)
    if unselected_options:
        raise ValueError(
            "model_options contains unselected model(s): "
            f"{', '.join(sorted(unselected_options))}."
        )

    validated = {}
    for model_id in models:
        try:
            validated[model_id] = MODEL_SPECS[model_id].options_class.model_validate(
                raw_options.get(model_id, {})
            ).model_dump()
        except ValidationError as error:
            raise ValueError(f"Invalid options for '{model_id}': {error}") from error
    return models, validated


class MultiModelPredictionService:
    def __init__(self, models: list[str], model_options: dict[str, dict[str, Any]]):
        self.models = models
        self.options = model_options
        self.services = {}
        self.executors = {}
        self.owned_executors: list[ThreadPoolExecutor] = []
        self._closed = False
        self._prepare()

    def _prepare(self):
        for model_id in self.models:
            options = self.options[model_id]
            spec = MODEL_SPECS[model_id]
            tracking = options.get("tracking", False) or (
                spec.task == "pose" and options.get("angles")
            )
            service = spec.service_class() if tracking else _cached_service(model_id)
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"multi-session-{model_id}"
            ) if tracking else _cached_executor(model_id)
            self.services[model_id] = service
            self.executors[model_id] = executor
            if tracking:
                self.owned_executors.append(executor)

            if spec.task in {"detection", "segmentation", "pothole"}:
                classes = options.get("classes")
                if classes is not None:
                    try:
                        options["classes"] = executor.submit(
                            service.model_repository.resolve_class_ids, classes
                        ).result()
                    except ValueError as error:
                        self.close()
                        raise ValueError(f"Invalid classes for '{model_id}': {error}") from error
            if spec.task == "pose":
                try:
                    options["_angles"] = PoseAngleService.parse_requests(
                        json.dumps(options.get("angles")) if options.get("angles") is not None else None
                    )
                except ValueError as error:
                    self.close()
                    raise ValueError(f"Invalid options for '{model_id}': {error}") from error
                if options.get("tracking") or options.get("_angles"):
                    options["tracking"] = True

    async def predict_frame(self, frame: np.ndarray) -> dict[str, list[dict]]:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self.executors[model_id], self._predict_one, model_id, frame
            )
            for model_id in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return {
            model_id: [item.model_dump() for item in result]
            for model_id, result in zip(self.models, results)
        }

    def _predict_one(self, model_id: str, frame: np.ndarray):
        service = self.services[model_id]
        options = self.options[model_id]
        task = MODEL_SPECS[model_id].task
        conf = options["conf"]

        if task == "pose":
            poses = (
                service.model_repository.track_frame(frame, conf, options["tracker"])
                if options["tracking"]
                else service.model_repository.predict(frame, conf)
            )
            return service.pose_angle_service.add_angles(poses, options["_angles"])
        if task in {"detection", "segmentation", "pothole"} and options.get("tracking"):
            tracking_options = {}
            if task == "pothole":
                tracking_options = {
                    "imgsz": options["imgsz"],
                    "max_det": options["max_det"],
                    "retina_masks": options["retina_masks"],
                }
            return service.model_repository.track_frame(
                frame,
                conf,
                options["tracker"],
                **({"classes": options.get("classes")} if options.get("classes") is not None else {}),
                **tracking_options,
            )
        if task == "oriented" and options.get("tracking"):
            return service.model_repository.track(frame, conf, options["tracker"])
        if task == "pothole":
            return service.predict_frame(
                frame,
                conf=conf,
                imgsz=options["imgsz"],
                classes=options.get("classes"),
                max_det=options["max_det"],
                retina_masks=options["retina_masks"],
            )
        if task in {"detection", "segmentation"}:
            return service.predict_frame(
                frame, conf=conf, classes=options.get("classes")
            )
        return service.predict_frame(frame, conf=conf)

    async def predict_video(self, video_path: str, progress_callback=None) -> list[dict]:
        reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="multi-video-reader")
        loop = asyncio.get_running_loop()
        capture = await loop.run_in_executor(reader, cv2.VideoCapture, video_path)
        if not capture.isOpened():
            await loop.run_in_executor(reader, capture.release)
            reader.shutdown(wait=True)
            self.close()
            raise ValueError("Could not open video source.")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        frames = []
        try:
            frame_index = 0
            while True:
                success, frame = await loop.run_in_executor(reader, capture.read)
                if not success:
                    break
                predictions = await self.predict_frame(frame)
                frames.append({
                    "frame_index": frame_index,
                    "predictions": predictions,
                })
                frame_index += 1
                if progress_callback:
                    progress_callback(frame_index, total_frames)
        finally:
            await loop.run_in_executor(reader, capture.release)
            reader.shutdown(wait=True)
            self.close()
        return frames

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for model_id, service in self.services.items():
                if self.options.get(model_id, {}).get("tracking"):
                    repository = service.model_repository
                    reset = getattr(repository, "reset_tracking", None)
                    if reset:
                        self.executors[model_id].submit(reset).result()
        finally:
            for executor in self.owned_executors:
                executor.shutdown(wait=True)
            self.owned_executors.clear()
