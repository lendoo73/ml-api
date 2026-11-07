from typing import Any

from pydantic import BaseModel, Field

from classification.models.dtos.classification_dto import ClassificationDto
from object_detection.models.dtos.detection_dto import DetectionDto
from oriented_bounding_box.models.dtos.car_parking_lot_dto import CarParkingLotDto
from oriented_bounding_box.models.dtos.oriented_bounding_box_dto import (
    OrientedBoundingBoxDto,
)
from oriented_bounding_box.models.dtos.oriented_tracking_dto import OrientedTrackingDto
from pose_estimation.models.dtos.pose_dto import PoseDto
from segmentation.models.dtos.brain_tumor_segmentation_dto import (
    BrainTumorSegmentationDto,
)
from segmentation.models.dtos.pothole_segmentation_dto import PotholeSegmentationDto
from segmentation.models.dtos.segmentation_dto import SegmentationDto


PredictionDto = (
    ClassificationDto
    | DetectionDto
    | PoseDto
    | SegmentationDto
    | BrainTumorSegmentationDto
    | PotholeSegmentationDto
    | OrientedBoundingBoxDto
    | OrientedTrackingDto
    | CarParkingLotDto
)


class MultiModelImagePredictionDto(BaseModel):
    predictions: dict[str, list[PredictionDto]]


class MultiModelVideoFrameDto(BaseModel):
    frame_index: int
    predictions: dict[str, list[PredictionDto]]


class MultiModelJobDto(BaseModel):
    job_id: str


class WebcamPredictionConfigDto(BaseModel):
    models: list[str]
    model_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
