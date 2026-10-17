from pydantic import BaseModel

from object_detection.models.dtos.bounding_box_dto import BoundingBoxDto


class DetectionDto(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBoxDto
    track_id: int | None = None
    trajectory: list[tuple[int, int]] | None = None
