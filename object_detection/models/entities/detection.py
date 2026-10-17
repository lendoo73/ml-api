from pydantic import BaseModel

from object_detection.models.entities.bounding_box import BoundingBox


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    trajectory: list[tuple[int, int]] | None = None
