from pydantic import BaseModel, Field


class OrientedGeometry(BaseModel):
    """OBB angle is in radians, counter-clockwise, as provided by Ultralytics xywhr."""
    x: float
    y: float
    width: float
    height: float
    angle: float
    corners: list[list[float]] = Field(default_factory=list)


class OrientedDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    obb: OrientedGeometry
