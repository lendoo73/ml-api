from pydantic import BaseModel


class Segmentation(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    polygon: list[list[float]]
    track_id: int | None = None
    trajectory: list[tuple[int, int]] | None = None
