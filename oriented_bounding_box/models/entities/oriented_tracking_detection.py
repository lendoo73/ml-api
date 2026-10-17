from pydantic import Field

from .oriented_detection import OrientedDetection


class OrientedTrackingDetection(OrientedDetection):
    track_id: int | None = None
    trajectory: list[tuple[int, int]] | None = Field(default=None)
