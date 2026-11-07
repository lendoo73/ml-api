from pydantic import BaseModel, Field

from pose_estimation.models.dtos.pose_angle_dto import PoseAngleDto


class Pose(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    keypoints: list[list[float]]
    track_id: int | None = None
    trajectory: list[tuple[int, int]] | None = None
    angles: list[PoseAngleDto] | None = Field(default=None, exclude_if=lambda value: value is None)
