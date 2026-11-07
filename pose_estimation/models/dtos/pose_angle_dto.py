from pydantic import BaseModel


class PoseAngleDto(BaseModel):
    joints: list[str]
    joint_indices: list[int]
    keypoint_confidences: list[float | None]
    degrees: float | None
    valid: bool
