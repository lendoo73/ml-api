import json
import math
from typing import TypeAlias

from pose_estimation.models.dtos.pose_angle_dto import PoseAngleDto
from pose_estimation.models.entities.pose import Pose


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KEYPOINT_INDEX_BY_NAME = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}
AngleDefinition: TypeAlias = tuple[int, int, int]


class PoseAngleService:
    @staticmethod
    def parse_requests(raw_angles: str | None) -> list[AngleDefinition] | None:
        if raw_angles is None or raw_angles == "":
            return None

        try:
            requests = json.loads(raw_angles)
        except json.JSONDecodeError as error:
            raise ValueError("angles must be a JSON-encoded list of keypoint triplets.") from error

        if not isinstance(requests, list):
            raise ValueError("angles must be a list of keypoint triplets.")

        definitions = [
            PoseAngleService._parse_triplet(request, request_index)
            for request_index, request in enumerate(requests)
        ]
        return definitions or None

    @staticmethod
    def _parse_triplet(request: object, request_index: int) -> AngleDefinition:
        if not isinstance(request, list) or len(request) != 3:
            raise ValueError(f"angles[{request_index}] must contain exactly three keypoints.")

        is_name_triplet = all(isinstance(item, str) for item in request)
        is_index_triplet = all(type(item) is int for item in request)
        if not is_name_triplet and not is_index_triplet:
            raise ValueError(f"angles[{request_index}] must use either three names or three indices.")

        if is_name_triplet:
            unknown_names = [name for name in request if name not in KEYPOINT_INDEX_BY_NAME]
            if unknown_names:
                raise ValueError(
                    f"angles[{request_index}] contains unknown keypoint name(s): "
                    f"{', '.join(unknown_names)}."
                )
            indices = tuple(KEYPOINT_INDEX_BY_NAME[name] for name in request)
        else:
            if any(index < 0 or index >= len(COCO_KEYPOINT_NAMES) for index in request):
                raise ValueError(
                    f"angles[{request_index}] indices must be between 0 and "
                    f"{len(COCO_KEYPOINT_NAMES) - 1}."
                )
            indices = tuple(request)

        if len(set(indices)) != 3:
            raise ValueError(f"angles[{request_index}] must use three distinct keypoints.")
        return indices

    @staticmethod
    def add_angles(
        poses: list[Pose],
        definitions: list[AngleDefinition] | None,
    ) -> list[Pose]:
        if not definitions:
            return poses
        return [
            pose.model_copy(
                update={
                    "angles": [
                        PoseAngleService._calculate_angle(pose.keypoints, indices)
                        for indices in definitions
                    ]
                }
            )
            for pose in poses
        ]

    @staticmethod
    def add_angles_to_video_frames(
        frames: list[dict],
        definitions: list[AngleDefinition] | None,
    ) -> list[dict]:
        if not definitions:
            return frames
        for frame in frames:
            poses = [Pose.model_validate(pose_data) for pose_data in frame["poses"]]
            frame["poses"] = [
                pose.model_dump()
                for pose in PoseAngleService.add_angles(poses, definitions)
            ]
        return frames

    @staticmethod
    def _calculate_angle(
        keypoints: list[list[float]],
        indices: AngleDefinition,
    ) -> PoseAngleDto:
        names = [COCO_KEYPOINT_NAMES[index] for index in indices]
        confidences = []
        for index in indices:
            confidence = (
                keypoints[index][2]
                if index < len(keypoints) and len(keypoints[index]) > 2
                else None
            )
            confidences.append(
                confidence if confidence is not None and math.isfinite(confidence) else None
            )
        points = [keypoints[index] if index < len(keypoints) else [] for index in indices]

        valid = all(
            len(point) >= 2
            and math.isfinite(point[0])
            and math.isfinite(point[1])
            for point in points
        )
        degrees = None

        if valid:
            first, pivot, third = points
            first_vector = (first[0] - pivot[0], first[1] - pivot[1])
            second_vector = (third[0] - pivot[0], third[1] - pivot[1])
            if first_vector != (0, 0) and second_vector != (0, 0):
                cross_product = (
                    first_vector[0] * second_vector[1]
                    - first_vector[1] * second_vector[0]
                )
                dot_product = (
                    first_vector[0] * second_vector[0]
                    + first_vector[1] * second_vector[1]
                )
                degrees = round(
                    math.degrees(math.atan2(cross_product, dot_product)),
                    2,
                ) % 360
            else:
                valid = False

        return PoseAngleDto(
            joints=names,
            joint_indices=list(indices),
            keypoint_confidences=confidences,
            degrees=degrees,
            valid=valid,
        )
