import pytest

from pose_estimation.models.entities.pose import Pose
from pose_estimation.services.pose_angle_service import PoseAngleService


def _keypoints():
    return [[0.0, 0.0, 0.9] for _ in range(17)]


def test_pose_angle_service_resolves_names_and_indices():
    service = PoseAngleService()

    named = service.parse_requests(
        '[["left_shoulder","left_elbow","left_wrist"]]'
    )
    indexed = service.parse_requests("[[5,7,9]]")

    assert named == [(5, 7, 9)]
    assert indexed == named


@pytest.mark.parametrize(
    "raw_angles",
    [
        "not-json",
        "{}",
        "[[5, 7]]",
        '[["left_shoulder",7,"left_wrist"]]',
        "[[5, 7, 17]]",
        '[["left_shoulder","left_elbow","left_elbow"]]',
        "[[true, 7, 9]]",
    ],
)
def test_pose_angle_service_rejects_invalid_definitions(raw_angles):
    with pytest.raises(ValueError):
        PoseAngleService.parse_requests(raw_angles)


def test_pose_angle_service_calculates_clockwise_angle_in_image_coordinates():
    keypoints = _keypoints()
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]

    angle = PoseAngleService._calculate_angle(keypoints, (5, 7, 9))

    assert angle.degrees == 270.0
    assert angle.valid is True
    assert angle.joints == ["left_shoulder", "left_elbow", "left_wrist"]
    assert angle.joint_indices == [5, 7, 9]


def test_pose_angle_service_calculates_angle_for_low_confidence_keypoints():
    keypoints = _keypoints()
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.1]
    keypoints[9] = [1.0, 1.0, 0.9]

    angle = PoseAngleService._calculate_angle(keypoints, (5, 7, 9))

    assert angle.degrees == 270.0
    assert angle.valid is True
    assert angle.keypoint_confidences == [0.9, 0.1, 0.9]


def test_pose_angle_service_calculates_angle_when_keypoint_confidence_is_missing():
    keypoints = _keypoints()
    keypoints[5] = [0.0, 0.0]
    keypoints[7] = [1.0, 0.0]
    keypoints[9] = [1.0, 1.0]

    angle = PoseAngleService._calculate_angle(keypoints, (5, 7, 9))

    assert angle.degrees == 270.0
    assert angle.valid is True
    assert angle.keypoint_confidences == [None, None, None]


def test_pose_angle_service_marks_degenerate_vectors_unavailable():
    keypoints = _keypoints()
    keypoints[5] = [1.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]

    angle = PoseAngleService._calculate_angle(keypoints, (5, 7, 9))

    assert angle.degrees is None
    assert angle.valid is False


def test_pose_without_requested_angles_keeps_existing_serialized_shape():
    pose = Pose(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=_keypoints(),
    )

    assert "angles" not in pose.model_dump()


def test_pose_angle_service_adds_results_per_pose():
    pose = Pose(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=_keypoints(),
    )
    keypoints = [point[:] for point in pose.keypoints]
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]
    pose = pose.model_copy(update={"keypoints": keypoints})

    result = PoseAngleService.add_angles([pose], [(5, 7, 9)])

    assert result[0].angles[0].degrees == 270.0
    assert result[0].model_dump()["angles"][0]["degrees"] == 270.0


def test_pose_angle_service_adds_results_to_video_frames_and_preserves_track_id():
    pose = Pose(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=_keypoints(),
        track_id=23,
    )
    keypoints = [point[:] for point in pose.keypoints]
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]
    frame = {"frame_index": 0, "poses": [pose.model_copy(update={"keypoints": keypoints}).model_dump()]}

    frames = PoseAngleService.add_angles_to_video_frames([frame], [(5, 7, 9)])

    assert frames[0]["poses"][0]["track_id"] == 23
    assert frames[0]["poses"][0]["angles"][0]["degrees"] == 270.0
