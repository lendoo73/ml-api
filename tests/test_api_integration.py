import asyncio
import json
import os
from pathlib import Path
import inspect
import shutil
import tempfile

import cv2
import pytest
from fastapi.testclient import TestClient

from main import app
from pose_estimation.controllers.pose_estimation_rest_controller import get_service
from pose_estimation.models.entities.pose import Pose
from pose_estimation.services.pose_angle_service import PoseAngleService


FIXTURES = Path(__file__).parent / "fixtures"
IMAGE_PATH = FIXTURES / "img" / "horses.png"
VIDEO_PATH = FIXTURES / "video" / "suspicious.mp4"
REMOTE_IMAGE_URL = (
    "https://github.com/ultralytics/ultralytics/blob/main/"
    "ultralytics/assets/bus.jpg?raw=true"
)
REMOTE_VIDEO_URL = "https://samplelib.com/mp4/sample-5s-360p.mp4"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def short_video():
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    assert capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
        output_path = output.name
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for _ in range(3):
            success, frame = capture.read()
            if not success:
                break
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    yield Path(output_path)
    Path(output_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("endpoint", "response_key"),
    [
        ("/api/object_detection/predict", "class_name"),
        ("/api/classification/predict", "class_name"),
        ("/api/segmentation/predict", "class_name"),
    ],
)
def test_image_prediction_endpoints_return_predictions(client, endpoint, response_key):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            endpoint,
            files={"source": (
                IMAGE_PATH.name,
                image,
                "image/png",
            )},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)
    if payload:
        assert response_key in payload[0]
        assert isinstance(payload[0][response_key], str)


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/object_detection/predict",
        "/api/classification/predict",
        "/api/segmentation/predict",
    ],
)
def test_image_endpoints_return_rendered_images(client, endpoint):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            f"{endpoint}?output=image&font_size=24&font=custom.ttf&show=false",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content.startswith(b"\xff\xd8\xff")


def test_segmentation_webcam_returns_segmentations(client):
    with IMAGE_PATH.open("rb") as image:
        image_bytes = image.read()

    with client.websocket_connect("/api/segmentation/predict/webcam") as websocket:
        websocket.send_bytes(image_bytes)
        payload = websocket.receive_json()

    assert isinstance(payload, list)
    if payload:
        assert {"class_id", "class_name", "confidence", "polygon"} <= payload[0].keys()


def test_pothole_segmentation_contract_and_task_parameters_are_registered(client):
    paths = client.app.openapi()["paths"]
    assert "/api/segmentation/pothole/predict" in paths
    assert "/api/segmentation/pothole/predict/video" in paths
    assert "/api/segmentation/pothole/predict/video/progress/{job_id}" in paths
    params = {item["name"] for item in paths["/api/segmentation/pothole/predict"]["post"]["parameters"]}
    assert {"conf", "imgsz", "classes", "max_det", "retina_masks"} <= params
    for method in ("post", "get"):
        video_params = {
            item["name"]
            for item in paths["/api/segmentation/pothole/predict/video"][method]["parameters"]
        }
        assert {"tracking", "tracker"} <= video_params


def test_pothole_segmentation_rejects_invalid_confidence(client):
    response = client.get(
        "/api/segmentation/pothole/predict",
        params={"source": REMOTE_IMAGE_URL, "conf": 2},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", [
    "/api/segmentation/predict/webcam?tracking=true&tracker=bytetrack",
    "/api/pose_estimation/predict/webcam?tracking=true&tracker=bytetrack",
])
def test_tracking_webcam_returns_track_ids(client, endpoint):
    with IMAGE_PATH.open("rb") as image:
        image_bytes = image.read()

    with client.websocket_connect(endpoint) as websocket:
        websocket.send_bytes(image_bytes)
        payload = websocket.receive_json()

    assert isinstance(payload, list)
    if payload:
        assert any(item["track_id"] is not None for item in payload)


def test_brain_tumor_segmentation_routes_are_registered(client):
    paths = client.app.openapi()["paths"]

    assert "/api/segmentation/brain_tumor/predict" in paths
    assert "/api/segmentation/brain_tumor/predict/batch" in paths
    assert "/api/segmentation/brain_tumor/predict/batch/progress/{job_id}" in paths
    assert "/api/segmentation/brain_tumor/predict/video" not in paths


def test_face_mask_detection_routes_are_registered(client):
    paths = client.app.openapi()["paths"]

    assert "/api/object_detection/face_mask/predict" in paths
    assert "/api/object_detection/face_mask/predict/video" in paths


def test_chest_xray_classification_routes_are_registered(client):
    paths = client.app.openapi()["paths"]

    assert "/api/classification/chest_xray/predict" in paths
    assert "/api/classification/chest_xray/predict/video" not in paths


@pytest.mark.parametrize(
    ("endpoint", "field", "filename"),
    [
        ("/api/object_detection/predict/video", "source", "object_detection.mp4"),
        ("/api/classification/predict/video", "source", "classification.mp4"),
        ("/api/segmentation/predict/video", "source", "segmentation.mp4"),
    ],
)
def test_video_endpoints_return_rendered_videos(client, short_video, endpoint, field, filename):
    with short_video.open("rb") as video:
        response = client.post(
            f"{endpoint}?output=video&conf=0.7",
            files={field: (short_video.name, video, "video/mp4")},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content.startswith(b"\x00\x00\x00")
    assert response.headers["content-disposition"].find(filename) >= 0


@pytest.mark.parametrize(
    ("endpoint", "parameter"),
    [
        ("/api/object_detection/predict/video", "source"),
        ("/api/classification/predict/video", "source"),
        ("/api/segmentation/predict/video", "source"),
    ],
)
def test_video_url_prediction_endpoints_return_frame_results(client, endpoint, parameter):
    response = client.get(endpoint, params={parameter: REMOTE_VIDEO_URL})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)
    assert payload


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/object_detection/predict",
        "/api/classification/predict",
        "/api/segmentation/predict",
    ],
)
def test_remote_image_prediction_endpoints_return_predictions(client, endpoint):
    response = client.get(endpoint, params={"source": REMOTE_IMAGE_URL})

    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)



@pytest.mark.parametrize(
    ("endpoint", "field", "content_type"),
    [
        ("/api/object_detection/predict/video", "source", "video/mp4"),
        ("/api/classification/predict/video", "source", "video/mp4"),
        ("/api/segmentation/predict/video", "source", "video/mp4"),
    ],
)
def test_video_prediction_endpoints_return_frame_results(client, short_video, endpoint, field, content_type):
    with short_video.open("rb") as video:
        response = client.post(
            endpoint,
            files={field: (short_video.name, video, content_type)},
        )

    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_media_search_rejects_empty_query_without_external_provider(client):
    response = client.get("/api/media/images/search", params={"query": ""})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "method", "parameters"),
    [
        (
            "/api/classification/predict",
            "get",
            {"output", "conf", "font_size", "font", "probs", "show", "txt_color"},
        ),
        (
            "/api/object_detection/predict",
            "get",
            {"output", "conf", "show_conf", "line_width", "font_size", "font", "labels", "boxes", "color_mode", "txt_color"},
        ),
        (
            "/api/pose_estimation/predict",
            "get",
            {"angles", "output", "conf", "show_conf", "line_width", "font_size", "font", "kpt_radius", "kpt_line", "labels", "boxes", "color_mode", "txt_color"},
        ),
        (
            "/api/segmentation/predict",
            "get",
            {"output", "conf", "show_conf", "line_width", "font_size", "font", "labels", "masks", "boxes", "color_mode"},
        ),
    ],
)
def test_url_prediction_contract_keeps_optional_parameters(client, path, method, parameters):
    operation = client.app.openapi()["paths"][path][method]
    documented_parameters = {parameter["name"] for parameter in operation["parameters"]}

    assert parameters <= documented_parameters


def test_object_detection_color_mode_is_limited_to_supported_literals(client):
    parameter = next(
        parameter
        for parameter in client.app.openapi()["paths"]["/api/object_detection/predict"]["get"]["parameters"]
        if parameter["name"] == "color_mode"
    )

    assert parameter["schema"]["enum"] == ["instance", "class"]

    response = client.get(
        "/api/object_detection/predict",
        params={"source": "https://example.com/image.jpg", "color_mode": "red"},
    )

    assert response.status_code == 422


def test_object_detection_video_tracking_parameters_are_documented(client):
    for method in ("get", "post"):
        operation = client.app.openapi()["paths"]["/api/object_detection/predict/video"][method]
        parameters = {
            parameter["name"]: parameter
            for parameter in operation["parameters"]
        }

        assert parameters["tracking"]["schema"]["default"] is False
        assert parameters["tracker"]["schema"]["default"] == "bytetrack"
        assert parameters["tracker"]["schema"]["enum"] == ["bytetrack", "botsort"]


def test_object_detection_class_filter_is_available_on_prediction_routes(client):
    paths = client.app.openapi()["paths"]
    for path, methods in (
        ("/api/object_detection/predict", ("get", "post")),
        ("/api/object_detection/predict/search", ("get",)),
        ("/api/object_detection/predict/video", ("get", "post")),
    ):
        for method in methods:
            classes = next(
                parameter
                for parameter in paths[path][method]["parameters"]
                if parameter["name"] == "classes"
            )
            assert not classes["required"]
            assert "Ultralytics COCO class list" in classes["description"]
            assert "coco.yaml" in classes["description"]
            array_schema = next(
                schema
                for schema in classes["schema"]["anyOf"]
                if schema.get("type") == "array"
            )
            assert array_schema["items"]["type"] == "string"


def test_object_detection_image_class_filter_limits_predictions(client):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/object_detection/predict?classes=0&classes=2",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 200, response.text
    assert all(item["class_id"] in {0, 2} for item in response.json())


def test_object_detection_class_name_filter_limits_predictions(client):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/object_detection/predict?classes=HoRsE&conf=0",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 200, response.text
    assert response.json()
    assert all(item["class_name"].casefold() == "horse" for item in response.json())


def test_object_detection_unknown_class_name_is_rejected(client):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/object_detection/predict?classes=not-a-model-class",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 422
    assert "Unknown class name" in response.json()["detail"]


def test_segmentation_class_filter_is_available_on_prediction_routes(client):
    paths = client.app.openapi()["paths"]
    for path in (
        "/api/segmentation/predict",
        "/api/segmentation/predict/video",
    ):
        for method in ("get", "post"):
            classes = next(
                parameter
                for parameter in paths[path][method]["parameters"]
                if parameter["name"] == "classes"
            )
            assert not classes["required"]
            assert "Ultralytics COCO class list" in classes["description"]
            assert "coco.yaml" in classes["description"]
            array_schema = next(
                schema
                for schema in classes["schema"]["anyOf"]
                if schema.get("type") == "array"
            )
            assert array_schema["items"]["type"] == "string"


@pytest.mark.parametrize("class_filter", ["HoRsE", "17"])
def test_segmentation_class_filter_limits_predictions(client, class_filter):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            f"/api/segmentation/predict?classes={class_filter}&conf=0.1",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 200, response.text
    assert response.json()
    assert all(item["class_name"].casefold() == "horse" for item in response.json())


def test_segmentation_unknown_class_name_is_rejected(client):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/segmentation/predict?classes=not-a-model-class",
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 422
    assert "Unknown class name" in response.json()["detail"]


def test_segmentation_webcam_class_filter_limits_predictions(client):
    with IMAGE_PATH.open("rb") as image:
        image_bytes = image.read()

    with client.websocket_connect("/api/segmentation/predict/webcam?classes=horse") as websocket:
        websocket.send_bytes(image_bytes)
        payload = websocket.receive_json()

    assert payload
    assert all(item["class_name"].casefold() == "horse" for item in payload)


def test_segmentation_endpoints_do_not_expose_txt_color(client):
    paths = client.app.openapi()["paths"]
    for path, method in (
        ("/api/segmentation/predict", "get"),
        ("/api/segmentation/predict", "post"),
        ("/api/segmentation/brain_tumor/predict", "post"),
    ):
        parameter_names = {
            parameter["name"]
            for parameter in paths[path][method]["parameters"]
        }
        assert "txt_color" not in parameter_names


def test_multi_model_image_returns_each_selected_model_prediction(client):
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/multi_model/predict",
            data={
                "models": ["object-detection", "pose-estimation"],
                "model_options": json.dumps({
                    "object-detection": {"classes": ["horse"]},
                }),
            },
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert set(predictions) == {"object-detection", "pose-estimation"}
    assert predictions["object-detection"]
    assert all(item["class_name"] == "horse" for item in predictions["object-detection"])
    if predictions["pose-estimation"]:
        pose = predictions["pose-estimation"][0]
        assert len(pose["keypoints"]) == 17
        assert "keypoint_names" not in pose


def test_multi_model_description_is_attached_to_its_swagger_tag(client):
    openapi = client.app.openapi()
    tag = next(
        item for item in openapi["tags"]
        if item["name"] == "multi-model prediction"
    )

    assert "application/octet-stream" in tag["description"]
    assert "## Multi-model prediction" not in client.app.description


@pytest.mark.parametrize(
    ("models", "model_options"),
    [
        (["unsupported-model"], None),
        (["classification"], {"classification": {"tracking": True}}),
        (["classification"], {"object-detection": {"classes": ["person"]}}),
        (["pose-estimation"], {"pose-estimation": {"keypoints": ["left_wrist"]}}),
    ],
)
def test_multi_model_image_rejects_invalid_model_configuration(
    client, models, model_options
):
    form_data = {"models": models}
    if model_options is not None:
        form_data["model_options"] = json.dumps(model_options)
    with IMAGE_PATH.open("rb") as image:
        response = client.post(
            "/api/multi_model/predict",
            data=form_data,
            files={"source": (IMAGE_PATH.name, image, "image/png")},
        )

    assert response.status_code == 422


def test_multi_model_webcam_uses_configuration_then_binary_frames(client):
    with IMAGE_PATH.open("rb") as image:
        image_bytes = image.read()

    with client.websocket_connect("/api/multi_model/predict/webcam") as websocket:
        websocket.send_json({
            "models": ["object-detection"],
            "model_options": {"object-detection": {"classes": ["horse"]}},
        })
        websocket.send_bytes(image_bytes)
        response = websocket.receive_json()

    predictions = response["predictions"]
    assert set(predictions) == {"object-detection"}
    assert predictions["object-detection"]
    assert all(item["class_name"] == "horse" for item in predictions["object-detection"])


def test_multi_model_video_results_are_grouped_and_progress_is_aggregate(
    client, short_video
):
    with short_video.open("rb") as video:
        response = client.post(
            "/api/multi_model/predict/video",
            data={"models": ["object-detection", "classification"]},
            files={"source": (short_video.name, video, "video/mp4")},
        )

    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    progress = client.get(f"/api/multi_model/predict/video/progress/{job_id}")
    assert progress.status_code == 200
    events = [
        json.loads(line[6:])
        for line in progress.text.splitlines()
        if line.startswith("data: ")
    ]
    completed = next(event for event in reversed(events) if event["status"] == "completed")
    frames = completed["result"]
    assert [frame["frame_index"] for frame in frames] == list(range(len(frames)))
    assert frames
    assert all(set(frame["predictions"]) == {"object-detection", "classification"} for frame in frames)


def test_multi_model_image_url_source(client, monkeypatch):
    image = cv2.imread(str(IMAGE_PATH))
    success, encoded_image = cv2.imencode(".png", image)
    assert success
    response = type(
        "ImageResponse",
        (),
        {
            "headers": {"content-type": "application/octet-stream"},
            "content": encoded_image.tobytes(),
            "raise_for_status": lambda self: None,
        },
    )()
    monkeypatch.setattr(
        "utils.image.requests.get",
        lambda *args, **kwargs: response,
    )
    response = client.get(
        "/api/multi_model/predict",
        params=[
            ("source", "https://example.test/horses.png"),
            ("models", "object-detection"),
            (
                "model_options",
                json.dumps({"object-detection": {"classes": ["horse"]}}),
            ),
        ],
    )

    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert set(predictions) == {"object-detection"}
    assert predictions["object-detection"]
    assert all(
        item["class_name"] == "horse"
        for item in predictions["object-detection"]
    )


def test_multi_model_video_url_source(client, short_video, monkeypatch):
    import multi_model.controllers.multi_model_rest_controller as controller

    def copy_video_to_temp(_source):
        descriptor, path = tempfile.mkstemp(suffix=".mp4")
        os.close(descriptor)
        shutil.copyfile(short_video, path)
        return path

    monkeypatch.setattr(controller, "download_video_to_temp", copy_video_to_temp)
    response = client.get(
        "/api/multi_model/predict/video",
        params=[
            ("source", "https://example.test/video.mp4"),
            ("models", "classification"),
        ],
    )

    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    progress = client.get(f"/api/multi_model/predict/video/progress/{job_id}")
    events = [
        json.loads(line[6:])
        for line in progress.text.splitlines()
        if line.startswith("data: ")
    ]
    completed = next(event for event in reversed(events) if event["status"] == "completed")
    assert completed["result"]
    assert all(
        set(frame["predictions"]) == {"classification"}
        for frame in completed["result"]
    )


@pytest.mark.parametrize("path", [
    "/api/pose_estimation/predict/video",
    "/api/segmentation/predict/video",
])
def test_video_tracking_parameters_are_documented(client, path):
    for method in ("get", "post"):
        parameters = {
            parameter["name"]: parameter
            for parameter in client.app.openapi()["paths"][path][method]["parameters"]
        }
        assert parameters["tracking"]["schema"]["default"] is False
        assert parameters["tracker"]["schema"]["default"] == "bytetrack"
        assert parameters["tracker"]["schema"]["enum"] == ["bytetrack", "botsort"]


def test_object_detection_video_rejects_unsupported_tracker(client):
    response = client.get(
        "/api/object_detection/predict/video",
        params={"source": REMOTE_VIDEO_URL, "tracker": "unsupported"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "method", "parameters"),
    [
        ("/api/classification/predict/video", "post", {"output", "conf", "font_size", "font", "probs", "show", "txt_color"}),
        ("/api/object_detection/predict/video", "post", {"output", "conf", "show_conf", "line_width", "font_size", "font", "labels", "boxes", "color_mode", "txt_color"}),
        ("/api/pose_estimation/predict/video", "post", {"output", "conf"}),
        ("/api/segmentation/predict/video", "post", {"output", "conf"}),
    ],
)
def test_video_prediction_contract_keeps_optional_parameters(client, path, method, parameters):
    operation = client.app.openapi()["paths"][path][method]
    documented_parameters = {parameter["name"] for parameter in operation["parameters"]}

    assert parameters <= documented_parameters


def test_rendering_options_match_installed_ultralytics_plot_api():
    from ultralytics.engine.results import Results

    supported = set(inspect.signature(Results.plot).parameters)
    configured = {
        "conf", "line_width", "font_size", "font", "kpt_radius", "kpt_line",
        "labels", "boxes", "masks", "probs", "show", "color_mode", "txt_color",
    }

    assert configured <= supported


def test_pose_image_angle_form_field_returns_requested_measurements(client):
    keypoints = [[0.0, 0.0, 0.9] for _ in range(17)]
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]
    pose = Pose(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=keypoints,
    )

    class FakePoseService:
        async def predict_image(self, source, output, plot_options, conf, angles=None):
            return PoseAngleService.add_angles([pose], angles)

    app.dependency_overrides[get_service] = lambda: FakePoseService()
    try:
        response = client.post(
            "/api/pose_estimation/predict",
            files={"source": ("test.png", b"image-bytes", "image/png")},
            params={"angles": '[["left_shoulder","left_elbow","left_wrist"]]'},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200, response.text
    result = response.json()[0]["angles"][0]
    assert result["joint_indices"] == [5, 7, 9]
    assert result["keypoint_confidences"] == [0.9, 0.9, 0.9]
    assert result["degrees"] == 270.0
    assert result["valid"] is True


def test_pose_webcam_angles_automatically_enable_tracking(client, monkeypatch):
    keypoints = [[0.0, 0.0, 0.9] for _ in range(17)]
    keypoints[5] = [0.0, 0.0, 0.9]
    keypoints[7] = [1.0, 0.0, 0.9]
    keypoints[9] = [1.0, 1.0, 0.9]
    pose = Pose(
        class_id=0,
        class_name="person",
        confidence=0.9,
        keypoints=keypoints,
        track_id=42,
    )

    class FakeTrackingService:
        def reset_tracking(self):
            pass

    class FakePoseService:
        tracking_service = FakeTrackingService()

        def predict_frame_bytes(
            self, frame_bytes, conf, tracking, tracker, angles=None
        ):
            assert tracking is True
            return PoseAngleService.add_angles([pose], angles)

    import pose_estimation.controllers.pose_estimation_rest_controller as pose_controller

    monkeypatch.setattr(pose_controller, "get_service", lambda: FakePoseService())
    with client.websocket_connect(
        "/api/pose_estimation/predict/webcam"
        "?angles=%5B%5B%22left_shoulder%22%2C%22left_elbow%22%2C%22left_wrist%22%5D%5D"
    ) as websocket:
        websocket.send_bytes(b"frame")
        result = websocket.receive_json()[0]

    assert result["track_id"] == 42
    assert result["angles"][0]["degrees"] == 270.0


def test_pose_video_angles_automatically_enable_tracking(client, monkeypatch):
    captured = {}

    class FakePoseService:
        async def predict_video_path_with_angles(
            self, path, conf, progress, tracking, tracker, angles
        ):
            captured["tracking"] = tracking
            captured["tracker"] = tracker
            captured["angles"] = angles
            return []

    import pose_estimation.controllers.pose_estimation_rest_controller as pose_controller

    async def save_upload(_source):
        return "temporary-video.mp4"

    def submit_path(_path, processor):
        captured["processor"] = processor
        return "job-id"

    app.dependency_overrides[get_service] = lambda: FakePoseService()
    monkeypatch.setattr(pose_controller, "save_uploadfile_to_temp", save_upload)
    monkeypatch.setattr(pose_controller.video_job_manager, "submit_path", submit_path)
    try:
        response = client.post(
            "/api/pose_estimation/predict/video",
            files={"source": ("video.mp4", b"video-bytes", "video/mp4")},
            params={"angles": "[[5,7,9]]"},
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": "job-id"}
    asyncio.run(captured["processor"]("temporary-video.mp4", lambda *_args: None))
    assert captured["tracking"] is True
    assert captured["tracker"] == "bytetrack"
    assert captured["angles"] == [(5, 7, 9)]


def test_pose_video_url_angles_automatically_enable_tracking(client, monkeypatch):
    captured = {}

    class FakePoseService:
        async def predict_video_url_with_angles(
            self, source, conf, progress, tracking, tracker, angles
        ):
            captured["source"] = source
            captured["tracking"] = tracking
            captured["tracker"] = tracker
            captured["angles"] = angles
            return []

    import pose_estimation.controllers.pose_estimation_rest_controller as pose_controller

    def submit(worker):
        captured["worker"] = worker
        return "job-id"

    app.dependency_overrides[get_service] = lambda: FakePoseService()
    monkeypatch.setattr(pose_controller.video_job_manager, "submit", submit)
    try:
        response = client.get(
            "/api/pose_estimation/predict/video",
            params={
                "source": "https://example.com/video.mp4",
                "angles": '[["left_shoulder","left_elbow","left_wrist"]]',
            },
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": "job-id"}
    asyncio.run(captured["worker"](lambda *_args: None))
    assert captured["source"] == "https://example.com/video.mp4"
    assert captured["tracking"] is True
    assert captured["tracker"] == "bytetrack"
    assert captured["angles"] == [(5, 7, 9)]


def test_pose_angle_request_rejects_media_output_and_invalid_keypoints(client):
    invalid_angles = client.get(
        "/api/pose_estimation/predict",
        params={
            "source": "https://example.com/image.jpg",
            "angles": "[[5,7,17]]",
        },
    )
    media_output = client.get(
        "/api/pose_estimation/predict",
        params={
            "source": "https://example.com/image.jpg",
            "angles": "[[5,7,9]]",
            "output": "image",
        },
    )

    assert invalid_angles.status_code == 422
    assert media_output.status_code == 422


def test_pose_angle_parameter_is_documented_for_multipart_and_url_requests(client):
    schema = client.app.openapi()
    image_post = schema["paths"]["/api/pose_estimation/predict"]["post"]
    video_post = schema["paths"]["/api/pose_estimation/predict/video"]["post"]
    video_get = schema["paths"]["/api/pose_estimation/predict/video"]["get"]

    image_body_schema = image_post["requestBody"]["content"]["multipart/form-data"]["schema"]
    video_body_schema = video_post["requestBody"]["content"]["multipart/form-data"]["schema"]
    image_body = schema["components"]["schemas"][image_body_schema["$ref"].split("/")[-1]]
    video_body = schema["components"]["schemas"][video_body_schema["$ref"].split("/")[-1]]
    assert "source" in image_body["properties"]
    assert "source" in video_body["properties"]
    assert "angles" not in image_body["properties"]
    assert "angles" not in video_body["properties"]
    assert "angles" in {parameter["name"] for parameter in image_post["parameters"]}
    assert "angles" in {parameter["name"] for parameter in video_post["parameters"]}
    assert "angles" in {parameter["name"] for parameter in video_get["parameters"]}


def test_object_detection_show_conf_is_forwarded_to_ultralytics_plot():
    from object_detection.repositories.yolo_model_repository import YoloModelRepository

    captured = {}

    class FakeResult:
        names = {}
        boxes = None
        obb = None

        def plot(self, **options):
            captured.update(options)
            return __import__("numpy").zeros((4, 4, 3), dtype="uint8")

    repository = YoloModelRepository.__new__(YoloModelRepository)
    repository._predict_result = lambda source, conf: FakeResult()

    repository.predict_annotated_image(
        source=object(),
        plot_options={"conf": False, "line_width": 10, "font_size": 155},
    )

    assert captured["conf"] is False
    assert captured["line_width"] == 10
    assert captured["font_size"] == 155


def test_object_detection_text_color_is_forwarded_to_box_labels(monkeypatch):
    import object_detection.repositories.yolo_model_repository as repository_module

    captured = {}

    class FakeAnnotator:
        def __init__(self, image, **options):
            self.dark_colors = {(0, 0, 0)}
            self.light_colors = {(255, 255, 255)}

        def box_label(self, box, label, color, txt_color):
            captured["txt_color"] = (255, 255, 255) if self.dark_colors else txt_color

        def result(self):
            return __import__("numpy").zeros((4, 4, 3), dtype="uint8")

    class FakeBox:
        cls = type("Value", (), {"item": lambda self: 0})()
        conf = type("Value", (), {"item": lambda self: 0.9})()
        xyxy = type("Value", (), {"squeeze": lambda self: type("Value", (), {"tolist": lambda self: [0, 0, 2, 2]})()})()

    class FakeResult:
        names = {0: "sign"}
        obb = None
        boxes = [FakeBox()]

        def plot(self, **options):
            return __import__("numpy").zeros((4, 4, 3), dtype="uint8")

    monkeypatch.setattr(repository_module, "Annotator", FakeAnnotator)
    repository_module.YoloModelRepository._plot_result(
        FakeResult(),
        {"conf": True, "labels": True, "boxes": True, "txt_color": (0, 125, 255)},
    )

    assert captured["txt_color"] == (0, 125, 255)
