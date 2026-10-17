from pathlib import Path
import inspect
import tempfile

import cv2
import pytest
from fastapi.testclient import TestClient

from main import app


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
            {"output", "conf", "show_conf", "line_width", "font_size", "font", "kpt_radius", "kpt_line", "labels", "boxes", "color_mode", "txt_color"},
        ),
        (
            "/api/segmentation/predict",
            "get",
            {"output", "conf", "show_conf", "line_width", "font_size", "font", "labels", "masks", "boxes", "color_mode", "txt_color"},
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
