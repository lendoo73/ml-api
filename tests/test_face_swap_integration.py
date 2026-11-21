import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from controllers.controller import api_router
from face_swap.controllers import face_swap_rest_controller
from face_swap.controllers.face_swap_rest_controller import (
    get_service,
    get_video_service,
)
from face_swap.models.face_swap_model import (
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
)
from face_swap.models.face_swap_image_model import FaceSwapImageModel
from face_swap.repositories.face_swap_image_worker_repository import (
    FaceSwapImageWorkerRepository,
)
from face_swap.services.face_swap_video_service import FaceSwapVideoService
import face_swap.services.face_swap_service as face_swap_service_module
from face_swap.services.face_swap_service import FaceSwapService
from main import app


class FakeModelRepository:
    def swap(self, target, reference):
        return target


class FakeFaceSwapService(FaceSwapService):
    def __init__(self):
        super().__init__(FakeModelRepository())

    def swap_image(self, reference_bytes, target_bytes):
        self.decode_reference_image(reference_bytes)
        return b"swapped-image"

    def decode_reference_image(self, image_bytes):
        if image_bytes == b"invalid":
            raise ValueError("Invalid portrait image.")
        return image_bytes

    def swap_frame(self, reference, frame_bytes):
        return b"swapped-frame"


class FakeFaceSwapVideoService:
    def swap_video(self, reference_bytes, video_path, progress_callback=None):
        if progress_callback:
            progress_callback(1, 1)
        output_path = str(Path(video_path).with_name("fake-face-swap.mp4"))
        Path(output_path).write_bytes(b"swapped-video")
        return output_path


def test_face_swap_video_service_invokes_facefusion_locally(tmp_path):
    class FakeFaceFusionVideoService:
        def swap_video(self, source_path, video_path, progress_callback):
            assert Path(source_path).read_bytes() == b"portrait"
            assert Path(video_path).read_bytes() == b"video"
            progress_callback(1, 2)
            return "swapped-video.mp4"

    progress = []
    service = FaceSwapVideoService(FakeFaceFusionVideoService())
    video_path = tmp_path / "target.mp4"
    video_path.write_bytes(b"video")

    result = service.swap_video(
        b"portrait",
        str(video_path),
        lambda current, total: progress.append((current, total)),
    )

    assert result == "swapped-video.mp4"
    assert progress == [(1, 2)]


def test_face_swap_routes_are_registered():
    paths = {route.path for route in api_router.routes}
    assert "/face_swap/predict/image" in paths
    assert "/face_swap/predict/video" in paths
    assert "/face_swap/predict/video/progress/{job_id}" in paths
    assert "/face_swap/predict/video/result/{job_id}" in paths
    assert "/face_swap/predict/webcam" in paths


def test_face_swap_image_endpoint_returns_jpeg(monkeypatch, capsys):
    app.dependency_overrides[get_service] = FakeFaceSwapService
    try:
        response = TestClient(app).post(
            "/api/face_swap/predict/image?model=qwen-image-2.1-gguf-q3-k-xl-local",
            files={
                "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                "target": ("target.jpg", b"target", "image/jpeg"),
            },
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"swapped-image"
    assert "Face swap image execution time:" in capsys.readouterr().out


def test_face_swap_image_rejects_invalid_portrait(monkeypatch):
    app.dependency_overrides[get_service] = FakeFaceSwapService
    try:
        response = TestClient(app).post(
            "/api/face_swap/predict/image",
            files={
                "source": ("portrait.jpg", b"invalid", "image/jpeg"),
                "target": ("target.jpg", b"target", "image/jpeg"),
            },
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid portrait image."


def test_face_swap_image_endpoint_accepts_facefusion_model():
    app.dependency_overrides[get_service] = FakeFaceSwapService
    try:
        response = TestClient(app).post(
            "/api/face_swap/predict/image?model=facefusion-3.9.2-local",
            files={
                "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                "target": ("target.jpg", b"target", "image/jpeg"),
            },
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200
    assert response.content == b"swapped-image"


def test_face_swap_model_parameter_is_limited_to_available_models():
    paths = app.openapi()["paths"]
    parameters = paths["/api/face_swap/predict/image"]["post"]["parameters"]
    model_parameter = next(
        parameter for parameter in parameters if parameter["name"] == "model"
    )
    assert app.openapi()["components"]["schemas"]["FaceSwapImageModel"]["enum"] == [
        "qwen-image-2.1-bf16-local",
        "qwen-image-2.1-gguf-q3-k-xl-local",
        "qwen-image-2.1-huggingface-space",
        "facefusion-3.9.2-local",
    ]
    assert model_parameter["schema"]["default"] == "qwen-image-2.1-bf16-local"
    video_model_parameter = next(
        parameter
        for parameter in paths["/api/face_swap/predict/video"]["post"]["parameters"]
        if parameter["name"] == "model"
    )
    assert app.openapi()["components"]["schemas"]["FaceSwapVideoModel"]["enum"] == [
        "facefusion-3.9.2-local"
    ]
    assert video_model_parameter["schema"]["default"] == "facefusion-3.9.2-local"

    response = TestClient(app).post(
        "/api/face_swap/predict/image?model=unknown",
        files={
            "source": ("portrait.jpg", b"portrait", "image/jpeg"),
            "target": ("target.jpg", b"target", "image/jpeg"),
        },
    )
    assert response.status_code == 422


def test_face_swap_service_selects_gguf_repository(monkeypatch):
    repository = FakeModelRepository()
    monkeypatch.setitem(
        face_swap_service_module._MODEL_REPOSITORY_FACTORIES,
        QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
        lambda: repository,
    )

    service = FaceSwapService(
        model=QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL
    )

    assert service.model_repository is repository


def test_face_swap_service_selects_facefusion_image_repository():
    service = FaceSwapService(
        model=FaceSwapImageModel.FACEFUSION_3_9_2_LOCAL
    )

    assert isinstance(service.model_repository, FaceSwapImageWorkerRepository)


def test_face_swap_weight_directories_use_the_shared_model_weights_root():
    from face_swap.repositories.qwen_image_21_bf16_local_face_swap_repository import (
        WEIGHTS_DIR as bf16_weights_dir,
    )
    from face_swap.repositories.qwen_image_21_gguf_q3_k_xl_local_face_swap_repository import (
        WEIGHTS_DIR as gguf_weights_dir,
    )

    assert bf16_weights_dir.parent.name == "face-swap"
    assert gguf_weights_dir.parent.name == "face-swap"
    assert bf16_weights_dir.parent.parent == gguf_weights_dir.parent.parent


def test_face_swap_video_endpoint_returns_job_and_result():
    app.dependency_overrides[get_video_service] = FakeFaceSwapVideoService
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/face_swap/predict/video?model=facefusion-3.9.2-local",
                files={
                    "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                    "target_video": ("target.mp4", b"video", "video/mp4"),
                },
            )
            assert response.status_code == 200
            job_id = response.json()["job_id"]

            progress = client.get(f"/api/face_swap/predict/video/progress/{job_id}")
            events = [
                json.loads(line[6:])
                for line in progress.text.splitlines()
                if line.startswith("data: ")
            ]
            completed = next(event for event in reversed(events) if event["status"] == "completed")
            assert completed["progress"] == 100
            assert completed["result"] == {
                "download_url": f"/api/face_swap/predict/video/result/{job_id}"
            }

            result = client.get(completed["result"]["download_url"])
    finally:
        app.dependency_overrides.pop(get_video_service, None)

    assert progress.status_code == 200
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"
    assert result.content == b"swapped-video"


def test_face_swap_video_endpoint_rejects_unknown_model():
    response = TestClient(app).post(
        "/api/face_swap/predict/video?model=unknown",
        files={
            "source": ("portrait.jpg", b"portrait", "image/jpeg"),
            "target_video": ("target.mp4", b"video", "video/mp4"),
        },
    )

    assert response.status_code == 422


def test_face_swap_video_job_reports_failure():
    class FailingFaceSwapService(FakeFaceSwapVideoService):
        def swap_video(self, reference_bytes, video_path, progress_callback=None):
            raise ValueError("Could not open target video.")

    app.dependency_overrides[get_video_service] = FailingFaceSwapService
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/face_swap/predict/video",
                files={
                    "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                    "target_video": ("target.mp4", b"video", "video/mp4"),
                },
            )
            job_id = response.json()["job_id"]
            progress = client.get(f"/api/face_swap/predict/video/progress/{job_id}")
            events = [
                json.loads(line[6:])
                for line in progress.text.splitlines()
                if line.startswith("data: ")
            ]
            failed = next(event for event in reversed(events) if event["status"] == "failed")
            result = client.get(f"/api/face_swap/predict/video/result/{job_id}")
    finally:
        app.dependency_overrides.pop(get_video_service, None)

    assert response.status_code == 200
    assert failed["error"] == "Could not open target video."
    assert result.status_code == 409


def test_face_swap_webcam_receives_portrait_then_streams_frames(monkeypatch):
    selected_models = []

    def fake_get_service(model):
        selected_models.append(model)
        return FakeFaceSwapService()

    monkeypatch.setattr(
        face_swap_rest_controller,
        "get_webcam_service",
        fake_get_service,
    )
    with TestClient(app).websocket_connect(
        "/api/face_swap/predict/webcam?model=qwen-image-2.1-bf16-local"
    ) as websocket:
        websocket.send_bytes(b"portrait")
        websocket.send_bytes(b"target-frame")
        assert websocket.receive_bytes() == b"swapped-frame"
    assert selected_models == ["qwen-image-2.1-bf16-local"]


def test_face_swap_webcam_rejects_unknown_model():
    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(app).websocket_connect(
            "/api/face_swap/predict/webcam?model=unknown"
        ):
            pass

    assert error.value.code == 1008


def test_face_swap_webcam_does_not_accept_facefusion_image_model():
    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(app).websocket_connect(
            "/api/face_swap/predict/webcam?model=facefusion-3.9.2-local"
        ):
            pass

    assert error.value.code == 1008


def test_facefusion_image_repository_calls_worker():
    from io import BytesIO

    from PIL import Image

    from face_swap.repositories.face_swap_image_worker_repository import (
        FaceSwapImageWorkerRepository,
    )

    output = BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="JPEG")

    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.content = output.getvalue()

        def raise_for_status(self):
            return None

    class FakeSession:
        def post(self, url, files, timeout):
            assert url == "http://worker:8001/images"
            assert set(files) == {"source", "target_image"}
            assert files["source"][1].startswith(b"\x89PNG")
            assert files["target_image"][1].startswith(b"\x89PNG")
            assert timeout[0] == 15
            return FakeResponse()

    repository = FaceSwapImageWorkerRepository(
        worker_url="http://worker:8001",
        session=FakeSession(),
    )
    result = repository.swap(
        Image.new("RGB", (2, 2), "blue"),
        Image.new("RGB", (2, 2), "green"),
    )

    assert result.mode == "RGB"
    assert result.size == (2, 2)
