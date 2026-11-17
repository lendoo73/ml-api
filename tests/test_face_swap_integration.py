from io import BytesIO
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from controllers.controller import api_router
from face_swap.controllers import face_swap_rest_controller
from face_swap.controllers.face_swap_rest_controller import get_service
from face_swap.models.face_swap_model import (
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
)
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

    def swap_video(self, reference_bytes, video_path):
        output_path = str(Path(video_path).with_name("fake-face-swap.mp4"))
        Path(output_path).write_bytes(b"swapped-video")
        return output_path

    def decode_reference_image(self, image_bytes):
        if image_bytes == b"invalid":
            raise ValueError("Invalid portrait image.")
        return image_bytes

    def swap_frame(self, reference, frame_bytes):
        return b"swapped-frame"


def test_face_swap_routes_are_registered():
    paths = {route.path for route in api_router.routes}
    assert "/face_swap/predict/image" in paths
    assert "/face_swap/predict/video" in paths
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


def test_face_swap_model_parameter_is_limited_to_available_models():
    paths = app.openapi()["paths"]
    for path in ("/api/face_swap/predict/image", "/api/face_swap/predict/video"):
        parameters = paths[path]["post"]["parameters"]
        model_parameter = next(
            parameter for parameter in parameters if parameter["name"] == "model"
        )
        assert model_parameter["schema"]["enum"] == [
            "qwen-image-2.1-bf16-local",
            "qwen-image-2.1-gguf-q3-k-xl-local",
        ]

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


def test_face_swap_video_endpoint_returns_mp4():
    app.dependency_overrides[get_service] = FakeFaceSwapService
    try:
        response = TestClient(app).post(
            "/api/face_swap/predict/video",
            files={
                "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                "target_video": ("target.mp4", b"video", "video/mp4"),
            },
        )
    finally:
        app.dependency_overrides.pop(get_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"swapped-video"


def test_face_swap_webcam_receives_portrait_then_streams_frames(monkeypatch):
    selected_models = []

    def fake_get_service(model):
        selected_models.append(model)
        return FakeFaceSwapService()

    monkeypatch.setattr(face_swap_rest_controller, "get_service", fake_get_service)
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


def test_face_swap_video_service_processes_frames_and_writes_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for video processing.")

    input_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 64),
    )
    assert writer.isOpened()
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.write(np.full((64, 64, 3), 127, dtype=np.uint8))
    writer.release()

    portrait = BytesIO()
    Image.new("RGB", (32, 32), color="blue").save(portrait, format="JPEG")
    output_path = FaceSwapService(FakeModelRepository()).swap_video(
        portrait.getvalue(),
        str(input_path),
    )
    try:
        capture = cv2.VideoCapture(output_path)
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        capture.release()
    finally:
        Path(output_path).unlink(missing_ok=True)
