import base64
from io import BytesIO

import pytest
from PIL import Image

from face_swap.repositories.qwen_image_21_gguf_q3_k_xl_local_face_swap_repository import (
    BFS_FILE,
    TURBO_FILE,
    QwenImage21GgufQ3KxlLocalFaceSwapRepository,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data
        self.ok = True

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeSdcppSession:
    def __init__(self, job):
        self.job = job
        self.submission = None

    def post(self, url, json, timeout):
        self.submission = {"url": url, "json": json, "timeout": timeout}
        return FakeResponse({"id": "test-job"})

    def get(self, url, timeout):
        assert url.endswith("/sdcpp/v1/jobs/test-job")
        return FakeResponse(self.job)


def test_gguf_repository_submits_both_references_and_returns_image():
    output = BytesIO()
    Image.new("RGB", (64, 64), color="green").save(output, format="PNG")
    session = FakeSdcppSession(
        {
            "status": "completed",
            "result": {
                "images": [
                    {"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}
                ]
            },
        }
    )
    repository = QwenImage21GgufQ3KxlLocalFaceSwapRepository(
        server_url="http://127.0.0.1:1235"
    )
    repository._session = session

    result = repository.swap(
        Image.new("RGB", (64, 64), color="red"),
        Image.new("RGB", (64, 64), color="blue"),
    )

    payload = session.submission["json"]
    assert result.getpixel((0, 0)) == (0, 128, 0)
    assert len(payload["ref_images"]) == 2
    assert payload["width"] * payload["height"] <= 512 * 512
    assert payload["ref_image_args"] == "preset=qwen,vae_input_max_pixels=262144"
    assert payload["sample_params"]["sample_steps"] == 6
    assert [entry["path"] for entry in payload["lora"]] == [BFS_FILE, TURBO_FILE]


def test_gguf_repository_surfaces_generation_failure():
    session = FakeSdcppSession(
        {
            "status": "failed",
            "error": {"message": "GPU allocation failed"},
        }
    )
    repository = QwenImage21GgufQ3KxlLocalFaceSwapRepository(
        server_url="http://127.0.0.1:1235"
    )
    repository._session = session

    with pytest.raises(RuntimeError, match="GPU allocation failed"):
        repository.swap(
            Image.new("RGB", (64, 64)),
            Image.new("RGB", (64, 64)),
        )
