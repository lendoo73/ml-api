import json
from io import BytesIO

from PIL import Image
import pytest
import requests

from face_swap.repositories.qwen_image_21_huggingface_space_face_swap_repository import (
    QwenImage21HuggingFaceSpaceFaceSwapRepository,
)


class FakeResponse:
    def __init__(self, data=None, content=b"", lines=None, status_code=200, text=""):
        self._data = data
        self.content = content
        self._lines = lines or []
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._data

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


class FakeSpaceSession:
    def __init__(self):
        output = BytesIO()
        Image.new("RGB", (16, 16), color="green").save(output, format="PNG")
        self.posts = []
        self.gets = []
        self._upload_count = 0
        self._image_bytes = output.getvalue()
        self.generation_response = FakeResponse({"event_id": "test-event"})

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/gradio_api/upload"):
            self._upload_count += 1
            return FakeResponse([f"/tmp/upload-{self._upload_count}.png"])
        return self.generation_response

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if kwargs.get("stream"):
            lines = [
                "event: complete",
                f'data: {json.dumps([{"path": "/tmp/result.png"}])}',
            ]
            return FakeResponse(lines=lines)
        return FakeResponse(content=self._image_bytes)


def test_huggingface_space_repository_uploads_images_and_returns_image():
    session = FakeSpaceSession()
    repository = QwenImage21HuggingFaceSpaceFaceSwapRepository(session=session)

    result = repository.swap(
        Image.new("RGB", (32, 24), color="red"),
        Image.new("RGB", (24, 32), color="blue"),
    )

    assert result.getpixel((0, 0)) == (0, 128, 0)
    assert len(session.posts) == 3
    generation_url, generation_request = session.posts[2]
    assert generation_url.endswith("/gradio_api/call/generate")
    generation_data = generation_request["json"]["data"]
    images = generation_data[1]
    assert [entry["image"]["path"] for entry in images] == [
        "/tmp/upload-1.png",
        "/tmp/upload-2.png",
    ]
    assert generation_data == [
        "Replace the head in <image1> with the head from <image2>. Preserve the scene, "
        "lighting, pose, and expression from <image1>, and retain the identity and defining "
        "facial features of <image2>. Make the result photorealistic and seamless.",
        images,
        "",
        1.0,
        8,
        0,
        1024,
        "Auto",
    ]
    assert session.gets[0][0].endswith("/gradio_api/call/generate/test-event")
    assert session.gets[1][0].endswith("/gradio_api/file=/tmp/result.png")


def test_huggingface_space_generation_error_includes_provider_response():
    session = FakeSpaceSession()
    session.generation_response = FakeResponse(
        status_code=422,
        text='{"detail":"Invalid gallery image input"}',
    )
    repository = QwenImage21HuggingFaceSpaceFaceSwapRepository(session=session)

    with pytest.raises(
        RuntimeError,
        match="HTTP 422: \\{.detail.:.Invalid gallery image input.\\}",
    ):
        repository.swap(Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16)))
