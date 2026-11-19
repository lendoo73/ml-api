import json
import os
import time
from io import BytesIO
from urllib.parse import quote, urljoin, urlparse

import requests
from PIL import Image


SPACE_URL = "https://hugging-apps-qwen-image-2-1.hf.space"
SPACE_TIMEOUT_ENV = "FACE_SWAP_HF_SPACE_TIMEOUT"
FACE_SWAP_PROMPT = (
    "Replace the head in <image1> with the head from <image2>. Preserve the scene, "
    "lighting, pose, and expression from <image1>, and retain the identity and defining "
    "facial features of <image2>. Make the result photorealistic and seamless."
)


class QwenImage21HuggingFaceSpaceFaceSwapRepository:
    def __init__(self, session=None, space_url: str = SPACE_URL):
        self._session = session or requests.Session()
        self._space_url = space_url.rstrip("/")

    def swap(self, target: Image.Image, reference: Image.Image) -> Image.Image:
        timeout = int(os.getenv(SPACE_TIMEOUT_ENV, "1800"))
        if timeout <= 0:
            raise ValueError(f"{SPACE_TIMEOUT_ENV} must be positive.")

        uploaded_images = [
            self._upload_image(target, "target.png"),
            self._upload_image(reference, "portrait.png"),
        ]
        response = self._session.post(
            f"{self._space_url}/gradio_api/call/generate",
            json={
                "data": [
                    FACE_SWAP_PROMPT,
                    [
                        {
                            "image": {
                                "path": path,
                                "meta": {"_type": "gradio.FileData"},
                            }
                        }
                        for path in uploaded_images
                    ],
                    "",
                    1.0,
                    8,
                    0,
                    1024,
                    "Auto",
                ]
            },
            timeout=30,
        )
        _raise_for_status(response, "generation submission")
        event_id = response.json().get("event_id")
        if not event_id:
            raise RuntimeError("Hugging Face Space returned no generation event ID.")

        result_path = self._wait_for_result(event_id, timeout)
        result_url = self._file_url(result_path)
        result_response = self._session.get(result_url, timeout=60)
        _raise_for_status(result_response, "image download")
        try:
            with Image.open(BytesIO(result_response.content)) as image:
                return image.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Hugging Face Space returned an invalid face-swap image."
            ) from error

    def _upload_image(self, image: Image.Image, filename: str) -> str:
        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")
        response = self._session.post(
            f"{self._space_url}/gradio_api/upload",
            files=[("files", (filename, output.getvalue(), "image/png"))],
            timeout=60,
        )
        _raise_for_status(response, "image upload")
        paths = response.json()
        if not isinstance(paths, list) or not paths or not isinstance(paths[0], str):
            raise RuntimeError("Hugging Face Space returned no uploaded image path.")
        return paths[0]

    def _wait_for_result(self, event_id: str, timeout: int) -> str:
        response = self._session.get(
            f"{self._space_url}/gradio_api/call/generate/{quote(event_id, safe='')}",
            stream=True,
            timeout=(15, timeout),
        )
        _raise_for_status(response, "generation polling")
        event_type = None
        deadline = time.monotonic() + timeout
        for line in response.iter_lines(decode_unicode=True):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Hugging Face Space face swap exceeded {timeout} seconds."
                )
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line:
                continue
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
                if event_type == "error":
                    raise RuntimeError(f"Hugging Face Space face swap failed: {data}")
                if event_type == "complete":
                    return _extract_result_path(json.loads(data))
        raise RuntimeError("Hugging Face Space ended without a face-swap result.")

    def _file_url(self, result_path: str) -> str:
        parsed = urlparse(result_path)
        if parsed.scheme or parsed.netloc:
            if parsed.netloc != urlparse(self._space_url).netloc:
                raise RuntimeError("Hugging Face Space returned an unexpected file URL.")
            return result_path
        return urljoin(
            self._space_url + "/",
            f"gradio_api/file={quote(result_path, safe='/')}",
        )


def _extract_result_path(data) -> str:
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        result_path = data.get("path") or data.get("url")
        if isinstance(result_path, str) and result_path:
            return result_path
    raise RuntimeError("Hugging Face Space completed without returning an image.")


def _raise_for_status(response, operation: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        response_text = getattr(response, "text", "").strip()
        detail = response_text[:1000] if response_text else str(error)
        raise RuntimeError(
            f"Hugging Face Space {operation} failed with HTTP "
            f"{response.status_code}: {detail}"
        ) from error
