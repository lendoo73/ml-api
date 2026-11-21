import os
from io import BytesIO

import requests
from PIL import Image


class FaceSwapImageWorkerRepository:
    def __init__(self, worker_url: str | None = None, session=None):
        self._worker_url = (
            worker_url or os.getenv(
                "FACE_SWAP_VIDEO_WORKER_URL",
                "http://face-swap-video-worker:8001",
            )
        ).rstrip("/")
        self._session = session or requests.Session()

    def swap(self, target: Image.Image, reference: Image.Image) -> Image.Image:
        timeout = _positive_int("FACE_SWAP_IMAGE_WORKER_TIMEOUT", 1800)
        response = None
        try:
            response = self._session.post(
                f"{self._worker_url}/images",
                files={
                    "source": ("portrait.png", _encode_png(reference), "image/png"),
                    "target_image": ("target.png", _encode_png(target), "image/png"),
                },
                timeout=(15, timeout),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            detail = _response_detail(response) if response else None
            if detail:
                raise RuntimeError(f"FaceFusion image worker failed: {detail}") from error
            status = response.status_code if response else "unavailable"
            raise RuntimeError(
                "FaceFusion image worker request failed with "
                f"HTTP {status}."
            ) from error

        try:
            with Image.open(BytesIO(response.content)) as image:
                return image.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "FaceFusion image worker returned an invalid image."
            ) from error


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _response_detail(response) -> str | None:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return str(detail) if detail else None
