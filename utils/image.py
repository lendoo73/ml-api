import cv2
import numpy as np
import requests

from exceptions import InvalidImageSourceError


def download_image_to_ndarray(
    url: str, timeout: int = 10, allow_octet_stream: bool = False
) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as error:
        raise InvalidImageSourceError(
            "Image URL could not be downloaded. Provide a direct, publicly accessible image URL."
        ) from error

    content_type = resp.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    is_image = content_type.startswith("image/")
    is_allowed_octet_stream = allow_octet_stream and content_type == "application/octet-stream"
    if not is_image and not is_allowed_octet_stream:
        raise InvalidImageSourceError(
            f"URL does not point to an image (content-type: {content_type or 'unknown'})"
        )

    image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise InvalidImageSourceError("Downloaded content could not be decoded as an image")

    return img