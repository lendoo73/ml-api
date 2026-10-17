import cv2
import numpy as np
import requests

from exceptions import InvalidImageSourceError


def download_image_to_ndarray(url: str, timeout: int = 10) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as error:
        raise InvalidImageSourceError(
            "Image URL could not be downloaded. Provide a direct, publicly accessible image URL."
        ) from error

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise InvalidImageSourceError(
            f"URL does not point to an image (content-type: {content_type or 'unknown'})"
        )

    image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise InvalidImageSourceError("Downloaded content could not be decoded as an image")

    return img