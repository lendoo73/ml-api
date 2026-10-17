import cv2
import numpy as np
from fastapi import UploadFile


async def decode_upload_file_to_ndarray(image: UploadFile):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    return img