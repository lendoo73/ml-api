import os
import shutil
import tempfile

import requests

from exceptions import InvalidVideoSourceError

async def save_uploadfile_to_temp(upload) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.split(".")[-1].lower()
    fd, temp_path = tempfile.mkstemp(prefix="pc_video_", suffix=suffix)
    os.close(fd)
    upload.file.seek(0)
    with open(temp_path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
    upload.file.seek(0)
    return temp_path


def download_video_to_temp(url: str, timeout: int = 30) -> str:
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise InvalidVideoSourceError(
            "Video URL could not be downloaded. Provide a direct, publicly accessible video URL."
        ) from error
    suffix = ".mp4"
    content_type = response.headers.get("content-type", "")
    if "webm" in content_type:
        suffix = ".webm"
    fd, temp_path = tempfile.mkstemp(prefix="video_url_", suffix=suffix)
    os.close(fd)
    try:
        with open(temp_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    video_file.write(chunk)
    except Exception:
        os.remove(temp_path)
        raise
    return temp_path