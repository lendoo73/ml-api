import os
import tempfile
import time
from pathlib import Path

import requests


class FaceSwapVideoWorkerRepository:
    def __init__(self, worker_url: str | None = None):
        self._worker_url = (
            worker_url or os.getenv(
                "FACE_SWAP_VIDEO_WORKER_URL",
                "http://face-swap-video-worker:8001",
            )
        ).rstrip("/")
        self._session = requests.Session()

    def swap_video(
        self,
        reference_bytes: bytes,
        video_path: str,
        progress_callback=None,
    ) -> str:
        deadline = time.monotonic() + _positive_int(
            "FACE_SWAP_VIDEO_WORKER_TIMEOUT",
            14400,
        )
        job_id = self._submit_job(reference_bytes, video_path)
        while time.monotonic() < deadline:
            job = self._get_job(job_id)
            if job["status"] == "completed":
                return self._download_result(job_id)
            if job["status"] == "failed":
                raise RuntimeError(job.get("error") or "FaceFusion video job failed.")
            if progress_callback:
                progress_callback(int(job.get("progress", 0)), 100)
            time.sleep(1)
        raise TimeoutError("FaceFusion video job exceeded the configured timeout.")

    def _submit_job(self, reference_bytes: bytes, video_path: str) -> str:
        with open(video_path, "rb") as video_file:
            response = self._session.post(
                f"{self._worker_url}/jobs",
                files={
                    "source": ("portrait.jpg", reference_bytes, "image/jpeg"),
                    "target_video": (
                        Path(video_path).name,
                        video_file,
                        "video/mp4",
                    ),
                },
                timeout=(15, 600),
            )
        _raise_worker_error(response)
        job_id = response.json().get("job_id")
        if not job_id:
            raise RuntimeError("FaceFusion video worker returned no job ID.")
        return job_id

    def _get_job(self, job_id: str) -> dict:
        response = self._session.get(
            f"{self._worker_url}/jobs/{job_id}",
            timeout=15,
        )
        _raise_worker_error(response)
        return response.json()

    def _download_result(self, job_id: str) -> str:
        response = self._session.get(
            f"{self._worker_url}/jobs/{job_id}/result",
            stream=True,
            timeout=(15, 600),
        )
        try:
            _raise_worker_error(response)
            descriptor, output_path = tempfile.mkstemp(
                prefix="face_swap_",
                suffix=".mp4",
            )
            with os.fdopen(descriptor, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)
            if os.path.getsize(output_path) == 0:
                raise RuntimeError("FaceFusion video worker returned an empty result.")
        except Exception:
            if "output_path" in locals():
                os.remove(output_path)
            raise
        finally:
            response.close()
        return output_path


def _raise_worker_error(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.RequestException as error:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if detail:
            raise RuntimeError(f"FaceFusion video worker failed: {detail}") from error
        raise RuntimeError(
            f"FaceFusion video worker request failed with HTTP {response.status_code}."
        ) from error


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value
