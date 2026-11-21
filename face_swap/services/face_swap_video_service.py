from face_swap.repositories.face_swap_video_worker_repository import (
    FaceSwapVideoWorkerRepository,
)


import os
import tempfile

from face_swap_video_worker.services.face_fusion_video_service import (
    FaceFusionVideoService,
)


class FaceSwapVideoService:
    def __init__(self, facefusion_service=None):
        self.facefusion_service = facefusion_service or FaceFusionVideoService()

    def swap_video(
        self,
        reference_bytes: bytes,
        video_path: str,
        progress_callback=None,
    ) -> str:
        reference_path = _save_reference_image(reference_bytes)
        try:
            return self.facefusion_service.swap_video(
                reference_path,
                video_path,
                progress_callback,
            )
        finally:
            _remove_file(reference_path)


def _save_reference_image(reference_bytes: bytes) -> str:
    descriptor, path = tempfile.mkstemp(prefix="facefusion_source_", suffix=".jpg")
    try:
        with os.fdopen(descriptor, "wb") as image_file:
            image_file.write(reference_bytes)
    except Exception:
        _remove_file(path)
        raise
    return path


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
