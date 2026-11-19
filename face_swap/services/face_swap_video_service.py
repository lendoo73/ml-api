from face_swap.repositories.face_swap_video_worker_repository import (
    FaceSwapVideoWorkerRepository,
)


class FaceSwapVideoService:
    def __init__(self, video_repository=None):
        self.video_repository = video_repository or FaceSwapVideoWorkerRepository()

    def swap_video(
        self,
        reference_bytes: bytes,
        video_path: str,
        progress_callback=None,
    ) -> str:
        return self.video_repository.swap_video(
            reference_bytes,
            video_path,
            progress_callback,
        )
