import os

from face_swap_video_worker.services.face_fusion_process_runner import (
    FaceFusionProcessRunner,
    create_temporary_path,
    remove_file,
)


class FaceFusionImageService:
    def __init__(self, process_runner=None):
        self._process_runner = process_runner or FaceFusionProcessRunner()

    def swap_image(self, source_path: str, target_path: str) -> str:
        output_path = create_temporary_path(".jpg")
        try:
            remove_file(output_path)
            self._process_runner.run(
                source_path,
                target_path,
                output_path,
                "image-to-image",
            )
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FaceFusion completed without creating an image.")
            return output_path
        except Exception:
            remove_file(output_path)
            raise
