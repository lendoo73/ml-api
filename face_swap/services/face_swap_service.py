import os
import shutil
import subprocess
import tempfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from face_swap.models.face_swap_model import (
    FaceSwapModel,
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
)
from face_swap.repositories.qwen_image_21_gguf_q3_k_xl_local_face_swap_repository import (
    QwenImage21GgufQ3KxlLocalFaceSwapRepository,
)
from face_swap.repositories.qwen_image_21_bf16_local_face_swap_repository import (
    QwenImage21Bf16LocalFaceSwapRepository,
)


_MODEL_REPOSITORY_FACTORIES = {
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL: QwenImage21Bf16LocalFaceSwapRepository,
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL: (
        QwenImage21GgufQ3KxlLocalFaceSwapRepository
    ),
}


class FaceSwapService:
    def __init__(
        self,
        model_repository=None,
        model: FaceSwapModel = QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
    ):
        if model_repository is None:
            try:
                repository_factory = _MODEL_REPOSITORY_FACTORIES[model]
            except KeyError as error:
                raise ValueError(f"Unsupported face-swap model: {model}") from error
            model_repository = repository_factory()
        self.model_repository = model_repository

    def swap_image(self, reference_bytes: bytes, target_bytes: bytes) -> bytes:
        reference = self.decode_reference_image(reference_bytes)
        target = self._decode_image(target_bytes, "target")
        return self._encode_jpeg(self.model_repository.swap(target, reference))

    def decode_reference_image(self, image_bytes: bytes) -> Image.Image:
        return self._decode_image(image_bytes, "portrait")

    def swap_frame(self, reference: Image.Image, frame_bytes: bytes) -> bytes:
        target = self._decode_image(frame_bytes, "webcam frame")
        return self._encode_jpeg(self.model_repository.swap(target, reference))

    def swap_video(self, reference_bytes: bytes, video_path: str) -> str:
        reference = self.decode_reference_image(reference_bytes)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Could not open target video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Target video has invalid dimensions.")

        raw_path = self._temporary_video_path("face_swap_frames_")
        output_path = self._temporary_video_path("face_swap_")
        writer = cv2.VideoWriter(
            raw_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            _remove_files(raw_path, output_path)
            raise RuntimeError("Could not initialize the video encoder.")

        try:
            try:
                processed_frames = 0
                while True:
                    has_frame, frame = capture.read()
                    if not has_frame:
                        break
                    target = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    swapped = self.model_repository.swap(target, reference)
                    swapped = swapped.resize((width, height), Image.Resampling.LANCZOS)
                    writer.write(cv2.cvtColor(np.asarray(swapped), cv2.COLOR_RGB2BGR))
                    processed_frames += 1
            finally:
                capture.release()
                writer.release()
        except Exception:
            _remove_files(raw_path, output_path)
            raise

        if processed_frames == 0:
            _remove_files(raw_path, output_path)
            raise ValueError("Target video contains no readable frames.")
        if shutil.which("ffmpeg") is None:
            _remove_files(raw_path, output_path)
            raise RuntimeError("ffmpeg is required to create the face-swapped video.")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", raw_path, "-i", video_path,
                    "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy",
                    "-c:a", "aac", "-shortest", output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            _remove_files(raw_path, output_path)
            raise RuntimeError(
                f"Could not mux the swapped video and original audio: {error.stderr[-500:]}"
            ) from error
        except OSError:
            _remove_files(output_path)
            raise
        finally:
            _remove_files(raw_path)
        return output_path

    @staticmethod
    def _decode_image(image_bytes: bytes, description: str) -> Image.Image:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"Invalid {description} image.") from error

    @staticmethod
    def _encode_jpeg(image: Image.Image) -> bytes:
        output = BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()

    @staticmethod
    def _temporary_video_path(prefix: str) -> str:
        file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4", delete=False)
        file.close()
        return file.name


def _remove_files(*paths: str):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
