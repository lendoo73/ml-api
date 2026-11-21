from enum import Enum
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from face_swap.models.face_swap_model import (
    FaceSwapModel,
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
    QWEN_IMAGE_21_HUGGINGFACE_SPACE_FACE_SWAP_MODEL,
)
from face_swap.models.face_swap_image_model import FaceSwapImageModel
from face_swap.repositories.face_swap_image_worker_repository import (
    FaceSwapImageWorkerRepository,
)
from face_swap.repositories.qwen_image_21_gguf_q3_k_xl_local_face_swap_repository import (
    QwenImage21GgufQ3KxlLocalFaceSwapRepository,
)
from face_swap.repositories.qwen_image_21_bf16_local_face_swap_repository import (
    QwenImage21Bf16LocalFaceSwapRepository,
)
from face_swap.repositories.qwen_image_21_huggingface_space_face_swap_repository import (
    QwenImage21HuggingFaceSpaceFaceSwapRepository,
)


_MODEL_REPOSITORY_FACTORIES = {
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL: QwenImage21Bf16LocalFaceSwapRepository,
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL: (
        QwenImage21GgufQ3KxlLocalFaceSwapRepository
    ),
    QWEN_IMAGE_21_HUGGINGFACE_SPACE_FACE_SWAP_MODEL: (
        QwenImage21HuggingFaceSpaceFaceSwapRepository
    ),
    FaceSwapImageModel.FACEFUSION_3_9_2_LOCAL.value: FaceSwapImageWorkerRepository,
}


class FaceSwapService:
    def __init__(
        self,
        model_repository=None,
        model: FaceSwapModel | FaceSwapImageModel = (
            QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL
        ),
    ):
        if model_repository is None:
            model_name = model.value if isinstance(model, Enum) else model
            try:
                repository_factory = _MODEL_REPOSITORY_FACTORIES[model_name]
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
