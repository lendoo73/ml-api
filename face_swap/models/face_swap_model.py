from enum import Enum


class FaceSwapModel(str, Enum):
    QWEN_IMAGE_21_BF16_LOCAL = "qwen-image-2.1-bf16-local"
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL = "qwen-image-2.1-gguf-q3-k-xl-local"
    QWEN_IMAGE_21_HUGGINGFACE_SPACE = "qwen-image-2.1-huggingface-space"


QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL = FaceSwapModel.QWEN_IMAGE_21_BF16_LOCAL
QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL = (
    FaceSwapModel.QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL
)
QWEN_IMAGE_21_HUGGINGFACE_SPACE_FACE_SWAP_MODEL = (
    FaceSwapModel.QWEN_IMAGE_21_HUGGINGFACE_SPACE
)
