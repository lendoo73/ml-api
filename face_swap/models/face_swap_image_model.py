from enum import Enum


class FaceSwapImageModel(str, Enum):
    QWEN_IMAGE_21_BF16_LOCAL = "qwen-image-2.1-bf16-local"
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL = "qwen-image-2.1-gguf-q3-k-xl-local"
    QWEN_IMAGE_21_HUGGINGFACE_SPACE = "qwen-image-2.1-huggingface-space"
    FACEFUSION_3_9_2_LOCAL = "facefusion-3.9.2-local"
