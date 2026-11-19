import threading
from pathlib import Path

from PIL import Image

from face_swap.models.face_swap_model import (
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
)
from utils.model_weights import model_weights_dir


WEIGHTS_DIR = model_weights_dir(
    "face-swap",
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
)
BASE_MODEL_ID = "Qwen/Qwen-Image-2.1"
BFS_REPO_ID = "Alissonerdx/BFS-Best-Face-Swap"
BFS_WEIGHT = "bfs_head_v1.1_qwen_2.1.safetensors"
TURBO_REPO_ID = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_WEIGHT = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
FACE_SWAP_PROMPT = (
    "head_swap: replace the head in <image1> with the head from <image2>. Preserve the scene, "
    "lighting, pose, and expression from <image1>, and retain the identity and defining facial "
    "features of <image2>."
)


class QwenImage21Bf16LocalFaceSwapRepository:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def swap(self, target: Image.Image, reference: Image.Image) -> Image.Image:
        with self._lock:
            pipeline, torch, calculate_dimensions = self._get_pipeline()
            width, height, _ = calculate_dimensions(
                1024 * 1024,
                target.width / target.height,
            )
            result = pipeline(
                prompt=FACE_SWAP_PROMPT,
                image=[target, reference],
                width=width,
                height=height,
                output_resolution=1024,
                num_inference_steps=6,
                sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
                true_cfg_scale=1.0,
                use_kv_cache=True,
                generator=torch.Generator(device="cuda").manual_seed(0),
            ).images[0]
            return result.convert("RGB")

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline, self._torch, self._calculate_dimensions = self._load_pipeline()
        return self._pipeline, self._torch, self._calculate_dimensions

    @staticmethod
    def _load_pipeline():
        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
            from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
                calculate_dimensions,
            )
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Face swapping requires the optional Qwen image-model dependencies."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("Face swapping requires a CUDA-enabled GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Face swapping requires a GPU with bfloat16 support.")

        pipeline = QwenImage21Pipeline.from_pretrained(
            BASE_MODEL_ID,
            cache_dir=str(WEIGHTS_DIR),
            torch_dtype=torch.bfloat16,
        )
        bfs_weights = load_file(
            hf_hub_download(
                BFS_REPO_ID,
                BFS_WEIGHT,
                cache_dir=str(WEIGHTS_DIR),
            )
        )
        pipeline.load_lora_weights(
            _split_fused_gate_up_weights(bfs_weights),
            adapter_name="bfs",
        )
        turbo_weights = load_file(
            hf_hub_download(
                TURBO_REPO_ID,
                TURBO_WEIGHT,
                cache_dir=str(WEIGHTS_DIR),
            )
        )
        pipeline.load_lora_weights(
            turbo_weights,
            adapter_name="turbo",
        )
        pipeline.set_adapters(["bfs", "turbo"], adapter_weights=[1.0, 1.0])
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            shift_terminal=None,
        )
        pipeline.to("cuda")
        return pipeline, torch, calculate_dimensions


def _split_fused_gate_up_weights(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if ".img_mlp.gate_up." not in key:
            converted[key] = value
            continue
        gate_key = key.replace(".gate_up.", ".gate_layer.")
        up_key = key.replace(".gate_up.", ".proj.")
        if "lora_B" in key:
            gate, up = value.chunk(2, dim=0)
            converted[gate_key] = gate.contiguous()
            converted[up_key] = up.contiguous()
        else:
            converted[gate_key] = value
            converted[up_key] = value.clone()
    return converted
