import atexit
import base64
import math
import os
import shutil
import subprocess
import threading
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from utils.model_weights import model_weights_dir

WEIGHTS_DIR = model_weights_dir(
    "face-swap",
    "qwen-image-2.1-gguf-q3-k-xl-local",
)
DIFFUSION_MODEL_ID = "unsloth/Qwen-Image-2.1-GGUF"
DIFFUSION_MODEL_FILE = "qwen-image-2.1-Q3_K_XL.gguf"
TEXT_ENCODER_ID = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
TEXT_ENCODER_FILE = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
VISION_PROJECTOR_FILE = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
VAE_ID = "Comfy-Org/Qwen-Image-2.1"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"
BFS_ID = "Alissonerdx/BFS-Best-Face-Swap"
BFS_FILE = "bfs_head_v1.1_qwen_2.1.safetensors"
TURBO_ID = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
SERVER_URL_ENV = "FACE_SWAP_SD_CPP_URL"
SERVER_PATH_ENV = "FACE_SWAP_SD_CPP_SERVER_PATH"
OUTPUT_PIXEL_BUDGET = 512 * 512
_SERVER_LOCK = threading.RLock()
_SERVER_PROCESS = None
_SERVER_URL = None


class QwenImage21GgufQ3KxlLocalFaceSwapRepository:
    def __init__(self, server_url: str | None = None):
        self._session = requests.Session()
        self._server_url = (
            server_url or os.getenv(SERVER_URL_ENV, "")
        ).rstrip("/")

    def swap(self, target: Image.Image, reference: Image.Image) -> Image.Image:
        self._ensure_server()
        target_width, target_height = _output_dimensions(target.width, target.height)
        response = self._session.post(
            f"{self._server_url}/sdcpp/v1/img_gen",
            json={
                "prompt": (
                    "head_swap: replace the head in <image1> with the head from <image2>. "
                    "Preserve the scene, lighting, pose, and expression from <image1>, "
                    "and retain the identity and defining facial features of <image2>."
                ),
                "width": target_width,
                "height": target_height,
                "ref_image_args": (
                    "preset=qwen,vae_input_max_pixels=262144"
                ),
                "ref_images": [
                    _encode_png(target),
                    _encode_png(reference),
                ],
                "sample_params": {
                    "sample_method": "euler",
                    "sample_steps": 6,
                    "custom_sigmas": [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0],
                    "guidance": {"txt_cfg": 1.0},
                },
                "lora": [
                    {"path": BFS_FILE, "multiplier": 1.0, "is_high_noise": False},
                    {"path": TURBO_FILE, "multiplier": 1.0, "is_high_noise": False},
                ],
                "output_format": "png",
            },
            timeout=30,
        )
        response.raise_for_status()
        job_id = response.json().get("id")
        if not job_id:
            raise RuntimeError("stable-diffusion.cpp returned no face-swap job ID.")
        return self._wait_for_result(job_id)

    def _wait_for_result(self, job_id: str) -> Image.Image:
        timeout = int(os.getenv("FACE_SWAP_SD_CPP_TIMEOUT", "1800"))
        if timeout <= 0:
            raise ValueError("FACE_SWAP_SD_CPP_TIMEOUT must be positive.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self._session.get(
                f"{self._server_url}/sdcpp/v1/jobs/{job_id}",
                timeout=15,
            )
            response.raise_for_status()
            job = response.json()
            if job.get("status") == "completed":
                images = (job.get("result") or {}).get("images") or []
                if not images or not images[0].get("b64_json"):
                    raise RuntimeError(
                        "stable-diffusion.cpp completed the job without an image."
                    )
                image_bytes = base64.b64decode(images[0]["b64_json"], validate=True)
                with Image.open(BytesIO(image_bytes)) as image:
                    return image.convert("RGB")
            if job.get("status") in {"failed", "cancelled"}:
                error = job.get("error") or {}
                message = error.get("message", job.get("status"))
                raise RuntimeError(f"stable-diffusion.cpp face swap failed: {message}")
            time.sleep(1)
        raise TimeoutError(
            f"stable-diffusion.cpp face swap exceeded {timeout} seconds."
        )

    def _ensure_server(self) -> None:
        if self._server_url:
            return
        global _SERVER_PROCESS, _SERVER_URL
        with _SERVER_LOCK:
            if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
                self._server_url = _SERVER_URL
                return
            server_path = _resolve_server_path()
            paths = _download_model_files()
            port = int(os.getenv("FACE_SWAP_SD_CPP_PORT", "1235"))
            if not 1 <= port <= 65535:
                raise ValueError("FACE_SWAP_SD_CPP_PORT must be between 1 and 65535.")
            command = [
                server_path,
                "--listen-port",
                str(port),
                "--diffusion-model",
                str(paths["diffusion_model"]),
                "--vae",
                str(paths["vae"]),
                "--llm",
                str(paths["text_encoder"]),
                "--llm_vision",
                str(paths["vision_projector"]),
                "--lora-model-dir",
                str(paths["loras"]),
                "--lora-apply-mode",
                "at_runtime",
                "--offload-to-cpu",
                "--model-args",
                "qwen_image_2_1_prefix_cache_type=q8_0",
            ]
            log_path = WEIGHTS_DIR / "sd-server.log"
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open("a", encoding="utf-8") as log_file:
                    _SERVER_PROCESS = subprocess.Popen(
                        command,
                        cwd=WEIGHTS_DIR,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
            except OSError as error:
                raise RuntimeError(
                    f"Could not start stable-diffusion.cpp server: {error}"
                ) from error
            _SERVER_URL = f"http://127.0.0.1:{port}"
            self._server_url = _SERVER_URL
            atexit.register(_stop_server)
            try:
                self._wait_for_server(log_path)
            except Exception:
                _stop_server()
                raise

    def _wait_for_server(self, log_path: Path) -> None:
        timeout = int(os.getenv("FACE_SWAP_SD_CPP_STARTUP_TIMEOUT", "900"))
        if timeout <= 0:
            raise ValueError("FACE_SWAP_SD_CPP_STARTUP_TIMEOUT must be positive.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _SERVER_PROCESS is None or _SERVER_PROCESS.poll() is not None:
                raise RuntimeError(
                    "stable-diffusion.cpp server exited during startup. "
                    f"Recent log output: {_read_log_tail(log_path)}"
                )
            try:
                response = self._session.get(
                    f"{self._server_url}/sdcpp/v1/models/status",
                    timeout=5,
                )
                if response.ok:
                    status = response.json()
                    if status.get("state") == "ready":
                        return
                    if status.get("state") == "error":
                        raise RuntimeError(
                            "stable-diffusion.cpp model load failed: "
                            f"{status.get('error') or _read_log_tail(log_path)}"
                        )
            except requests.RequestException:
                pass
            time.sleep(2)
        _stop_server()
        raise TimeoutError(
            "stable-diffusion.cpp model did not load before the startup timeout. "
            f"Recent log output: {_read_log_tail(log_path)}"
        )


def _download_model_files() -> dict[str, Path]:
    diffusion_dir = WEIGHTS_DIR / "diffusion_models"
    text_encoder_dir = WEIGHTS_DIR / "text_encoders"
    lora_dir = WEIGHTS_DIR / "loras"
    assets = {
        "diffusion_model": (
            DIFFUSION_MODEL_ID,
            DIFFUSION_MODEL_FILE,
            diffusion_dir,
            diffusion_dir / DIFFUSION_MODEL_FILE,
        ),
        "text_encoder": (
            TEXT_ENCODER_ID,
            TEXT_ENCODER_FILE,
            text_encoder_dir,
            text_encoder_dir / TEXT_ENCODER_FILE,
        ),
        "vision_projector": (
            TEXT_ENCODER_ID,
            VISION_PROJECTOR_FILE,
            text_encoder_dir,
            text_encoder_dir / VISION_PROJECTOR_FILE,
        ),
        "vae": (
            VAE_ID,
            VAE_FILE,
            WEIGHTS_DIR,
            WEIGHTS_DIR / VAE_FILE,
        ),
        "bfs_lora": (
            BFS_ID,
            BFS_FILE,
            lora_dir,
            lora_dir / BFS_FILE,
        ),
        "turbo_lora": (
            TURBO_ID,
            TURBO_FILE,
            lora_dir,
            lora_dir / TURBO_FILE,
        ),
    }
    missing = [asset for asset in assets.values() if not asset[3].is_file()]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Downloading the Q3_K_XL face-swap model requires huggingface_hub."
            ) from error
        for repo_id, filename, local_dir, path in missing:
            local_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            if not path.is_file():
                raise RuntimeError(f"Downloaded model file is missing: {path}")
    return {
        "diffusion_model": assets["diffusion_model"][3],
        "text_encoder": assets["text_encoder"][3],
        "vision_projector": assets["vision_projector"][3],
        "vae": assets["vae"][3],
        "loras": lora_dir,
    }


def _resolve_server_path() -> str:
    configured_path = os.getenv(SERVER_PATH_ENV)
    if configured_path:
        if not Path(configured_path).is_file():
            raise RuntimeError(
                f"{SERVER_PATH_ENV} does not point to a file: {configured_path}"
            )
        return configured_path
    server_path = shutil.which("sd-server")
    if server_path is None:
        raise RuntimeError(
            "The Q3_K_XL face-swap model requires stable-diffusion.cpp. "
            f"Install sd-server or set {SERVER_PATH_ENV}."
        )
    return server_path


def _encode_png(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _output_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = math.sqrt(OUTPUT_PIXEL_BUDGET / (width * height))
    output_width = max(32, round(width * scale / 32) * 32)
    output_height = max(32, round(height * scale / 32) * 32)
    return output_width, output_height


def _read_log_tail(log_path: Path) -> str:
    if not log_path.is_file():
        return "No server log is available."
    return "\n".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    )


def _stop_server() -> None:
    global _SERVER_PROCESS, _SERVER_URL
    with _SERVER_LOCK:
        if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
            _SERVER_PROCESS.terminate()
            try:
                _SERVER_PROCESS.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _SERVER_PROCESS.kill()
                _SERVER_PROCESS.wait()
        _SERVER_PROCESS = None
        _SERVER_URL = None
