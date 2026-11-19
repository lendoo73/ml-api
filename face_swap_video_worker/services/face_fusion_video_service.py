import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from utils.model_weights import model_weights_dir


FACEFUSION_ROOT = Path(os.getenv("FACEFUSION_ROOT", "/opt/facefusion"))
FACEFUSION_PYTHON = os.getenv(
    "FACEFUSION_PYTHON",
    str(FACEFUSION_ROOT / ".venv" / "bin" / "python"),
)
FACE_SWAP_MODEL = os.getenv("FACE_SWAP_VIDEO_MODEL", "inswapper_128")


class FaceFusionVideoService:
    def swap_video(
        self,
        source_path: str,
        target_path: str,
        progress_callback=None,
    ) -> str:
        output_path = _temporary_path(".mp4")
        try:
            _remove_file(output_path)
            self._run_facefusion(source_path, target_path, output_path, progress_callback)
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FaceFusion completed without creating a video.")
            return output_path
        except Exception:
            _remove_file(output_path)
            raise

    def _run_facefusion(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        progress_callback,
    ) -> None:
        _ensure_model_directory()
        command = [
            FACEFUSION_PYTHON,
            str(FACEFUSION_ROOT / "facefusion.py"),
            "headless-run",
            "--source-paths",
            source_path,
            "--target-path",
            target_path,
            "--output-path",
            output_path,
            "--processors",
            "face_swapper",
            "--face-swapper-model",
            FACE_SWAP_MODEL,
            "--face-selector-mode",
            "reference",
            "--reference-frame-number",
            "0",
            "--execution-providers",
            "cuda",
            "--video-memory-strategy",
            "moderate",
            "--log-level",
            "info",
        ]
        timeout = _positive_int("FACE_SWAP_VIDEO_TIMEOUT", 10800)
        _run_process(command, timeout, progress_callback)


def _run_process(command: list[str], timeout: int, progress_callback) -> None:
    try:
        process = subprocess.Popen(
            command,
            cwd=FACEFUSION_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as error:
        raise RuntimeError(f"Could not start FaceFusion: {error}") from error

    output_lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=_read_process_output,
        args=(process.stdout, output_lines),
        daemon=True,
    )
    reader.start()
    output_tail: list[str] = []
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None or reader.is_alive() or not output_lines.empty():
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise TimeoutError(f"FaceFusion exceeded the {timeout}-second timeout.")
            try:
                line = output_lines.get(timeout=0.25)
            except queue.Empty:
                continue
            output_tail.append(line)
            output_tail = output_tail[-20:]
            _update_progress(line, progress_callback)
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=1)
        if process.stdout:
            process.stdout.close()

    if return_code != 0:
        detail = "\n".join(output_tail)[-2000:]
        raise RuntimeError(f"FaceFusion failed with exit code {return_code}: {detail}")


def _read_process_output(stream, output_lines: queue.Queue[str]) -> None:
    remainder = bytearray()
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        for char in chunk:
            if char in (10, 13):
                if remainder:
                    output_lines.put(remainder.decode("utf-8", errors="replace"))
                    remainder.clear()
            else:
                remainder.append(char)
    if remainder:
        output_lines.put(remainder.decode("utf-8", errors="replace"))


def _update_progress(line: str, progress_callback) -> None:
    if progress_callback is None:
        return
    match = re.search(r"(\d+)\s*/\s*(\d+)", line)
    if match:
        progress_callback(int(match.group(1)), int(match.group(2)))


def _ensure_model_directory() -> None:
    assets_directory = model_weights_dir("face-swap", "facefusion") / ".assets"
    assets_directory.mkdir(parents=True, exist_ok=True)
    model_directory = FACEFUSION_ROOT / ".assets"
    if model_directory.is_symlink():
        if model_directory.resolve() != assets_directory.resolve():
            raise RuntimeError("FaceFusion model directory points to an unexpected location.")
    elif model_directory.exists():
        raise RuntimeError("FaceFusion model directory must be a symlink to persistent weights.")
    else:
        model_directory.symlink_to(assets_directory, target_is_directory=True)


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _temporary_path(suffix: str) -> str:
    descriptor, path = tempfile.mkstemp(prefix="facefusion_output_", suffix=suffix)
    os.close(descriptor)
    return path


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
