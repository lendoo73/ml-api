import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from utils.model_weights import model_weights_dir


_PROGRESS_LINE = re.compile(r"^\s*[\w-]+:\s+\d+%")
FACEFUSION_ROOT = Path(os.getenv("FACEFUSION_ROOT", "/opt/facefusion"))
FACEFUSION_PYTHON = os.getenv(
    "FACEFUSION_PYTHON",
    str(FACEFUSION_ROOT / ".venv" / "bin" / "python"),
)
FACE_SWAP_MODEL = os.getenv("FACE_SWAP_VIDEO_MODEL", "inswapper_128")


class FaceFusionProcessRunner:
    def run(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        workflow_mode: str,
        progress_callback=None,
    ) -> None:
        if workflow_mode not in {"image-to-image", "image-to-video"}:
            raise ValueError(f"Unsupported FaceFusion workflow: {workflow_mode}")
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
            "--execution-providers",
            "cuda",
            "--workflow-mode",
            workflow_mode,
            "--log-level",
            "info",
        ]
        if workflow_mode == "image-to-video":
            command.extend(
                [
                    "--reference-frame-number",
                    "0",
                    "--video-memory-strategy",
                    "moderate",
                ]
            )

        timeout_name, default_timeout = _timeout_settings(workflow_mode)
        _run_process(
            command,
            _positive_int(timeout_name, default_timeout),
            progress_callback,
        )


def create_temporary_path(suffix: str) -> str:
    descriptor, path = tempfile.mkstemp(
        prefix="facefusion_output_",
        suffix=suffix,
    )
    os.close(descriptor)
    return path


def remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _timeout_settings(workflow_mode: str) -> tuple[str, int]:
    if workflow_mode == "image-to-image":
        return "FACE_SWAP_IMAGE_TIMEOUT", 1800
    return "FACE_SWAP_VIDEO_TIMEOUT", 10800


def _run_process(command: list[str], timeout: int, progress_callback) -> None:
    try:
        process = subprocess.Popen(
            command,
            cwd=FACEFUSION_ROOT,
            env=_facefusion_environment(),
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
    diagnostic_output: deque[str] = deque(maxlen=50)
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
            if not _PROGRESS_LINE.match(line):
                diagnostic_output.append(line)
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
        detail_lines = list(diagnostic_output)
        if not detail_lines:
            detail_lines.append("FaceFusion emitted no non-progress diagnostic output.")
        if output_tail:
            detail_lines.extend(["Recent FaceFusion output:", *output_tail[-5:]])
        detail = "\n".join(detail_lines)[-8000:]
        raise RuntimeError(f"FaceFusion failed with exit code {return_code}: {detail}")


def _facefusion_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if os.name == "nt":
        site_packages = (
            Path(FACEFUSION_PYTHON).resolve().parent.parent
            / "Lib"
            / "site-packages"
            / "nvidia"
        )
        dll_directories = [
            str(path)
            for path in site_packages.glob("*/bin")
            if path.is_dir()
        ]
        if dll_directories:
            environment["PATH"] = os.pathsep.join(
                [*dll_directories, environment.get("PATH", "")]
            )
    return environment


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
        if model_directory.resolve() != assets_directory.resolve():
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
