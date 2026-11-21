import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from face_swap_video_worker import main as worker_main
import face_swap_video_worker.services.face_fusion_process_runner as process_runner_module
from face_swap_video_worker.services.face_fusion_image_service import (
    FaceFusionImageService,
)
from face_swap_video_worker.services.face_fusion_process_runner import (
    FaceFusionProcessRunner,
    _update_progress,
)
from face_swap_video_worker.services.face_fusion_video_service import (
    FaceFusionVideoService,
)


class FakeFaceFusionProcess:
    def __init__(self, command, **kwargs):
        self.stdout = io.BytesIO(b"processing: 20%|==| 2/10 [00:01]\r")
        self._poll_count = 0
        output_path = command[command.index("--output-path") + 1]
        Path(output_path).write_bytes(b"rendered-video")

    def poll(self):
        self._poll_count += 1
        return 0 if self._poll_count > 1 else None

    def wait(self):
        return 0

    def kill(self):
        return None


def test_facefusion_video_service_uses_reference_tracking_and_reports_progress(
    tmp_path,
):
    captured = {}
    progress = []

    class FakeProcessRunner:
        def run(
            self,
            source_path,
            target_path,
            output_path,
            workflow_mode,
            progress_callback=None,
        ):
            captured.update(
                source_path=source_path,
                target_path=target_path,
                output_path=output_path,
                workflow_mode=workflow_mode,
            )
            Path(output_path).write_bytes(b"rendered-video")
            progress_callback(2, 10)

    output_path = FaceFusionVideoService(FakeProcessRunner()).swap_video(
        "source.jpg",
        "target.mp4",
        lambda current, total: progress.append((current, total)),
    )
    try:
        assert captured["source_path"] == "source.jpg"
        assert captured["target_path"] == "target.mp4"
        assert captured["workflow_mode"] == "image-to-video"
        assert progress == [(2, 10)]
        assert Path(output_path).read_bytes() == b"rendered-video"
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_facefusion_image_service_runs_image_to_image_workflow(tmp_path):
    captured = {}

    class FakeProcessRunner:
        def run(
            self,
            source_path,
            target_path,
            output_path,
            workflow_mode,
            progress_callback=None,
        ):
            captured.update(
                source_path=source_path,
                target_path=target_path,
                workflow_mode=workflow_mode,
            )
            Path(output_path).write_bytes(b"rendered-image")

    output_path = FaceFusionImageService(FakeProcessRunner()).swap_image(
        "source.png",
        "target.png",
    )
    try:
        assert captured == {
            "source_path": "source.png",
            "target_path": "target.png",
            "workflow_mode": "image-to-image",
        }
        assert Path(output_path).read_bytes() == b"rendered-image"
    finally:
        Path(output_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("workflow_mode", "expected_video_flags"),
    [
        ("image-to-image", False),
        ("image-to-video", True),
    ],
)
def test_facefusion_process_runner_builds_workflow_command(
    tmp_path,
    monkeypatch,
    workflow_mode,
    expected_video_flags,
):
    root = tmp_path / "facefusion"
    root.mkdir()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeFaceFusionProcess(command, **kwargs)

    monkeypatch.setattr(process_runner_module, "FACEFUSION_ROOT", root)
    monkeypatch.setattr(process_runner_module, "FACEFUSION_PYTHON", "python")
    monkeypatch.setattr(process_runner_module, "_ensure_model_directory", lambda: None)
    monkeypatch.setattr(process_runner_module.subprocess, "Popen", fake_popen)
    output_path = str(tmp_path / "result")

    FaceFusionProcessRunner().run(
        "source.png",
        "target.mp4" if expected_video_flags else "target.png",
        output_path,
        workflow_mode,
    )

    command = captured["command"]
    assert command[command.index("--workflow-mode") + 1] == workflow_mode
    assert command[command.index("--face-selector-mode") + 1] == "reference"
    assert command[command.index("--execution-providers") + 1] == "cuda"
    assert ("--reference-frame-number" in command) is expected_video_flags
    assert ("--video-memory-strategy" in command) is expected_video_flags
    assert Path(output_path).read_bytes() == b"rendered-video"


def test_facefusion_progress_parser_ignores_non_progress_output():
    progress = []
    _update_progress("Loaded face swapper model.", progress.append)
    assert progress == []


def test_facefusion_environment_includes_cuda_dll_directories(tmp_path, monkeypatch):
    root = tmp_path / "facefusion"
    python = root / ".venv" / "Scripts" / "python.exe"
    cuda_dll_directory = root / ".venv" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
    cuda_dll_directory.mkdir(parents=True)
    monkeypatch.setattr(process_runner_module, "FACEFUSION_PYTHON", str(python))

    environment = process_runner_module._facefusion_environment()
    if process_runner_module.os.name == "nt":
        assert str(cuda_dll_directory) in environment["PATH"].split(
            process_runner_module.os.pathsep
        )
    else:
        assert "PATH" in environment


def test_facefusion_failure_includes_diagnostic_output(tmp_path, monkeypatch):
    root = tmp_path / "facefusion"
    root.mkdir()
    progress_output = "".join(
        f"analysing: {index}%|==| {index}/100 [00:01]\r"
        for index in range(1, 101)
    )
    output = (
        "ERROR: Face detector initialization failed.\r" + progress_output
    ).encode()

    class FailedFaceFusionProcess:
        def __init__(self, command, **kwargs):
            self.stdout = io.BytesIO(output)
            self._poll_count = 0

        def poll(self):
            self._poll_count += 1
            return 1 if self._poll_count > 1 else None

        def wait(self):
            return 1

        def kill(self):
            return None

    monkeypatch.setattr(process_runner_module, "FACEFUSION_ROOT", root)
    monkeypatch.setattr(process_runner_module, "FACEFUSION_PYTHON", "python")
    monkeypatch.setattr(
        process_runner_module,
        "_ensure_model_directory",
        lambda: None,
    )
    monkeypatch.setattr(
        process_runner_module.subprocess,
        "Popen",
        FailedFaceFusionProcess,
    )

    with pytest.raises(RuntimeError, match="FaceFusion failed with exit code 1") as error:
        FaceFusionProcessRunner().run(
            "source.jpg",
            "target.mp4",
            str(tmp_path / "output.mp4"),
            "image-to-video",
        )

    assert "Face detector initialization failed." in str(error.value)


class FakeVideoService:
    def swap_video(self, source_path, target_path, progress_callback):
        if progress_callback:
            progress_callback(1, 1)
        output_path = str(Path(target_path).with_name("worker-result.mp4"))
        Path(output_path).write_bytes(b"worker-video")
        return output_path


def test_worker_job_endpoints_return_progress_and_video(monkeypatch):
    monkeypatch.setattr(worker_main, "video_service", FakeVideoService())
    with TestClient(worker_main.app) as client:
        response = client.post(
            "/jobs",
            files={
                "source": ("portrait.jpg", b"portrait", "image/jpeg"),
                "target_video": ("target.mp4", b"video", "video/mp4"),
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        for _ in range(50):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
        assert job["status"] == "completed"
        result = client.get(f"/jobs/{job_id}/result")

    assert result.status_code == 200
    assert result.content == b"worker-video"


def test_worker_image_endpoint_returns_jpeg(monkeypatch, tmp_path):
    class FakeImageService:
        def swap_image(self, source_path, target_path):
            assert Path(source_path).read_bytes() == b"portrait"
            assert Path(target_path).read_bytes() == b"target-image"
            output_path = tmp_path / "worker-image.jpg"
            output_path.write_bytes(b"worker-image")
            return str(output_path)

    monkeypatch.setattr(worker_main, "image_service", FakeImageService())
    with TestClient(worker_main.app) as client:
        response = client.post(
            "/images",
            files={
                "source": ("portrait.png", b"portrait", "image/png"),
                "target_image": ("target.png", b"target-image", "image/png"),
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"worker-image"
