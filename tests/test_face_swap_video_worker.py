import io
from pathlib import Path

from fastapi.testclient import TestClient

from face_swap_video_worker import main as worker_main
import face_swap_video_worker.services.face_fusion_video_service as video_service_module
from face_swap_video_worker.services.face_fusion_video_service import (
    FaceFusionVideoService,
    _update_progress,
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
    monkeypatch,
):
    root = tmp_path / "facefusion"
    root.mkdir()
    captured = {}
    progress = []

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeFaceFusionProcess(command, **kwargs)

    monkeypatch.setattr(video_service_module, "FACEFUSION_ROOT", root)
    monkeypatch.setattr(video_service_module, "FACEFUSION_PYTHON", "python")
    monkeypatch.setattr(video_service_module, "_ensure_model_directory", lambda: None)
    monkeypatch.setattr(video_service_module.subprocess, "Popen", fake_popen)

    output_path = FaceFusionVideoService().swap_video(
        "source.jpg",
        "target.mp4",
        lambda current, total: progress.append((current, total)),
    )
    try:
        command = captured["command"]
        assert command[command.index("--face-selector-mode") + 1] == "reference"
        assert command[command.index("--reference-frame-number") + 1] == "0"
        assert command[command.index("--execution-providers") + 1] == "cuda"
        assert command[command.index("--target-path") + 1] == "target.mp4"
        assert progress == [(2, 10)]
        assert Path(output_path).read_bytes() == b"rendered-video"
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_facefusion_progress_parser_ignores_non_progress_output():
    progress = []
    _update_progress("Loaded face swapper model.", progress.append)
    assert progress == []


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
