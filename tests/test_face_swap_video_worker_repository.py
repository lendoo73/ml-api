from pathlib import Path

import face_swap.repositories.face_swap_video_worker_repository as repository_module
from face_swap.repositories.face_swap_video_worker_repository import (
    FaceSwapVideoWorkerRepository,
)


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self.payload = payload or {}
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise repository_module.requests.HTTPError(response=self)

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        return None


class FakeWorkerSession:
    def __init__(self, job_responses):
        self.job_responses = iter(job_responses)
        self.calls = []

    def post(self, url, files, timeout):
        source = files["source"][1]
        target = files["target_video"][1]
        self.calls.append(("post", url, source, target.read(), timeout))
        return FakeResponse({"job_id": "worker-job"})

    def get(self, url, **kwargs):
        self.calls.append(("get", url))
        if url.endswith("/result"):
            return FakeResponse(content=b"rendered-video")
        return next(self.job_responses)


def test_video_worker_repository_submits_polls_and_downloads_result(
    tmp_path,
    monkeypatch,
):
    target_path = tmp_path / "input.mp4"
    target_path.write_bytes(b"target-video")
    session = FakeWorkerSession(
        [
            FakeResponse({"status": "running", "progress": 35}),
            FakeResponse({"status": "completed", "progress": 100}),
        ]
    )
    repository = FaceSwapVideoWorkerRepository("http://worker:8001")
    repository._session = session
    progress = []
    monkeypatch.setattr(repository_module.time, "sleep", lambda _: None)

    output_path = repository.swap_video(
        b"source-image",
        str(target_path),
        lambda current, total: progress.append((current, total)),
    )
    try:
        assert Path(output_path).read_bytes() == b"rendered-video"
        assert progress == [(35, 100)]
        assert session.calls[0][0:4] == (
            "post",
            "http://worker:8001/jobs",
            b"source-image",
            b"target-video",
        )
        assert session.calls[-1] == (
            "get",
            "http://worker:8001/jobs/worker-job/result",
        )
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_video_worker_repository_surfaces_worker_job_failure(tmp_path):
    target_path = tmp_path / "input.mp4"
    target_path.write_bytes(b"target-video")
    session = FakeWorkerSession(
        [FakeResponse({"status": "failed", "error": "No face was detected."})]
    )
    repository = FaceSwapVideoWorkerRepository("http://worker:8001")
    repository._session = session

    try:
        repository.swap_video(b"source-image", str(target_path))
    except RuntimeError as error:
        assert str(error) == "No face was detected."
    else:
        raise AssertionError("Expected the worker error to be raised.")
