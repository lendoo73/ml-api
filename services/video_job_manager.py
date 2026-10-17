import asyncio
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from fastapi.encoders import jsonable_encoder


@dataclass
class VideoJob:
    job_id: str
    status: str = "pending"
    progress: int = 0
    result: Any = None
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


WorkerFactory = Callable[[Callable[[int, int | None], None]], Awaitable[Any]]


class VideoJobManager:
    def __init__(self):
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def submit(self, worker_factory: WorkerFactory) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = VideoJob(job_id=job_id)
        asyncio.create_task(self._run(job_id, worker_factory))
        return job_id

    def submit_path(self, video_path: str, processor: Callable[..., Awaitable[Any]]) -> str:
        async def worker(progress_callback):
            try:
                return await processor(video_path, progress_callback)
            finally:
                try:
                    os.remove(video_path)
                except OSError:
                    pass

        return self.submit(worker)

    async def _run(self, job_id: str, worker_factory: WorkerFactory):
        self._update(job_id, status="running")
        try:
            result = await worker_factory(
                lambda current, total=None: self.update_progress(job_id, current, total)
            )
            self._update(job_id, status="completed", progress=100, result=result)
        except Exception as error:
            self._update(job_id, status="failed", error=str(error))

    def update_progress(self, job_id: str, current: int, total: int | None = None):
        progress = min(99, round(current * 100 / total)) if total else 0
        self._update(job_id, progress=progress)

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return VideoJob(**job.__dict__)

    async def stream(self, job_id: str):
        while True:
            job = self.get(job_id)
            if job is None:
                return
            payload = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
            }
            if job.status == "completed":
                payload["result"] = jsonable_encoder(job.result)
            if job.status == "failed":
                payload["error"] = job.error
            yield f"data: {json.dumps(payload)}\n\n"
            if job.status in {"completed", "failed"}:
                return
            await asyncio.sleep(0.25)

    def _update(self, job_id: str, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()


video_job_manager = VideoJobManager()
