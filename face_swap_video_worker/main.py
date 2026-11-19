import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from face_swap_video_worker.services.face_fusion_video_service import (
    FaceFusionVideoService,
)
from services.video_job_manager import VideoJobManager


app = FastAPI(title="Face-swap video worker")
job_manager = VideoJobManager()
video_service = FaceFusionVideoService()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/jobs")
async def submit_video_job(
    source: UploadFile = File(...),
    target_video: UploadFile = File(...),
):
    source_path = await run_in_threadpool(_save_upload, source)
    try:
        target_path = await run_in_threadpool(_save_upload, target_video)
    except Exception:
        _remove_file(source_path)
        raise

    async def process_video(path, progress_callback):
        try:
            return await run_in_threadpool(
                video_service.swap_video,
                source_path,
                path,
                progress_callback,
            )
        finally:
            _remove_file(source_path)

    try:
        job_id = job_manager.submit_path(target_path, process_video)
    except Exception:
        _remove_file(source_path, target_path)
        raise
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_video_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
    }


@app.get("/jobs/{job_id}/result")
async def get_video_result(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=job.error)
    if job.status != "completed" or not isinstance(job.result, str):
        raise HTTPException(status_code=409, detail="Video prediction job is not complete.")
    if not os.path.isfile(job.result):
        raise HTTPException(status_code=404, detail="Video prediction result was not found.")
    return FileResponse(
        job.result,
        media_type="video/mp4",
        filename="face_swap.mp4",
        background=BackgroundTask(_remove_file, job.result),
    )


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    descriptor, path = tempfile.mkstemp(prefix="facefusion_input_", suffix=suffix)
    os.close(descriptor)
    try:
        upload.file.seek(0)
        with open(path, "wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        _remove_file(path)
        raise
    return path


def _remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
