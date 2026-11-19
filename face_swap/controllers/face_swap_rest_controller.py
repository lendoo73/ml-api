import os
from functools import lru_cache

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from face_swap.models.face_swap_model import (
    FaceSwapModel,
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
)
from face_swap.services.face_swap_service import FaceSwapService
from services.video_job_manager import video_job_manager
from utils.execution_time import execution_time
from utils.video import save_uploadfile_to_temp


MODEL_DESCRIPTION = (
    "Face-swap model to use. Available models: "
    "'qwen-image-2.1-bf16-local' and "
    "'qwen-image-2.1-gguf-q3-k-xl-local' and "
    "'qwen-image-2.1-huggingface-space'."
)


@lru_cache(maxsize=None)
def _cached_service(model: FaceSwapModel) -> FaceSwapService:
    return FaceSwapService(model=model)


def get_service(
    model: FaceSwapModel = Query(
        QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
        description=MODEL_DESCRIPTION,
    ),
) -> FaceSwapService:
    return _cached_service(model)


router = APIRouter(prefix="/face_swap", tags=["face swap"])


@router.post("/predict/image")
@execution_time("Face swap image execution time:")
async def predict_image(
    source: UploadFile = File(..., description="Portrait whose head will be transferred."),
    target: UploadFile = File(..., description="Target image whose head will be replaced."),
    service: FaceSwapService = Depends(get_service),
):
    try:
        image_bytes = await run_in_threadpool(
            service.swap_image,
            await source.read(),
            await target.read(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(content=image_bytes, media_type="image/jpeg")


@router.post("/predict/video")
@execution_time("Face swap video execution time:")
async def predict_video(
    source: UploadFile = File(..., description="Portrait whose head will be transferred."),
    target_video: UploadFile = File(..., description="Target video whose head will be replaced."),
    service: FaceSwapService = Depends(get_service),
):
    reference_bytes = await source.read()
    video_path = await save_uploadfile_to_temp(target_video)

    async def process_video(path, progress_callback):
        return await run_in_threadpool(
            service.swap_video,
            reference_bytes,
            path,
            progress_callback,
        )

    job_id = video_job_manager.submit_path(
        video_path,
        process_video,
        result_serializer=lambda completed_job_id, _: {
            "download_url": (
                f"/api/face_swap/predict/video/result/{completed_job_id}"
            )
        },
    )
    return {"job_id": job_id}


@router.get("/predict/video/progress/{job_id}")
async def video_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/predict/video/result/{job_id}")
async def video_result(job_id: str):
    job = video_job_manager.get(job_id)
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
        background=BackgroundTask(_remove_files, job.result),
    )


@router.websocket("/predict/webcam")
@execution_time("Face swap webcam session execution time:")
async def predict_webcam(
    websocket: WebSocket,
    model: FaceSwapModel = Query(
        QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
        description=MODEL_DESCRIPTION,
    ),
):
    await websocket.accept()
    service = get_service(model)
    try:
        reference_bytes = await websocket.receive_bytes()
        reference_image = await run_in_threadpool(
            service.decode_reference_image,
            reference_bytes,
        )
        while True:
            frame_bytes = await websocket.receive_bytes()
            swapped_frame = await run_in_threadpool(
                service.swap_frame,
                reference_image,
                frame_bytes,
            )
            await websocket.send_bytes(swapped_frame)
    except WebSocketDisconnect:
        return
    except ValueError as error:
        await websocket.close(code=1003, reason=str(error))


def _remove_files(*paths: str):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
