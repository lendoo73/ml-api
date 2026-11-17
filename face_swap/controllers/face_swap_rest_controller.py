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
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from face_swap.models.face_swap_model import (
    FaceSwapModel,
    QWEN_IMAGE_21_BF16_LOCAL_FACE_SWAP_MODEL,
    QWEN_IMAGE_21_GGUF_Q3_K_XL_LOCAL_FACE_SWAP_MODEL,
)
from face_swap.services.face_swap_service import FaceSwapService
from utils.execution_time import execution_time
from utils.video import save_uploadfile_to_temp


MODEL_DESCRIPTION = (
    "Face-swap model to use. Available models: "
    "'qwen-image-2.1-bf16-local' and "
    "'qwen-image-2.1-gguf-q3-k-xl-local'."
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
    try:
        output_path = await run_in_threadpool(
            service.swap_video,
            reference_bytes,
            video_path,
        )
    except ValueError as error:
        _remove_files(video_path)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:
        _remove_files(video_path)
        raise
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="face_swap.mp4",
        background=BackgroundTask(_remove_files, output_path, video_path),
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
