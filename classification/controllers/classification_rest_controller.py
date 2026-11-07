import os
from functools import lru_cache
from typing import Literal, get_args

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
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from classification.models.dtos.classification_dto import ClassificationDto
from classification.repositories.yolo_classification_model_repository import (
    ClassificationModel,
)
from classification.services.classification_service import ClassificationService
from services.video_job_manager import video_job_manager
from utils.literals import ImageOutput, VideoOutput
from utils.video import save_uploadfile_to_temp


MediaClassificationModel = Literal["yolo26x-cls", "weather"]
MEDIA_MODEL_NAMES = set(get_args(MediaClassificationModel))
router = APIRouter(prefix="/classification", tags=["classification"])


@lru_cache(maxsize=None)
def _cached_service(model: ClassificationModel) -> ClassificationService:
    return ClassificationService(model)


def get_image_service(
    model: ClassificationModel = "yolo26x-cls",
) -> ClassificationService:
    return _cached_service(model)


def get_media_service(
    model: MediaClassificationModel = "yolo26x-cls",
) -> ClassificationService:
    return _cached_service(model)


def _plot_options(
    font_size: int | None = None,
    font: str = "Arial.ttf",
    probs: bool = True,
    show: bool = False,
    txt_color: tuple[int, int, int] | None = None,
) -> dict:
    options = {
        "font_size": font_size,
        "font": font,
        "probs": probs,
        "show": show,
    }
    if txt_color is not None:
        options["txt_color"] = txt_color
    return options


@router.post("/predict", response_model=list[ClassificationDto])
async def predict_from_file(
    source: UploadFile = File(..., description="Image to classify"),
    output: ImageOutput = Query(
        "json", description="Return JSON predictions or a rendered JPEG image."
    ),
    conf: float = Query(0.5, ge=0, le=1, description="Minimum prediction confidence."),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    probs: bool = Query(True),
    show: bool = Query(False),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: ClassificationService = Depends(get_image_service),
):
    result = await service.predict_image(
        source, output, _plot_options(font_size, font, probs, show, txt_color), conf
    )
    if output == "image":
        return Response(content=result, media_type="image/jpeg")
    return result


@router.get("/predict", response_model=list[ClassificationDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query(
        "json", description="Return JSON predictions or a rendered JPEG image."
    ),
    conf: float = Query(0.5, ge=0, le=1, description="Minimum prediction confidence."),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    probs: bool = Query(True),
    show: bool = Query(False),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: ClassificationService = Depends(get_image_service),
):
    result = await service.predict_url(
        str(source), output, _plot_options(font_size, font, probs, show, txt_color), conf
    )
    if output == "image":
        return Response(content=result, media_type="image/jpeg")
    return result


@router.get("/predict/search", response_model=list[ClassificationDto])
async def predict_from_search(
    query: str = Query(..., min_length=1),
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    probs: bool = Query(True),
    show: bool = Query(False),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: ClassificationService = Depends(get_image_service),
):
    try:
        result = await service.predict_search(
            query, output, _plot_options(font_size, font, probs, show, txt_color), conf
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if output == "image":
        return Response(content=result, media_type="image/jpeg")
    return result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    model = websocket.query_params.get("model", "yolo26x-cls")
    if model not in MEDIA_MODEL_NAMES:
        await websocket.close(
            code=1008, reason=f"Unsupported model for webcam: {model}"
        )
        return
    service = _cached_service(model)
    try:
        while True:
            frame = await websocket.receive_bytes()
            classifications = await run_in_threadpool(service.predict_frame_bytes, frame)
            await websocket.send_json(
                [classification.model_dump() for classification in classifications]
            )
    except WebSocketDisconnect:
        return


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(..., description="Video to classify"),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    probs: bool = Query(True),
    show: bool = Query(False),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: ClassificationService = Depends(get_media_service),
):
    if output == "video":
        video_path = await service.predict_annotated_video(
            source, _plot_options(font_size, font, probs, show, txt_color), conf
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="classification.mp4",
            background=BackgroundTask(os.remove, video_path),
        )
    video_path = await save_uploadfile_to_temp(source)
    return {
        "job_id": video_job_manager.submit_path(
            video_path,
            lambda path, progress: service.predict_video_path(path, conf, progress),
        )
    }


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    probs: bool = Query(True),
    show: bool = Query(False),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: ClassificationService = Depends(get_media_service),
):
    plot_options = _plot_options(font_size, font, probs, show, txt_color)
    if output == "video":
        result = await service.predict_video_url(
            str(source), output, plot_options, conf
        )
        return FileResponse(
            result,
            media_type="video/mp4",
            filename="classification.mp4",
            background=BackgroundTask(os.remove, result),
        )
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url(
                str(source), output, plot_options, conf, progress
            )
        )
    }


@router.get("/predict/video/progress/{job_id}")
async def video_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
