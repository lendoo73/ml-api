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
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from oriented_bounding_box.models.dtos.oriented_bounding_box_dto import (
    OrientedBoundingBoxDto,
)
from oriented_bounding_box.repositories.yolo_oriented_bounding_box_model_repository import (
    MODEL_WEIGHT_PATHS,
    OrientedDetectionModel,
)
from oriented_bounding_box.services.oriented_bounding_box_service import (
    OrientedBoundingBoxService,
)
from services.video_job_manager import video_job_manager
from utils.literals import ImageOutput, TrackerType, VideoOutput
from utils.video import save_uploadfile_to_temp


router = APIRouter(
    prefix="/oriented_bounding_box",
    tags=["oriented bounding box"],
)


@lru_cache(maxsize=None)
def _cached_service(model: OrientedDetectionModel) -> OrientedBoundingBoxService:
    return OrientedBoundingBoxService(model)


def get_service(model: OrientedDetectionModel = "yolo26x-obb") -> OrientedBoundingBoxService:
    return _cached_service(model)


def _plots(show_conf=True, line_width=None, font_size=None, font="Arial.ttf", labels=True, boxes=True):
    return {
        "conf": show_conf,
        "line_width": line_width,
        "font_size": font_size,
        "font": font,
        "labels": labels,
        "boxes": boxes,
    }


@router.post("/predict", response_model=list[OrientedBoundingBoxDto])
async def predict_from_file(
    source: UploadFile = File(...),
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None, ge=1),
    font_size: int | None = Query(None, ge=1),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    boxes: bool = Query(True),
    service: OrientedBoundingBoxService = Depends(get_service),
):
    result = await service.predict_image(
        source, output, _plots(show_conf, line_width, font_size, font, labels, boxes), conf
    )
    if output == "image":
        return Response(content=result, media_type="image/jpeg")
    return result


@router.get("/predict", response_model=list[OrientedBoundingBoxDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None, ge=1),
    font_size: int | None = Query(None, ge=1),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    boxes: bool = Query(True),
    service: OrientedBoundingBoxService = Depends(get_service),
):
    result = await service.predict_url(
        str(source), output, _plots(show_conf, line_width, font_size, font, labels, boxes), conf
    )
    if output == "image":
        return Response(content=result, media_type="image/jpeg")
    return result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    model = websocket.query_params.get("model", "yolo26x-obb")
    if model not in MODEL_WEIGHT_PATHS:
        await websocket.close(code=1008, reason=f"Unsupported model: {model}")
        return
    service = _cached_service(model)
    tracking = websocket.query_params.get("tracking", "false").lower() == "true"
    tracker = websocket.query_params.get("tracker", "bytetrack")
    if tracker not in {"bytetrack", "botsort"}:
        await websocket.close(code=1008, reason="Unsupported tracker")
        return
    try:
        while True:
            frame = await websocket.receive_bytes()
            detections = await run_in_threadpool(
                service.predict_frame_bytes, frame, 0.5, tracking, tracker
            )
            await websocket.send_json([item.model_dump() for item in detections])
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.model_repository.reset_tracking()


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(...),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: OrientedBoundingBoxService = Depends(get_service),
):
    if output == "video":
        path = await service.predict_annotated_video(
            source, _plots(), conf, tracking=tracking, tracker=tracker
        )
        return FileResponse(
            path,
            media_type="video/mp4",
            filename="oriented_bounding_box.mp4",
            background=BackgroundTask(os.remove, path),
        )
    path = await save_uploadfile_to_temp(source)
    return {
        "job_id": video_job_manager.submit_path(
            path,
            lambda video_path, progress: service.predict_video_path(
                video_path, conf, progress, tracking=tracking, tracker=tracker
            ),
        )
    }


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: OrientedBoundingBoxService = Depends(get_service),
):
    if output == "video":
        path = await service.predict_video_url(
            str(source), output, _plots(), conf, tracking=tracking, tracker=tracker
        )
        return FileResponse(
            path,
            media_type="video/mp4",
            filename="oriented_bounding_box.mp4",
            background=BackgroundTask(os.remove, path),
        )
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url(
                str(source),
                output,
                _plots(),
                conf,
                progress,
                tracking=tracking,
                tracker=tracker,
            )
        )
    }


@router.get("/predict/video/progress/{job_id}")
async def video_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(
            status_code=404, detail="Video prediction job was not found."
        )
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
    )
