import os
from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from segmentation.models.dtos.segmentation_dto import SegmentationDto
from segmentation.services.segmentation_service import SegmentationService
from utils.literals import ColorMode, ImageOutput, VideoOutput, TrackerType
from services.video_job_manager import video_job_manager
from utils.video import save_uploadfile_to_temp


@lru_cache
def get_service() -> SegmentationService:
    return SegmentationService()


router = APIRouter(prefix="/segmentation", tags=["segmentation"])
CLASSES_DESCRIPTION = (
    "Optional zero-based class IDs or class names to include. Repeat this parameter to select multiple classes. "
    "IDs and names are specific to the model; see the [Ultralytics COCO class list]"
    "(https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) "
    "for COCO-trained models. For custom models, use the names and ordering that the model was trained with."
)


async def resolve_classes(service: SegmentationService, classes: list[str] | None):
    if classes is None:
        return None
    try:
        return await run_in_threadpool(service.model_repository.resolve_class_ids, classes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def plot_options(
    show_conf: bool = True,
    line_width: int | None = None,
    font_size: int | None = None,
    font: str = "Arial.ttf",
    labels: bool = True,
    masks: bool = True,
    boxes: bool = True,
    color_mode: ColorMode = "class",
) -> dict:
    return {
        "conf": show_conf,
        "line_width": line_width,
        "font_size": font_size,
        "font": font,
        "labels": labels,
        "masks": masks,
        "boxes": boxes,
        "color_mode": color_mode,
    }


@router.post("/predict", response_model=list[SegmentationDto])
async def predict_from_file(
    source: UploadFile = File(..., description="Image to segment"),
    output: ImageOutput = Query("json", description="Return JSON or an annotated JPEG image."),
    conf: float = Query(0.5, ge=0, le=1, description="Minimum prediction confidence."),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    show_conf: bool = Query(True, description="Show confidence values in labels."),
    line_width: int | None = Query(None, description="Rendered line width in pixels."),
    font_size: int | None = Query(None, description="Rendered label text size in pixels."),
    font: str = Query("Arial.ttf", description="Font file used for labels."),
    labels: bool = Query(True, description="Render class labels."),
    masks: bool = Query(True, description="Render segmentation masks."),
    boxes: bool = Query(True, description="Render bounding boxes."),
    color_mode: ColorMode = Query("class", description="Use instance or class colors."),
    service: SegmentationService = Depends(get_service),
):
    classes = await resolve_classes(service, classes)
    result = await service.predict_image(
        source,
        output,
        plot_options(show_conf, line_width, font_size, font, labels, masks, boxes, color_mode),
        conf,
        classes,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.get("/predict", response_model=list[SegmentationDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    masks: bool = Query(True),
    boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"),
    service: SegmentationService = Depends(get_service),
):
    classes = await resolve_classes(service, classes)
    result = await service.predict_url(
        str(source),
        output,
        plot_options(show_conf, line_width, font_size, font, labels, masks, boxes, color_mode),
        conf,
        classes,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    service = get_service()
    tracking = websocket.query_params.get("tracking", "false").lower() == "true"
    tracker = websocket.query_params.get("tracker", "bytetrack")
    try:
        classes = await resolve_classes(
            service, websocket.query_params.getlist("classes") or None
        )
    except HTTPException as error:
        await websocket.close(code=1008, reason=str(error.detail))
        return
    try:
        while True:
            frame = await websocket.receive_bytes()
            segmentations = await run_in_threadpool(
                service.predict_frame_bytes, frame, 0.5, tracking, tracker, classes
            )
            await websocket.send_json([
                segmentation.model_dump() for segmentation in segmentations
            ])
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.tracking_service.reset_tracking()


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(..., description="Video to segment"),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: SegmentationService = Depends(get_service),
):
    classes = await resolve_classes(service, classes)
    if output == "video":
        video_path = await service.predict_annotated_video(
            source, conf=conf, tracking=tracking, tracker=tracker, classes=classes
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="segmentation.mp4",
            background=BackgroundTask(os.remove, video_path),
        )
    video_path = await save_uploadfile_to_temp(source)
    return {
        "job_id": video_job_manager.submit_path(
            video_path,
            lambda path, progress: service.predict_video_path(
                path, conf, progress, tracking=tracking, tracker=tracker, classes=classes
            ),
        )
    }


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: SegmentationService = Depends(get_service),
):
    classes = await resolve_classes(service, classes)
    if output == "video":
        result = await service.predict_video_url(
            str(source), output, conf=conf, tracking=tracking, tracker=tracker, classes=classes
        )
        return FileResponse(result, media_type="video/mp4", filename="segmentation.mp4",
                            background=BackgroundTask(os.remove, result))
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url(
                str(source),
                output,
                conf=conf,
                progress_callback=progress,
                tracking=tracking,
                tracker=tracker,
                classes=classes,
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
