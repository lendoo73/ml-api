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

from segmentation.models.dtos.segmentation_dto import SegmentationDto
from segmentation.repositories.yolo_segmentation_model_repository import (
    MODEL_WEIGHT_PATHS,
    SegmentationModel,
)
from segmentation.services.brain_tumor_segmentation_service import (
    BrainTumorSegmentationService,
)
from segmentation.services.pothole_segmentation_service import PotholeSegmentationService
from segmentation.services.segmentation_service import SegmentationService
from services.base_vision_service import BaseVisionService
from services.video_job_manager import video_job_manager
from utils.literals import ColorMode, ImageOutput, TrackerType, VideoOutput
from utils.video import save_uploadfile_to_temp


ImageSegmentationModel = Literal["yolo26x-seg", "brain-tumor", "pothole"]
MediaSegmentationModel = Literal["yolo26x-seg", "pothole"]
BrainTumorModel = Literal["brain-tumor"]
MEDIA_MODEL_NAMES = set(get_args(MediaSegmentationModel))
router = APIRouter(prefix="/segmentation", tags=["segmentation"])
CLASSES_DESCRIPTION = (
    "Optional zero-based class IDs or class names to include. Repeat this parameter to select multiple classes. "
    "IDs and names are specific to the selected model; see the [Ultralytics COCO class list]"
    "(https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) "
    "for COCO-trained models. Pothole classes must be numeric IDs."
)
_SERVICE_CLASSES = {
    "yolo26x-seg": SegmentationService,
    "brain-tumor": BrainTumorSegmentationService,
    "pothole": PotholeSegmentationService,
}


@lru_cache(maxsize=None)
def _cached_service(model: SegmentationModel) -> BaseVisionService:
    return _SERVICE_CLASSES[model](model)


def get_image_service(
    model: ImageSegmentationModel = "yolo26x-seg",
) -> BaseVisionService:
    return _cached_service(model)


def get_media_service(
    model: MediaSegmentationModel = "yolo26x-seg",
) -> BaseVisionService:
    return _cached_service(model)


def get_brain_tumor_service(
    model: BrainTumorModel = "brain-tumor",
) -> BrainTumorSegmentationService:
    return _cached_service(model)


async def _resolve_classes(service: BaseVisionService, classes: list[str] | None):
    if classes is None:
        return None
    try:
        return await run_in_threadpool(service.model_repository.resolve_class_ids, classes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _plot_options(
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


def _pothole_classes(classes: list[str] | None) -> list[int] | None:
    if classes is None:
        return None
    try:
        return [int(value) for value in classes]
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail="Pothole classes must be zero-based integer IDs."
        ) from error


def _pothole_options(imgsz, classes, max_det, retina_masks):
    return {
        "imgsz": imgsz or 640,
        "classes": _pothole_classes(classes),
        "max_det": max_det or 300,
        "retina_masks": bool(retina_masks),
    }


def _validate_model_options(model, classes, imgsz, max_det, retina_masks):
    if model == "brain-tumor" and classes is not None:
        raise HTTPException(
            status_code=422, detail="Class filtering is not supported for brain-tumor."
        )
    if model != "pothole" and any(
        option is not None for option in (imgsz, max_det, retina_masks)
    ):
        raise HTTPException(
            status_code=422,
            detail="imgsz, max_det, and retina_masks are supported only for pothole.",
        )


@router.post("/predict", response_model=list[SegmentationDto])
async def predict_from_file(
    source: UploadFile = File(..., description="Image to segment"),
    output: ImageOutput = Query("json", description="Return JSON or an annotated JPEG image."),
    model: ImageSegmentationModel = Query("yolo26x-seg"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    imgsz: int | None = Query(None, gt=0),
    max_det: int | None = Query(None, gt=0),
    retina_masks: bool | None = Query(None),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    masks: bool = Query(True),
    boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"),
    service: BaseVisionService = Depends(get_image_service),
):
    _validate_model_options(model, classes, imgsz, max_det, retina_masks)
    options = _pothole_options(imgsz, classes, max_det, retina_masks) if model == "pothole" else {}
    resolved_classes = None
    if model == "yolo26x-seg":
        resolved_classes = await _resolve_classes(service, classes)
    result = await service.predict_image(
        source,
        output,
        _plot_options(show_conf, line_width, font_size, font, labels, masks, boxes, color_mode),
        conf,
        **({"classes": resolved_classes} if model == "yolo26x-seg" else {}),
        **options,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.get("/predict", response_model=list[SegmentationDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query("json"),
    model: ImageSegmentationModel = Query("yolo26x-seg"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    imgsz: int | None = Query(None, gt=0),
    max_det: int | None = Query(None, gt=0),
    retina_masks: bool | None = Query(None),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    masks: bool = Query(True),
    boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"),
    service: BaseVisionService = Depends(get_image_service),
):
    _validate_model_options(model, classes, imgsz, max_det, retina_masks)
    options = _pothole_options(imgsz, classes, max_det, retina_masks) if model == "pothole" else {}
    resolved_classes = None
    if model == "yolo26x-seg":
        resolved_classes = await _resolve_classes(service, classes)
    result = await service.predict_url(
        str(source),
        output,
        _plot_options(show_conf, line_width, font_size, font, labels, masks, boxes, color_mode),
        conf,
        **({"classes": resolved_classes} if model == "yolo26x-seg" else {}),
        **options,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    model = websocket.query_params.get("model", "yolo26x-seg")
    if model not in MEDIA_MODEL_NAMES:
        await websocket.close(code=1008, reason=f"Unsupported model for webcam: {model}")
        return
    service = _cached_service(model)
    tracking = websocket.query_params.get("tracking", "false").lower() == "true"
    tracker = websocket.query_params.get("tracker", "bytetrack")
    classes = websocket.query_params.getlist("classes") or None
    try:
        if model == "yolo26x-seg":
            classes = await _resolve_classes(service, classes)
        elif classes is not None:
            classes = _pothole_classes(classes)
        while True:
            frame = await websocket.receive_bytes()
            segmentations = await run_in_threadpool(
                service.predict_frame_bytes, frame, 0.5, tracking, tracker, classes
            )
            await websocket.send_json([item.model_dump() for item in segmentations])
    except HTTPException as error:
        await websocket.close(code=1008, reason=str(error.detail))
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.tracking_service.reset_tracking()


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(..., description="Video to segment"),
    output: VideoOutput = Query("json"),
    model: MediaSegmentationModel = Query("yolo26x-seg"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: BaseVisionService = Depends(get_media_service),
):
    resolved_classes = (
        await _resolve_classes(service, classes) if model == "yolo26x-seg" else _pothole_classes(classes)
    )
    if output == "video":
        video_path = await service.predict_annotated_video(
            source, conf=conf, tracking=tracking, tracker=tracker, classes=resolved_classes
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
                path, conf, progress, tracking=tracking, tracker=tracker, classes=resolved_classes
            ),
        )
    }


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    model: MediaSegmentationModel = Query("yolo26x-seg"),
    conf: float = Query(0.5, ge=0, le=1),
    classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: BaseVisionService = Depends(get_media_service),
):
    resolved_classes = (
        await _resolve_classes(service, classes) if model == "yolo26x-seg" else _pothole_classes(classes)
    )
    if output == "video":
        result = await service.predict_video_url(
            str(source), output, conf=conf, tracking=tracking, tracker=tracker, classes=resolved_classes
        )
        return FileResponse(
            result,
            media_type="video/mp4",
            filename="segmentation.mp4",
            background=BackgroundTask(os.remove, result),
        )
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url(
                str(source),
                output,
                conf=conf,
                progress_callback=progress,
                tracking=tracking,
                tracker=tracker,
                classes=resolved_classes,
            )
        )
    }


@router.post("/predict/batch", response_model=dict)
async def predict_from_zip(
    source: UploadFile = File(..., description="ZIP archive containing images to segment"),
    model: BrainTumorModel = Query("brain-tumor"),
    service: BrainTumorSegmentationService = Depends(get_brain_tumor_service),
):
    archive_bytes = await source.read()
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_zip(archive_bytes, progress)
        )
    }


@router.get("/predict/batch/progress/{job_id}")
async def batch_prediction_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Prediction job was not found.")
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/predict/video/progress/{job_id}")
async def video_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
