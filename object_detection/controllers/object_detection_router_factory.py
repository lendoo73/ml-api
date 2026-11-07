import os
from functools import lru_cache
from typing import Type

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from object_detection.models.dtos.detection_dto import DetectionDto
from services.base_vision_service import BaseVisionService
from utils.literals import ColorMode, ImageOutput, TrackerType
from services.video_job_manager import video_job_manager
from utils.video import save_uploadfile_to_temp


CLASSES_DESCRIPTION = (
    "Optional zero-based class IDs or class names to include. Repeat this parameter to select multiple classes. "
    "IDs and names are specific to the model; see the [Ultralytics COCO class list]"
    "(https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) "
    "for COCO-trained models."
)


def create_object_detection_router(
    service_class: Type[BaseVisionService],
    prefix: str,
    tags: list[str],
) -> APIRouter:
    @lru_cache
    def get_service() -> BaseVisionService:
        return service_class()

    router = APIRouter(prefix=prefix, tags=tags)

    def plot_options(
        show_conf=True, line_width=None, font_size=None, font="Arial.ttf",
        labels=True, boxes=True, color_mode="class", txt_color=None,
    ):
        return {"conf": show_conf, "line_width": line_width, "font_size": font_size,
                "font": font, "labels": labels, "boxes": boxes,
                "color_mode": color_mode, **({"txt_color": txt_color} if txt_color is not None else {})}

    async def resolve_classes(service, classes):
        if classes is None:
            return None
        try:
            return await run_in_threadpool(service.model_repository.resolve_class_ids, classes)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.post("/predict", response_model=list[DetectionDto])
    async def predict_from_file(
        source: UploadFile = File(..., description="Image to analyze"),
        output: ImageOutput = Query("json"),
        conf: float = Query(0.5, ge=0, le=1),
        classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
        show_conf: bool = Query(True), line_width: int | None = Query(None),
        font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
        labels: bool = Query(True), boxes: bool = Query(True),
        color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] =         Query(None),
        service: BaseVisionService = Depends(get_service),
    ):
        classes = await resolve_classes(service, classes)
        result = await service.predict_image(
            source,
            output,
            plot_options(show_conf, line_width, font_size, font, labels, boxes, color_mode, txt_color),
            conf,
            classes,
        )
        if output == "image":
            return Response(content=result, media_type="image/jpeg")
        return result

    @router.get("/predict", response_model=list[DetectionDto])
    async def predict_from_url(
        source: AnyUrl,
        output: ImageOutput = Query("json"),
        conf: float = Query(0.5, ge=0, le=1),
        classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
        show_conf: bool = Query(True), line_width: int | None = Query(None),
        font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
        labels: bool = Query(True), boxes: bool = Query(True),
        color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] = Query(None),
        service: BaseVisionService = Depends(get_service),
    ):
        classes = await resolve_classes(service, classes)
        result = await service.predict_url(
            str(source),
            output,
            plot_options(show_conf, line_width, font_size, font, labels, boxes, color_mode, txt_color),
            conf,
            classes,
        )
        if output == "image":
            return Response(content=result, media_type="image/jpeg")
        return result

    @router.get("/predict/search", response_model=list[DetectionDto])
    async def predict_from_search(
        query: str = Query(..., min_length=1),
        output: ImageOutput = Query("json"),
        conf: float = Query(0.5, ge=0, le=1),
        classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
        show_conf: bool = Query(True), line_width: int | None = Query(None),
        font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
        labels: bool = Query(True), boxes: bool = Query(True),
        color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] = Query(None),
        service: BaseVisionService = Depends(get_service),
    ):
        classes = await resolve_classes(service, classes)
        try:
            result = await service.predict_search(
                query,
                output,
                plot_options(show_conf, line_width, font_size, font, labels, boxes, color_mode, txt_color),
                conf,
                classes,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if output == "image":
            return Response(content=result, media_type="image/jpeg")
        return result

    @router.websocket("/predict/webcam")
    async def predict_from_webcam(websocket: WebSocket):
        await websocket.accept()
        service = get_service()
        tracking = websocket.query_params.get("tracking", "false").lower() == "true"
        tracker = websocket.query_params.get("tracker", "bytetrack")
        try:
            classes = await run_in_threadpool(
                service.model_repository.resolve_class_ids,
                websocket.query_params.getlist("classes") or None,
            )
        except ValueError as error:
            await websocket.close(code=1008, reason=str(error))
            return
        if tracker not in {"bytetrack", "botsort"}:
            await websocket.close(code=1008, reason="Unsupported tracker")
            return
        if tracking and not hasattr(service, "tracking_service"):
            await websocket.close(code=1008, reason="Tracking is not supported for this model")
            return
        try:
            while True:
                frame = await websocket.receive_bytes()
                detections = await run_in_threadpool(
                    service.predict_frame_bytes, frame, 0.5, tracking, tracker,
                    classes,
                )
                await websocket.send_json([detection.model_dump() for detection in detections])
        except WebSocketDisconnect:
            return
        finally:
            if tracking and hasattr(service, "tracking_service"):
                service.tracking_service.reset_tracking()

    @router.post("/predict/video")
    async def predict_from_video(
        source: UploadFile = File(..., description="Video to analyze"),
        output: str = Query("json"),
        conf: float = Query(0.5, ge=0, le=1),
        show_conf: bool = Query(True), line_width: int | None = Query(None),
        font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
        labels: bool = Query(True), boxes: bool = Query(True),
        color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] = Query(None),
        tracking: bool = Query(False),
        tracker: TrackerType = Query("bytetrack"),
        classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
        service: BaseVisionService = Depends(get_service),
    ):
        classes = await resolve_classes(service, classes)
        if output == "video":
            video_path = await service.predict_annotated_video(
                source, plot_options(show_conf, line_width, font_size, font, labels, boxes, color_mode, txt_color),
                conf, tracking=tracking, tracker=tracker, classes=classes,
            )
            return FileResponse(
                video_path,
                media_type="video/mp4",
                filename="object_detection.mp4",
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
        output: str = Query("json"),
        conf: float = Query(0.5, ge=0, le=1),
        tracking: bool = Query(False),
        tracker: TrackerType = Query("bytetrack"),
        classes: list[str] | None = Query(None, description=CLASSES_DESCRIPTION),
        service: BaseVisionService = Depends(get_service),
    ):
        classes = await resolve_classes(service, classes)
        if output == "video":
            result = await service.predict_video_url(
                str(source), output, conf=conf, tracking=tracking, tracker=tracker, classes=classes
            )
            return FileResponse(
                result,
                media_type="video/mp4",
                filename="object_detection.mp4",
                background=BackgroundTask(os.remove, result),
            )
        return {
            "job_id": video_job_manager.submit(
                lambda progress: service.predict_video_url(
                    str(source), output, conf=conf, progress_callback=progress,
                    tracking=tracking, tracker=tracker,
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

    return router
