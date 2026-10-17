import os
from functools import lru_cache
from typing import Type
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import AnyUrl
from starlette.background import BackgroundTask
from services.base_vision_service import BaseVisionService
from services.video_job_manager import video_job_manager
from utils.literals import ImageOutput, VideoOutput
from utils.video import save_uploadfile_to_temp
from oriented_bounding_box.models.dtos.oriented_bounding_box_dto import OrientedBoundingBoxDto
from oriented_bounding_box.models.dtos.oriented_tracking_dto import OrientedTrackingDto
from utils.literals import TrackerType


def create_oriented_bounding_box_router(service_class: Type[BaseVisionService], prefix: str, tags: list[str]):
    @lru_cache
    def get_service(): return service_class()
    router = APIRouter(prefix=prefix, tags=tags)

    def plots(show_conf=True, line_width=None, font_size=None, font="Arial.ttf", labels=True, boxes=True):
        return {"conf": show_conf, "line_width": line_width, "font_size": font_size,
                "font": font, "labels": labels, "boxes": boxes}

    @router.post("/predict", response_model=list[OrientedBoundingBoxDto])
    async def image_file(source: UploadFile = File(...), output: ImageOutput = Query("json"),
                         conf: float = Query(.5, ge=0, le=1), show_conf: bool = True,
                         line_width: int | None = Query(None, ge=1), font_size: int | None = Query(None, ge=1),
                         font: str = "Arial.ttf", labels: bool = True, boxes: bool = True,
                         service=Depends(get_service)):
        result = await service.predict_image(source, output, plots(show_conf, line_width, font_size, font, labels, boxes), conf)
        return Response(result, media_type="image/jpeg") if output == "image" else result

    @router.get("/predict", response_model=list[OrientedBoundingBoxDto])
    async def image_url(source: AnyUrl, output: ImageOutput = Query("json"), conf: float = Query(.5, ge=0, le=1),
                        show_conf: bool = True, line_width: int | None = Query(None, ge=1),
                        font_size: int | None = Query(None, ge=1), font: str = "Arial.ttf",
                        labels: bool = True, boxes: bool = True, service=Depends(get_service)):
        result = await service.predict_url(str(source), output, plots(show_conf, line_width, font_size, font, labels, boxes), conf)
        return Response(result, media_type="image/jpeg") if output == "image" else result

    @router.websocket("/predict/webcam")
    async def webcam(websocket: WebSocket):
        await websocket.accept()
        tracking = websocket.query_params.get("tracking", "false").lower() == "true"
        tracker = websocket.query_params.get("tracker", "bytetrack")
        if tracker not in {"bytetrack", "botsort"}:
            await websocket.close(code=1008, reason="Unsupported tracker")
            return
        try:
            while True:
                result = await run_in_threadpool(
                    get_service().predict_frame_bytes, await websocket.receive_bytes(),
                    0.5, tracking, tracker,
                )
                await websocket.send_json([item.model_dump() for item in result])
        except WebSocketDisconnect:
            return
        finally:
            if tracking:
                get_service().model_repository.reset_tracking()

    @router.post("/predict/video")
    async def video_file(source: UploadFile = File(...), output: VideoOutput = Query("json"),
                         conf: float = Query(.5, ge=0, le=1),
                         tracking: bool = Query(False), tracker: TrackerType = Query("bytetrack"),
                         service=Depends(get_service)):
        if output == "video":
            path = await service.predict_annotated_video(
                source, plots(), conf, tracking=tracking, tracker=tracker
            )
            return FileResponse(path, media_type="video/mp4", filename="oriented_bounding_box.mp4", background=BackgroundTask(os.remove, path))
        path = await save_uploadfile_to_temp(source)
        return {"job_id": video_job_manager.submit_path(
            path, lambda p, progress: service.predict_video_path(
                p, conf, progress, tracking=tracking, tracker=tracker
            )
        )}

    @router.get("/predict/video")
    async def video_url(source: AnyUrl, output: VideoOutput = Query("json"),
                        conf: float = Query(.5, ge=0, le=1),
                        tracking: bool = Query(False),
                        tracker: TrackerType = Query("bytetrack"),
                        service=Depends(get_service)):
        if output == "video":
            path = await service.predict_video_url(
                str(source), output, plots(), conf, tracking=tracking, tracker=tracker
            )
            return FileResponse(path, media_type="video/mp4", filename="oriented_bounding_box.mp4", background=BackgroundTask(os.remove, path))
        return {"job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url(
                str(source), output, plots(), conf, progress, tracking=tracking, tracker=tracker
            )
        )}

    @router.get("/predict/video/progress/{job_id}")
    async def progress(job_id: str):
        if video_job_manager.get(job_id) is None: raise HTTPException(404, "Video prediction job was not found.")
        return StreamingResponse(video_job_manager.stream(job_id), media_type="text/event-stream")
    return router
