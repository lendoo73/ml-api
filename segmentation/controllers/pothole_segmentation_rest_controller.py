import os
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from segmentation.controllers.segmentation_rest_controller import plot_options
from segmentation.models.dtos.pothole_segmentation_dto import PotholeSegmentationDto
from segmentation.services.pothole_segmentation_service import PotholeSegmentationService
from services.video_job_manager import video_job_manager
from utils.literals import ImageOutput, TrackerType, VideoOutput
from utils.video import save_uploadfile_to_temp


@lru_cache
def get_service():
    return PotholeSegmentationService()


router = APIRouter(prefix="/segmentation/pothole", tags=["pothole segmentation"])


def _options(imgsz, classes, max_det, retina_masks):
    return {"imgsz": imgsz, "classes": classes, "max_det": max_det, "retina_masks": retina_masks}


@router.post("/predict", response_model=list[PotholeSegmentationDto])
async def predict_file(source: UploadFile = File(...), output: ImageOutput = Query("json"),
                       conf: float = Query(.5, ge=0, le=1), imgsz: int = Query(640, gt=0),
                       classes: list[int] | None = Query(None), max_det: int = Query(300, gt=0),
                       retina_masks: bool = Query(False), service=Depends(get_service)):
    result = await service.predict_image(source, output, plot_options(), conf, **_options(imgsz, classes, max_det, retina_masks))
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.get("/predict", response_model=list[PotholeSegmentationDto])
async def predict_url(source: AnyUrl, output: ImageOutput = Query("json"),
                      conf: float = Query(.5, ge=0, le=1), imgsz: int = Query(640, gt=0),
                      classes: list[int] | None = Query(None), max_det: int = Query(300, gt=0),
                      retina_masks: bool = Query(False), service=Depends(get_service)):
    result = await service.predict_url(str(source), output, plot_options(), conf, **_options(imgsz, classes, max_det, retina_masks))
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.websocket("/predict/webcam")
async def webcam(websocket: WebSocket):
    await websocket.accept()
    service = get_service()
    tracking = websocket.query_params.get("tracking", "false").lower() == "true"
    tracker: TrackerType = websocket.query_params.get("tracker", "bytetrack")
    tracking_executor = ThreadPoolExecutor(max_workers=1) if tracking else None
    try:
        while True:
            frame = await websocket.receive_bytes()
            if tracking_executor:
                result = await asyncio.get_running_loop().run_in_executor(
                    tracking_executor,
                    service.predict_frame_bytes,
                    frame,
                    .5,
                    True,
                    tracker,
                )
            else:
                result = await run_in_threadpool(service.predict_frame_bytes, frame, .5)
            await websocket.send_json([item.model_dump() for item in result])
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.tracking_service.reset_tracking()
        if tracking_executor:
            tracking_executor.shutdown(wait=False)


@router.post("/predict/video", response_model=dict)
async def video(source: UploadFile = File(...), output: VideoOutput = Query("json"),
                conf: float = Query(.5, ge=0, le=1), tracking: bool = Query(False),
                tracker: TrackerType = Query("bytetrack"), service=Depends(get_service)):
    if output == "video":
        path = await service.predict_annotated_video(
            source, conf=conf, tracking=tracking, tracker=tracker
        )
        return FileResponse(path, media_type="video/mp4", filename="pothole_segmentation.mp4",
                            background=BackgroundTask(os.remove, path))
    path = await save_uploadfile_to_temp(source)
    return {"job_id": video_job_manager.submit_path(
        path,
        lambda p, progress: service.predict_video_path(
            p, conf, progress, tracking=tracking, tracker=tracker
        ),
    )}


@router.get("/predict/video", response_model=dict)
async def video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    conf: float = Query(.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service=Depends(get_service),
):
    if output == "video":
        path = await service.predict_video_url(
            str(source), output, conf=conf, tracking=tracking, tracker=tracker
        )
        return FileResponse(path, media_type="video/mp4", filename="pothole_segmentation.mp4", background=BackgroundTask(os.remove, path))
    return {"job_id": video_job_manager.submit(
        lambda progress: service.predict_video_url(
            str(source), output, conf=conf, progress_callback=progress,
            tracking=tracking, tracker=tracker,
        )
    )}


@router.get("/predict/video/progress/{job_id}")
async def progress(job_id: str):
    from fastapi import HTTPException
    if video_job_manager.get(job_id) is None:
        raise HTTPException(404, "Video prediction job was not found.")
    from fastapi.responses import StreamingResponse
    return StreamingResponse(video_job_manager.stream(job_id), media_type="text/event-stream")
