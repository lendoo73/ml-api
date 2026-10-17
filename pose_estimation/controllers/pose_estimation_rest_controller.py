import os
from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from pose_estimation.models.dtos.pose_dto import PoseDto
from pose_estimation.services.pose_estimation_service import PoseEstimationService
from utils.literals import ColorMode, ImageOutput, VideoOutput, TrackerType
from services.video_job_manager import video_job_manager
from utils.video import save_uploadfile_to_temp


@lru_cache
def get_service() -> PoseEstimationService:
    return PoseEstimationService()


router = APIRouter(prefix="/pose_estimation", tags=["pose estimation"])


def plot_options(
    show_conf: bool = True,
    line_width: int | None = None,
    font_size: int | None = None,
    font: str = "Arial.ttf",
    kpt_radius: int = 5,
    kpt_line: bool = True,
    labels: bool = True,
    boxes: bool = True,
    color_mode: ColorMode = "class",
    txt_color: tuple[int, int, int] | None = None,
) -> dict:
    return {
        "conf": show_conf,
        "line_width": line_width,
        "font_size": font_size,
        "font": font,
        "kpt_radius": kpt_radius,
        "kpt_line": kpt_line,
        "labels": labels,
        "boxes": boxes,
        "color_mode": color_mode,
        **({"txt_color": txt_color} if txt_color is not None else {}),
    }


@router.post("/predict", response_model=list[PoseDto])
async def predict_from_file(
    source: UploadFile = File(..., description="Image for pose estimation"),
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1, description="Minimum prediction confidence."),
    show_conf: bool = Query(True, description="Show confidence values in image labels."),
    line_width: int | None = Query(None),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    kpt_radius: int = Query(5, ge=1),
    kpt_line: bool = Query(True),
    labels: bool = Query(True),
    boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"),
    txt_color: tuple[int, int, int] | None = Query(None),
    service: PoseEstimationService = Depends(get_service),
):
    result = await service.predict_image(
        source,
        output,
        plot_options(
            show_conf,
            line_width,
            font_size,
            font,
            kpt_radius,
            kpt_line,
            labels,
            boxes,
            color_mode,
            txt_color,
        ),
        conf,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.get("/predict", response_model=list[PoseDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    show_conf: bool = Query(True), line_width: int | None = Query(None),
    font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
    kpt_radius: int = Query(5, ge=1), kpt_line: bool = Query(True),
    labels: bool = Query(True), boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] | None = Query(None),
    service: PoseEstimationService = Depends(get_service),
):
    result = await service.predict_url(
        str(source), output,
        plot_options(show_conf, line_width, font_size, font, kpt_radius, kpt_line, labels, boxes, color_mode, txt_color),
        conf,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    service = get_service()
    tracking = websocket.query_params.get("tracking", "false").lower() == "true"
    tracker = websocket.query_params.get("tracker", "bytetrack")
    try:
        while True:
            frame = await websocket.receive_bytes()
            poses = await run_in_threadpool(service.predict_frame_bytes, frame, 0.5, tracking, tracker)
            await websocket.send_json([pose.model_dump() for pose in poses])
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.tracking_service.reset_tracking()


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(..., description="Video for pose estimation"),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: PoseEstimationService = Depends(get_service),
):
    if output == "video":
        video_path = await (service.predict_tracked_annotated_video(source, conf=conf, tracker=tracker) if tracking else service.predict_annotated_video(source, conf=conf))
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="pose_estimation.mp4",
            background=BackgroundTask(os.remove, video_path),
        )
    video_path = await save_uploadfile_to_temp(source)
    return {"job_id": video_job_manager.submit_path(video_path, lambda path, progress: service.predict_tracked_video_path(path, conf, progress, tracker) if tracking else service.predict_video_path(path, conf, progress))}


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: PoseEstimationService = Depends(get_service),
):
    if output == "video":
        result = await (service.predict_tracked_video_url(str(source), output, conf=conf, tracker=tracker) if tracking else service.predict_video_url(str(source), output, conf=conf))
        return FileResponse(
            result,
            media_type="video/mp4",
            filename="pose_estimation.mp4",
            background=BackgroundTask(os.remove, result),
        )
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_tracked_video_url(str(source), output, conf=conf, progress_callback=progress, tracker=tracker) if tracking else service.predict_video_url(str(source), output, conf=conf, progress_callback=progress)
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
