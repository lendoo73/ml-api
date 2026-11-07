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

from pose_estimation.models.dtos.pose_dto import PoseDto
from pose_estimation.services.pose_estimation_service import PoseEstimationService
from pose_estimation.services.pose_angle_service import (
    AngleDefinition,
    PoseAngleService,
)
from utils.literals import ColorMode, ImageOutput, VideoOutput, TrackerType
from services.video_job_manager import video_job_manager
from utils.video import save_uploadfile_to_temp


@lru_cache
def get_service() -> PoseEstimationService:
    return PoseEstimationService()


router = APIRouter(prefix="/pose_estimation", tags=["pose estimation"])
ANGLES_DESCRIPTION = (
    "Optional JSON-encoded list of three-keypoint lists. Each triplet uses all names "
    "or all zero-based COCO keypoint indices, with the middle keypoint as the pivot. "
    "Angle requests require output=json. Keypoint confidences are returned for client-side filtering."
)


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


def parse_angles(raw_angles: str | None) -> list[AngleDefinition] | None:
    try:
        return PoseAngleService.parse_requests(raw_angles)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def require_json_for_angles(output: str, angles: list[AngleDefinition] | None):
    if angles and output != "json":
        raise HTTPException(
            status_code=422,
            detail="Angle results require output=json.",
        )


@router.post("/predict", response_model=list[PoseDto])
async def predict_from_file(
    source: UploadFile = File(..., description="Image for pose estimation"),
    angles: str | None = Query(None, description=ANGLES_DESCRIPTION),
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
    angle_definitions = parse_angles(angles)
    require_json_for_angles(output, angle_definitions)
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
        angles=angle_definitions,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.get("/predict", response_model=list[PoseDto])
async def predict_from_url(
    source: AnyUrl,
    angles: str | None = Query(None, description=ANGLES_DESCRIPTION),
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    show_conf: bool = Query(True), line_width: int | None = Query(None),
    font_size: int | None = Query(None), font: str = Query("Arial.ttf"),
    kpt_radius: int = Query(5, ge=1), kpt_line: bool = Query(True),
    labels: bool = Query(True), boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"), txt_color: tuple[int, int, int] | None = Query(None),
    service: PoseEstimationService = Depends(get_service),
):
    angle_definitions = parse_angles(angles)
    require_json_for_angles(output, angle_definitions)
    result = await service.predict_url(
        str(source),
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
        angles=angle_definitions,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.websocket("/predict/webcam")
async def predict_from_webcam(websocket: WebSocket):
    await websocket.accept()
    service = get_service()
    try:
        angle_definitions = parse_angles(websocket.query_params.get("angles"))
    except HTTPException as error:
        await websocket.close(code=1008, reason=str(error.detail))
        return
    tracking = (
        websocket.query_params.get("tracking", "false").lower() == "true"
        or bool(angle_definitions)
    )
    tracker = websocket.query_params.get("tracker", "bytetrack")
    try:
        while True:
            frame = await websocket.receive_bytes()
            poses = await run_in_threadpool(
                service.predict_frame_bytes,
                frame,
                0.5,
                tracking,
                tracker,
                angle_definitions,
            )
            await websocket.send_json([pose.model_dump() for pose in poses])
    except WebSocketDisconnect:
        return
    finally:
        if tracking:
            service.tracking_service.reset_tracking()


@router.post("/predict/video", response_model=dict)
async def predict_from_video(
    source: UploadFile = File(..., description="Video for pose estimation"),
    angles: str | None = Query(None, description=ANGLES_DESCRIPTION),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: PoseEstimationService = Depends(get_service),
):
    angle_definitions = parse_angles(angles)
    require_json_for_angles(output, angle_definitions)
    tracking = tracking or bool(angle_definitions)
    if output == "video":
        video_path = await (
            service.predict_tracked_annotated_video(
                source, conf=conf, tracker=tracker
            )
            if tracking
            else service.predict_annotated_video(source, conf=conf)
        )
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="pose_estimation.mp4",
            background=BackgroundTask(os.remove, video_path),
        )
    video_path = await save_uploadfile_to_temp(source)
    return {
        "job_id": video_job_manager.submit_path(
            video_path,
            lambda path, progress: service.predict_video_path_with_angles(
                path,
                conf,
                progress,
                tracking,
                tracker,
                angle_definitions,
            ),
        )
    }


@router.get("/predict/video", response_model=dict)
async def predict_from_video_url(
    source: AnyUrl,
    angles: str | None = Query(None, description=ANGLES_DESCRIPTION),
    output: VideoOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    tracking: bool = Query(False),
    tracker: TrackerType = Query("bytetrack"),
    service: PoseEstimationService = Depends(get_service),
):
    angle_definitions = parse_angles(angles)
    require_json_for_angles(output, angle_definitions)
    tracking = tracking or bool(angle_definitions)
    if output == "video":
        result = await (
            service.predict_tracked_video_url(
                str(source), output, conf=conf, tracker=tracker
            )
            if tracking
            else service.predict_video_url(str(source), output, conf=conf)
        )
        return FileResponse(
            result,
            media_type="video/mp4",
            filename="pose_estimation.mp4",
            background=BackgroundTask(os.remove, result),
        )
    return {
        "job_id": video_job_manager.submit(
            lambda progress: service.predict_video_url_with_angles(
                str(source),
                conf,
                progress,
                tracking,
                tracker,
                angle_definitions,
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
