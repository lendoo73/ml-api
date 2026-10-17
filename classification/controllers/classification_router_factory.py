import os
from functools import lru_cache
from typing import Type

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket
from fastapi import WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import AnyUrl
from starlette.background import BackgroundTask

from classification.models.dtos.classification_dto import ClassificationDto
from services.base_vision_service import BaseVisionService
from services.video_job_manager import video_job_manager
from utils.literals import ImageOutput, VideoOutput
from utils.video import save_uploadfile_to_temp


def plot_options(
    font_size: int | None = None,
    font: str = "Arial.ttf",
    probs: bool = True,
    show: bool = False,
    txt_color: tuple[int, int, int] | None = None,
) -> dict:
    return {
        "font_size": font_size,
        "font": font,
        "probs": probs,
        "show": show,
        **({"txt_color": txt_color} if txt_color is not None else {}),
    }


def create_classification_router(
    service_class: Type[BaseVisionService],
    prefix: str,
    tags: list[str],
) -> APIRouter:
    @lru_cache
    def get_service() -> BaseVisionService:
        return service_class()

    router = APIRouter(prefix=prefix, tags=tags)

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
        service: BaseVisionService = Depends(get_service),
    ):
        result = await service.predict_image(
            source, output, plot_options(font_size, font, probs, show, txt_color), conf
        )
        if output == "image":
            return Response(content=result, media_type="image/jpeg")
        return result

    @router.get("/predict", response_model=list[ClassificationDto])
    async def predict_from_url(
        source: AnyUrl,
        output: ImageOutput = Query("json", description="Return JSON predictions or a rendered JPEG image."),
        conf: float = Query(0.5, ge=0, le=1, description="Minimum prediction confidence."),
        font_size: int | None = Query(None),
        font: str = Query("Arial.ttf"),
        probs: bool = Query(True),
        show: bool = Query(False),
        txt_color: tuple[int, int, int] | None = Query(None),
        service: BaseVisionService = Depends(get_service),
    ):
        result = await service.predict_url(
            str(source), output, plot_options(font_size, font, probs, show, txt_color), conf
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
        service: BaseVisionService = Depends(get_service),
    ):
        try:
            result = await service.predict_search(
                query, output, plot_options(font_size, font, probs, show, txt_color), conf
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
        try:
            while True:
                frame = await websocket.receive_bytes()
                classifications = await run_in_threadpool(service.predict_frame_bytes, frame)
                await websocket.send_json([
                    classification.model_dump() for classification in classifications
                ])
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
        service: BaseVisionService = Depends(get_service),
    ):
        if output == "video":
            video_path = await service.predict_annotated_video(
                source, plot_options(font_size, font, probs, show, txt_color), conf
            )
            return FileResponse(
                video_path,
                media_type="video/mp4",
                filename="classification.mp4",
                background=BackgroundTask(os.remove, video_path),
            )
        video_path = await save_uploadfile_to_temp(source)
        return {"job_id": video_job_manager.submit_path(
            video_path, lambda path, progress: service.predict_video_path(path, conf, progress)
        )}

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
        service: BaseVisionService = Depends(get_service),
    ):
        if output == "video":
            result = await service.predict_video_url(
                str(source), output, plot_options(font_size, font, probs, show, txt_color), conf
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
                    str(source), output, plot_options(font_size, font, probs, show, txt_color), conf, progress
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
