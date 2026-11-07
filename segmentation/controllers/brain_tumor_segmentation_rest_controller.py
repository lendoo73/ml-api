from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import AnyUrl

from segmentation.controllers.segmentation_rest_controller import plot_options
from segmentation.models.dtos.brain_tumor_segmentation_dto import BrainTumorSegmentationDto
from segmentation.services.brain_tumor_segmentation_service import BrainTumorSegmentationService
from services.video_job_manager import video_job_manager
from utils.literals import ColorMode, ImageOutput


@lru_cache
def get_service() -> BrainTumorSegmentationService:
    return BrainTumorSegmentationService()


router = APIRouter(prefix="/segmentation/brain_tumor", tags=["brain tumor segmentation"])


@router.post("/predict", response_model=list[BrainTumorSegmentationDto])
async def predict_from_file(
    source: UploadFile = File(...),
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    show_conf: bool = Query(True),
    line_width: int | None = Query(None),
    font_size: int | None = Query(None),
    font: str = Query("Arial.ttf"),
    labels: bool = Query(True),
    masks: bool = Query(True),
    boxes: bool = Query(True),
    color_mode: ColorMode = Query("class"),
    service: BrainTumorSegmentationService = Depends(get_service),
):
    result = await service.predict_image(
        source,
        output,
        plot_options(show_conf, line_width, font_size, font, labels, masks, boxes, color_mode),
        conf,
    )
    return Response(content=result, media_type="image/jpeg") if output == "image" else result


@router.post("/predict/batch", response_model=dict)
async def predict_from_zip(
    source: UploadFile = File(..., description="ZIP archive containing images to segment"),
    service: BrainTumorSegmentationService = Depends(get_service),
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


@router.get("/predict", response_model=list[BrainTumorSegmentationDto])
async def predict_from_url(
    source: AnyUrl,
    output: ImageOutput = Query("json"),
    conf: float = Query(0.5, ge=0, le=1),
    service: BrainTumorSegmentationService = Depends(get_service),
):
    result = await service.predict_url(str(source), output, conf=conf)
    return Response(content=result, media_type="image/jpeg") if output == "image" else result
