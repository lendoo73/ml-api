from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from tcia.services.tcia_brain_scan_service import TciaBrainScanService


@lru_cache
def get_service() -> TciaBrainScanService:
    return TciaBrainScanService()


router = APIRouter(prefix="/brain_tumor", tags=["TCIA brain scans"])
BrainTumorClass = Literal["glioma", "meningioma", "pituitary", "healthy"]


@router.get("/random/{scan_class}")
def get_random_brain_tumor_scan(
    scan_class: BrainTumorClass,
    service: TciaBrainScanService = Depends(get_service),
):
    try:
        scan = service.random_class_scan(scan_class)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "scan": scan,
        "image_url": "/api/brain_tumor/image/" + scan.series_instance_uid,
        "stack_url": "/api/brain_tumor/stack/" + scan.series_instance_uid,
    }


@router.get("/random-image/{scan_class}", response_class=Response)
async def get_random_brain_tumor_image(
    scan_class: BrainTumorClass,
    service: TciaBrainScanService = Depends(get_service),
):
    try:
        image = await run_in_threadpool(service.random_class_image, scan_class)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(content=image, media_type="image/jpeg")


@router.get("/image/{series_instance_uid}", response_class=Response)
async def get_brain_tumor_image(
    series_instance_uid: str,
    service: TciaBrainScanService = Depends(get_service),
):
    try:
        image = await run_in_threadpool(service.image, series_instance_uid)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(content=image, media_type="image/jpeg")


@router.get("/stack/{series_instance_uid}", response_class=Response)
async def get_brain_tumor_stack(
    series_instance_uid: str,
    service: TciaBrainScanService = Depends(get_service),
):
    try:
        stack = await run_in_threadpool(service.stack, series_instance_uid)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(
        content=stack,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="brain_tumor_{series_instance_uid}.zip"'
            )
        },
    )
