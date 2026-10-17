from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from media_search.models.dtos.media_search_response_dto import MediaSearchResponseDto
from media_search.services.media_search_service import MediaSearchService


@lru_cache
def get_service() -> MediaSearchService:
    return MediaSearchService()


router = APIRouter(
    prefix="/media",
    tags=["media search"],
)


@router.get("/images/search", response_model=MediaSearchResponseDto)
def search_images(
    query: str = Query(..., min_length=1),
    per_page: int = Query(1, ge=1, le=10),
    randomize: bool = Query(True, alias="random"),
    service: MediaSearchService = Depends(get_service),
):
    try:
        return service.search_images(query, per_page, randomize)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/videos/search", response_model=MediaSearchResponseDto)
def search_videos(
    query: str = Query(..., min_length=1),
    per_page: int = Query(1, ge=1, le=10),
    randomize: bool = Query(True, alias="random"),
    service: MediaSearchService = Depends(get_service),
):
    try:
        return service.search_videos(query, per_page, randomize)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
