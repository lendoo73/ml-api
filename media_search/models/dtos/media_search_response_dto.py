from typing import List

from pydantic import BaseModel

from media_search.models.dtos.media_source_dto import MediaSourceDto
from utils.literals import MediaType


class MediaSearchResponseDto(BaseModel):
    query: str
    media_type: MediaType
    items: List[MediaSourceDto]
