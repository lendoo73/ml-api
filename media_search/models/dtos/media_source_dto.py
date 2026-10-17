from typing import Optional

from pydantic import BaseModel


class MediaSourceDto(BaseModel):
    id: str
    source_url: str
    preview_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    alt: Optional[str] = None
    provider: str
