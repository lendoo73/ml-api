from typing import Protocol

from media_search.models.entities.media_source import MediaSource


class MediaProviderRepository(Protocol):
    def search_images(self, query: str, per_page: int, randomize: bool) -> list[MediaSource]:
        ...

    def search_videos(self, query: str, per_page: int, randomize: bool) -> list[MediaSource]:
        ...
