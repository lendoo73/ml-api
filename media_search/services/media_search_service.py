from media_search.models.dtos.media_search_response_dto import MediaSearchResponseDto
from media_search.repositories.pexels_api_repository import PexelsApiRepository


class MediaSearchService:
    def __init__(self):
        self.provider = PexelsApiRepository()

    def search_images(self, query: str, per_page: int, randomize: bool) -> MediaSearchResponseDto:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty")

        items = self.provider.search_images(normalized_query, per_page, randomize)
        return MediaSearchResponseDto(
            query=normalized_query,
            media_type="image",
            items=[item.model_dump() for item in items],
        )

    def search_videos(self, query: str, per_page: int, randomize: bool) -> MediaSearchResponseDto:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be empty")

        items = self.provider.search_videos(normalized_query, per_page, randomize)
        return MediaSearchResponseDto(
            query=normalized_query,
            media_type="video",
            items=[item.model_dump() for item in items],
        )
