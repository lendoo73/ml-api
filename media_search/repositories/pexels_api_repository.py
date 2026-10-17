import os
import random
from urllib.parse import urlencode

from repositories.api_base_repository import ApiBaseRepository

from media_search.models.entities.media_source import MediaSource


class PexelsApiRepository(ApiBaseRepository):
    def __init__(self):
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            raise RuntimeError("PEXELS_API_KEY is not configured")

        super().__init__(
            base_url=os.getenv("PEXELS_API_BASE_URL", "https://api.pexels.com"),
            headers={"Authorization": api_key},
        )

    def search_images(self, query: str, per_page: int, randomize: bool) -> list[MediaSource]:
        response = self.get(
            "/v1/search",
            self._build_query("/v1/search", query, per_page, randomize),
        )
        return [
            MediaSource(
                id=str(item["id"]),
                source_url=item["src"]["original"],
                preview_url=item["src"]["medium"],
                width=item.get("width"),
                height=item.get("height"),
                alt=item.get("alt"),
                provider="pexels",
            )
            for item in response.get("photos", [])
        ]

    def search_videos(self, query: str, per_page: int, randomize: bool) -> list[MediaSource]:
        response = self.get(
            "/videos/search",
            self._build_query("/videos/search", query, per_page, randomize),
        )
        return [
            MediaSource(
                id=str(item["id"]),
                source_url=item["video_files"][0]["link"],
                preview_url=item.get("image", item["video_files"][0]["link"]),
                width=item.get("width"),
                height=item.get("height"),
                alt=None,
                provider="pexels",
            )
            for item in response.get("videos", [])
            if item.get("video_files")
        ]

    def _build_query(
        self,
        endpoint: str,
        query: str,
        per_page: int,
        randomize: bool,
    ) -> str:
        page = 1
        if randomize:
            first_page = self.get(
                endpoint,
                urlencode({"query": query, "per_page": 1, "page": 1}),
            )
            total_results = first_page.get("total_results", 0)
            page_count = min(
                80,
                max(1, (total_results + per_page - 1) // per_page),
            )
            page = random.randint(1, page_count)
        return urlencode({"query": query, "per_page": per_page, "page": page})
