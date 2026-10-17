# Media search API

## Front-end update

Media search is now split by media type:

```text
GET /api/media/images/search?query=cat&per_page=10&random=true
GET /api/media/videos/search?query=cat&per_page=10&random=true
```

Replace calls to `/api/media/search` and remove the `media_type` query parameter. The response shape is unchanged; `media_type` is still returned in the response.

Video-specific query options can be added to the video endpoint later without exposing them to image searches.
