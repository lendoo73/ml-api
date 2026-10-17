class InvalidVideoSourceError(ValueError):
    """Raised when a video URL cannot be downloaded or opened as a video."""


class InvalidImageSourceError(ValueError):
    """Raised when an image URL does not provide a readable image."""
