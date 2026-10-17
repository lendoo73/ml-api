from classification.repositories.weather_classification_repository import (
    WeatherClassificationRepository,
)
from services.base_vision_service import BaseVisionService


class WeatherClassificationService(BaseVisionService):
    def __init__(self):
        super().__init__(WeatherClassificationRepository())
