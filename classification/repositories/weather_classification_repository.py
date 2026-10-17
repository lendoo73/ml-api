from pathlib import Path

from classification.repositories.yolo_classification_model_repository import (
    YoloClassificationModelRepository,
)


WEATHER_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "weights" / "yolo26x_weather_cls.pt"
)


class WeatherClassificationRepository(YoloClassificationModelRepository):
    def __init__(self):
        super().__init__(WEATHER_MODEL_PATH)
