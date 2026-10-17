from classification.controllers.classification_router_factory import (
    create_classification_router,
)
from classification.services.weather_classification_service import (
    WeatherClassificationService,
)


router = create_classification_router(
    WeatherClassificationService,
    prefix="/classification/weather",
    tags=["weather classification"],
)
