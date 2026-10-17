from classification.controllers.classification_router_factory import (
    create_classification_router,
)
from classification.services.classification_service import ClassificationService


router = create_classification_router(
    ClassificationService,
    prefix="/classification",
    tags=["classification"],
)
