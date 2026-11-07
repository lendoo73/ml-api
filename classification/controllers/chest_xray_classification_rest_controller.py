from classification.controllers.classification_router_factory import (
    create_classification_router,
)
from classification.services.chest_xray_classification_service import (
    ChestXrayClassificationService,
)


router = create_classification_router(
    ChestXrayClassificationService,
    prefix="/classification/chest_xray",
    tags=["chest xray classification"],
    include_video=False,
    include_webcam=False,
)
