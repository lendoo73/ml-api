from object_detection.controllers.object_detection_router_factory import (
    create_object_detection_router,
)
from object_detection.services.object_detection_service import ObjectDetectionService


router = create_object_detection_router(
    ObjectDetectionService,
    prefix="/object_detection",
    tags=["object detection"],
)
