from object_detection.controllers.object_detection_router_factory import (
    create_object_detection_router,
)
from object_detection.services.road_sign_detection_service import RoadSignDetectionService


router = create_object_detection_router(
    RoadSignDetectionService,
    prefix="/object_detection/road_signs",
    tags=["road sign detection"],
)
