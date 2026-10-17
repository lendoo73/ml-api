from object_detection.controllers.object_detection_router_factory import (
    create_object_detection_router,
)
from object_detection.services.face_mask_detection_service import FaceMaskDetectionService


router = create_object_detection_router(
    FaceMaskDetectionService,
    prefix="/object_detection/face_mask",
    tags=["face mask detection"],
)
