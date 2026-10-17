import os
from pathlib import Path

from oriented_bounding_box.repositories.yolo_oriented_bounding_box_model_repository import (
    YoloOrientedBoundingBoxModelRepository,
)


class CarParkingLotModelRepository(YoloOrientedBoundingBoxModelRepository):
    """Repository for the car-parking-lot OBB model."""

    def __init__(self, model_path: str | None = None):
        default = Path(__file__).resolve().parent.parent / "weights" / "yolo11l_car_parking_lot.pt"
        super().__init__(
            model_path
            or os.getenv("YOLO_CAR_PARKING_LOT_MODEL", str(default))
        )
