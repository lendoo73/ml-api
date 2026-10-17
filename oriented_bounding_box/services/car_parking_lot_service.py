from oriented_bounding_box.repositories.car_parking_lot_model_repository import (
    CarParkingLotModelRepository,
)
from oriented_bounding_box.services.oriented_bounding_box_service import (
    OrientedBoundingBoxService,
)


class CarParkingLotService(OrientedBoundingBoxService):
    def __init__(self):
        super().__init__()
        self.model_repository = CarParkingLotModelRepository()
