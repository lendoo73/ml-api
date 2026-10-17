from oriented_bounding_box.controllers.oriented_bounding_box_router_factory import (
    create_oriented_bounding_box_router,
)
from oriented_bounding_box.services.car_parking_lot_service import CarParkingLotService


router = create_oriented_bounding_box_router(
    CarParkingLotService,
    "/car_parking_lot",
    ["car parking lot"],
)
