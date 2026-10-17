from pydantic import BaseModel


class BoundingBoxDto(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
