from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @classmethod
    def from_xyxy(cls, coordinates):
        x_min, y_min, x_max, y_max = coordinates
        return cls(
            x_min=int(x_min),
            y_min=int(y_min),
            x_max=int(x_max),
            y_max=int(y_max),
        )
