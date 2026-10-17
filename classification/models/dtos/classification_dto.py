from pydantic import BaseModel


class ClassificationDto(BaseModel):
    class_id: int
    class_name: str
    confidence: float
