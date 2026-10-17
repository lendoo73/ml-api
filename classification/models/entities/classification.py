from pydantic import BaseModel


class Classification(BaseModel):
    class_id: int
    class_name: str
    confidence: float
