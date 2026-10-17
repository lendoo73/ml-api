from pydantic import BaseModel


class TciaScan(BaseModel):
    patient_id: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str
    collection: str
    modality: str | None = None
    body_part_examined: str | None = None
    image_count: int | None = None
    viewer_url: str
    provider: str = "tcia"
