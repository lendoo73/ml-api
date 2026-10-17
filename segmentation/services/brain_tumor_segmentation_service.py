from io import BytesIO
from zipfile import BadZipFile, ZipFile

from fastapi.concurrency import run_in_threadpool

from segmentation.repositories.brain_tumor_segmentation_repository import (
    BrainTumorSegmentationRepository,
)
from services.base_vision_service import BaseVisionService


class BrainTumorSegmentationService(BaseVisionService):
    def __init__(self):
        super().__init__(BrainTumorSegmentationRepository())

    async def predict_zip(self, archive_bytes: bytes, progress_callback=None) -> list[dict]:
        try:
            zip_file = ZipFile(BytesIO(archive_bytes))
        except BadZipFile as error:
            raise ValueError("The uploaded file is not a valid ZIP archive.") from error

        predictions = []
        with zip_file:
            members = [member for member in zip_file.infolist() if not member.is_dir()]
            for index, member in enumerate(members, start=1):
                image_bytes = zip_file.read(member)
                result = await run_in_threadpool(self.predict_frame_bytes, image_bytes)
                predictions.append({
                    "filename": member.filename,
                    "predictions": [item.model_dump() for item in result],
                })
                if progress_callback:
                    progress_callback(index, len(members))
        return predictions
