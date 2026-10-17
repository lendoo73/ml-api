from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
from idc_index import IDCClient
import numpy as np
import pydicom

from tcia.models.entities.tcia_scan import TciaScan


class TciaApiRepository:
    @staticmethod
    @lru_cache(maxsize=1)
    def _client() -> IDCClient:
        return IDCClient.client()

    def search_series(
        self,
        collection: str,
        modality: str | None,
        limit: int,
        randomize: bool,
    ) -> list[TciaScan]:
        filters = [
            f"collection_id = '{collection.lower()}'",
            "(BodyPartExamined IS NULL OR upper(BodyPartExamined) NOT LIKE '%PHANTOM%')",
            "(SeriesDescription IS NULL OR upper(SeriesDescription) NOT LIKE '%LOCALIZER%')",
            "(SeriesDescription IS NULL OR upper(SeriesDescription) NOT LIKE '%SCOUT%')",
        ]
        if modality:
            filters.append(f"Modality = '{modality.upper()}'")
        order = "ORDER BY random()" if randomize else "ORDER BY SeriesInstanceUID"
        query = f"""
            SELECT collection_id, PatientID, StudyInstanceUID,
                   SeriesInstanceUID, Modality, BodyPartExamined,
                   SeriesDescription
            FROM index
            WHERE {' AND '.join(filters)}
            {order}
            LIMIT {limit}
        """
        rows = self._client().sql_query(query)
        return [self._to_scan(row) for row in rows.to_dict("records")]

    def _to_scan(self, item: dict) -> TciaScan:
        series_uid = item["SeriesInstanceUID"]
        return TciaScan(
            patient_id=item.get("PatientID"),
            study_instance_uid=item.get("StudyInstanceUID"),
            series_instance_uid=series_uid,
            collection=item["collection_id"],
            modality=item.get("Modality"),
            body_part_examined=item.get("BodyPartExamined"),
            image_count=None,
            viewer_url=self._client().get_viewer_URL(seriesInstanceUID=series_uid),
        )

    def get_glioma(self) -> TciaScan:
        scans = self.search_series("upenn_gbm", "MR", 1, True)
        if not scans:
            raise ValueError("No glioma MR series was found in IDC.")
        return scans[0]

    def get_meningioma(self) -> TciaScan:
        raise ValueError(
            "Meningioma-SEG-CLASS is a TCIA limited-access collection and "
            "requires an authorized TCIA download configuration."
        )

    def series_image(self, series_instance_uid: str) -> bytes:
        datasets = self._download_datasets(series_instance_uid)
        return self._encode_dataset(datasets[len(datasets) // 2])

    def series_stack_zip(self, series_instance_uid: str) -> bytes:
        datasets = self._download_datasets(series_instance_uid)
        archive = BytesIO()
        with ZipFile(archive, "w", ZIP_DEFLATED) as output:
            for index, dataset in enumerate(datasets, start=1):
                output.writestr(f"slice_{index:04d}.jpg", self._encode_dataset(dataset))
        return archive.getvalue()

    def _download_datasets(self, series_instance_uid: str):
        with TemporaryDirectory(prefix="glioma_") as directory:
            self._client().download_dicom_series(
                seriesInstanceUID=series_instance_uid,
                downloadDir=directory,
                quiet=True,
                show_progress_bar=False,
            )
            datasets = []
            for path in Path(directory).rglob("*"):
                if not path.is_file():
                    continue
                try:
                    datasets.append(pydicom.dcmread(path))
                except (pydicom.errors.InvalidDicomError, OSError):
                    continue
            if not datasets:
                raise ValueError("IDC returned no readable DICOM images.")
            datasets.sort(key=lambda item: getattr(item, "InstanceNumber", 0))
            return datasets

    @staticmethod
    def _encode_dataset(dataset) -> bytes:
        pixels = dataset.pixel_array.astype(np.float32)
        low, high = np.percentile(pixels, (1, 99))
        pixels = np.clip((pixels - low) / max(high - low, 1) * 255, 0, 255)
        success, encoded = cv2.imencode(".jpg", pixels.astype(np.uint8))
        if not success:
            raise ValueError("Could not encode the IDC DICOM slice as JPEG.")
        return encoded.tobytes()
