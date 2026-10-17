from tcia.models.dtos.tcia_scan_dto import TciaScanDto
from tcia.repositories.tcia_api_repository import TciaApiRepository


COLLECTIONS_BY_TERM = {
    "glioma": ("tcga_gbm", "cptac_gbm", "upenn_gbm", "gbm_dsc_mri_dro"),
    "meningioma": (),
    "pituitary": (),
    "healthy": (),
}


class TciaBrainScanService:
    def __init__(self):
        self.repository = TciaApiRepository()

    def search(
        self,
        diagnosis: str,
        collection: str | None,
        modality: str | None,
        limit: int,
        randomize: bool,
    ) -> list[TciaScanDto]:
        normalized = diagnosis.strip().lower()
        if normalized not in COLLECTIONS_BY_TERM:
            raise ValueError(
                "Unsupported diagnosis. Use one of: glioma, meningioma, pituitary, healthy."
            )
        collections = (collection,) if collection else COLLECTIONS_BY_TERM[normalized]
        if not collections:
            raise ValueError(
                "No verified IDC collection is mapped for this diagnosis. "
                "Provide a collection identifier explicitly."
            )

        results = []
        per_collection = max(1, (limit + len(collections) - 1) // len(collections))
        for collection_name in collections:
            results.extend(
                self.repository.search_series(
                    collection_name, modality, per_collection, randomize
                )
            )
        if randomize:
            import random

            random.shuffle(results)
        return [
            TciaScanDto.model_validate(item.model_dump())
            for item in results[:limit]
        ]

    def random_class_image(self, scan_class: str) -> bytes:
        scan = self.random_class_scan(scan_class)
        return self.repository.series_image(scan.series_instance_uid)

    def random_class_scan(self, scan_class: str):
        if scan_class == "glioma":
            return self.repository.get_glioma()
        if scan_class == "meningioma":
            return self.repository.get_meningioma()
        collections = COLLECTIONS_BY_TERM[scan_class]
        if not collections:
            raise ValueError(
                f"No verified IDC collection is configured for '{scan_class}'."
            )
        scans = []
        for collection in collections:
            scans.extend(self.repository.search_series(collection, "MR", 1, True))
        if not scans:
            raise ValueError(f"No MR scan was found for '{scan_class}'.")
        import random

        return random.choice(scans)

    def image(self, series_instance_uid: str) -> bytes:
        return self.repository.series_image(series_instance_uid)

    def stack(self, series_instance_uid: str) -> bytes:
        return self.repository.series_stack_zip(series_instance_uid)
