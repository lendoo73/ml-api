from fastapi.testclient import TestClient

from main import app
from tcia.controllers.tcia_brain_scan_rest_controller import get_service
from tcia.models.entities.tcia_scan import TciaScan


class FakeTciaBrainScanService:
    def random_class_scan(self, scan_class):
        assert scan_class == "glioma"
        return TciaScan(
            patient_id="patient-1",
            study_instance_uid="study-1",
            series_instance_uid="series-1",
            collection="upenn_gbm",
            modality="MR",
            viewer_url="https://viewer.example/series-1",
        )

    def random_class_image(self, scan_class):
        assert scan_class == "glioma"
        return b"jpeg-data"

    def image(self, series_instance_uid):
        assert series_instance_uid == "series-1"
        return b"jpeg-data"


def test_random_glioma_scan_returns_metadata_and_image_url():
    app.dependency_overrides[get_service] = FakeTciaBrainScanService
    try:
        with TestClient(app) as client:
            response = client.get("/api/brain_tumor/random/glioma")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan"]["collection"] == "upenn_gbm"
    assert payload["image_url"] == "/api/brain_tumor/image/series-1"
    assert payload["stack_url"] == "/api/brain_tumor/stack/series-1"
    assert payload["scan"]["viewer_url"] == "https://viewer.example/series-1"


def test_random_glioma_image_returns_jpeg():
    app.dependency_overrides[get_service] = FakeTciaBrainScanService
    try:
        with TestClient(app) as client:
            response = client.get("/api/brain_tumor/random-image/glioma")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"jpeg-data"


def test_glioma_series_image_returns_jpeg():
    app.dependency_overrides[get_service] = FakeTciaBrainScanService
    try:
        with TestClient(app) as client:
            response = client.get("/api/brain_tumor/image/series-1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"jpeg-data"
