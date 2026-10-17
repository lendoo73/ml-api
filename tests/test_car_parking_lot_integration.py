from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from controllers.controller import api_router
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_car_parking_lot_routes_are_registered():
    paths = {route.path for route in api_router.routes}
    assert "/car_parking_lot/predict" in paths
    assert "/car_parking_lot/predict/video" in paths
    assert "/car_parking_lot/predict/video/progress/{job_id}" in paths


def test_car_parking_lot_weight_is_present():
    weight = (
        Path(__file__).resolve().parents[1]
        / "oriented_bounding_box"
        / "weights"
        / "yolo11l_car_parking_lot.pt"
    )
    assert weight.is_file()


def test_car_parking_lot_openapi_exposes_all_media_routes(client):
    paths = client.app.openapi()["paths"]
    assert "/api/car_parking_lot/predict" in paths
    assert "/api/car_parking_lot/predict/video" in paths
    assert "/api/car_parking_lot/predict/video/progress/{job_id}" in paths
    assert any(
        route.path == "/api/car_parking_lot/predict/webcam"
        for route in client.app.routes
    )


def test_car_parking_lot_rejects_invalid_confidence(client):
    response = client.get(
        "/api/car_parking_lot/predict",
        params={"source": "https://example.com/image.jpg", "conf": 1.1},
    )
    assert response.status_code == 422


def test_car_parking_lot_missing_video_job_returns_not_found(client):
    response = client.get(
        "/api/car_parking_lot/predict/video/progress/does-not-exist"
    )
    assert response.status_code == 404
