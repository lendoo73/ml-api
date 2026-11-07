from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from controllers.controller import api_router
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_oriented_bounding_box_model_routes_are_registered():
    paths = {route.path for route in api_router.routes}
    assert "/oriented_bounding_box/predict" in paths
    assert "/oriented_bounding_box/predict/video" in paths
    assert "/oriented_bounding_box/predict/video/progress/{job_id}" in paths
    assert "/car_parking_lot/predict" not in paths


def test_car_parking_lot_weight_is_present():
    weight = (
        Path(__file__).resolve().parents[1]
        / "oriented_bounding_box"
        / "weights"
        / "yolo11l_car_parking_lot.pt"
    )
    assert weight.is_file()


def test_oriented_bounding_box_openapi_exposes_model_and_media_routes(client):
    paths = client.app.openapi()["paths"]
    assert "/api/oriented_bounding_box/predict" in paths
    assert "/api/oriented_bounding_box/predict/video" in paths
    assert "/api/oriented_bounding_box/predict/video/progress/{job_id}" in paths
    assert any(
        route.path == "/api/oriented_bounding_box/predict/webcam"
        for route in client.app.routes
    )
    parameters = {
        parameter["name"]: parameter
        for parameter in paths["/api/oriented_bounding_box/predict"]["post"]["parameters"]
    }
    assert parameters["model"]["schema"]["enum"] == ["yolo26x-obb", "car-parking-lot"]


@pytest.mark.parametrize(
    ("model", "weight_filename"),
    [
        ("yolo26x-obb", "yolo26x-obb.pt"),
        ("car-parking-lot", "yolo11l_car_parking_lot.pt"),
    ],
)
def test_oriented_bounding_box_model_selects_configured_weights(model, weight_filename):
    from oriented_bounding_box.controllers.oriented_bounding_box_rest_controller import (
        get_service,
    )

    service = get_service(model)

    assert Path(service.model_repository._model_path).name == weight_filename


def test_car_parking_lot_rejects_invalid_confidence(client):
    response = client.get(
        "/api/oriented_bounding_box/predict",
        params={
            "source": "https://example.com/image.jpg",
            "model": "car-parking-lot",
            "conf": 1.1,
        },
    )
    assert response.status_code == 422


def test_car_parking_lot_missing_video_job_returns_not_found(client):
    response = client.get(
        "/api/oriented_bounding_box/predict/video/progress/does-not-exist"
    )
    assert response.status_code == 404
