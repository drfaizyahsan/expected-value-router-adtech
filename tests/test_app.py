import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from src.train import train_and_evaluate

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = train_and_evaluate()
    monkeypatch.setattr(main_module, "model", model)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_route_traffic_success():
    payload = {
        "user": {
            "user_device": "desktop",
            "user_osName": "Windows",
            "user_browserName": "Chrome",
            "user_lat": 45.5017,
            "user_lng": -73.5673,
            "dest_lat": 40.7128,
            "dest_lng": -74.0060,
            "booked_flight": True,
            "booked_hotel": False,
            "booked_rental": False,
        },
        "candidates": [
            {
                "subscriber_id": "SUB_101",
                "subscriber_name": "Expedia",
                "subscriber_tier": "gold",
                "commission_rate": 0.12,
                "booking_rate": 250.00,
                "mobile_optimized": True,
            },
            {
                "subscriber_id": "SUB_102",
                "subscriber_name": "Booking.com",
                "subscriber_tier": "silver",
                "commission_rate": 0.10,
                "booking_rate": 200.00,
                "mobile_optimized": True,
            },
        ],
    }

    response = client.post("/route", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "selected_subscriber_id" in data
    assert "selected_subscriber_name" in data
    assert "max_expected_value" in data
    assert "propensity_score" in data
    assert "ranked_candidates" in data
    assert len(data["ranked_candidates"]) == 2
