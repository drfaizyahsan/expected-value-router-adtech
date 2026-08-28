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
        "user_device": "desktop",
        "user_osName": "Windows",
        "user_browserName_clean": "chrome",
        "subscriber_tier": "platinum",
        "travel_distance_km": 800.0,
        "is_long_haul": 0,
        "adr_clean": 200.0,
        "candidates": [
            {
                "partner_id": "partner_a",
                "cross_sell_score": 1.0,
                "mobile_ux_friction": 0,
                "expected_gross_commission": 50.0,
            },
            {
                "partner_id": "partner_b",
                "cross_sell_score": 0.0,
                "mobile_ux_friction": 0,
                "expected_gross_commission": 10.0,
            },
        ],
    }

    response = client.post("/route", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "selected_partner_id" in data
    assert "max_expected_value" in data
    assert len(data["routing_scores"]) == 2
