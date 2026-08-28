from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app, calculate_haversine_distance


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_decision_logger(monkeypatch):
    """Mocks decision logging to avoid writing test logs to disk during test runs."""
    mock_log = MagicMock()
    monkeypatch.setattr(main_module, "log_routing_decision", mock_log)
    return mock_log


def test_haversine_distance():
    """Validates Montreal to NYC distance calculation (~530km)."""
    # Montreal (45.5017, -73.5673) to NYC (40.7128, -74.0060)
    dist = calculate_haversine_distance(45.5017, -73.5673, 40.7128, -74.0060)
    assert 500.0 < dist < 600.0


def test_health_check(client):
    """Verifies health check endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_route_traffic_service_unavailable(client, monkeypatch):
    """Ensures 503 is returned when model artifact is missing."""
    monkeypatch.setattr(main_module, "model_artifact", None)

    payload = {
        "user": {
            "user_device": "mobile",
            "user_osName": "iOS",
            "user_browserName": "Safari",
            "user_lat": 45.5017,
            "user_lng": -73.5673,
            "dest_lat": 40.7128,
            "dest_lng": -74.0060,
        },
        "candidates": [
            {
                "subscriber_id": "SUB_1",
                "subscriber_name": "Partner A",
                "commission_rate": 0.10,
                "booking_rate": 200.0,
            }
        ],
    }

    response = client.post("/route", json=payload)
    assert response.status_code == 503


def test_route_traffic_successful_routing(client, monkeypatch, mock_decision_logger):
    """Tests complete routing flow with mock LightGBM predictions and propensity scores."""
    # Mock LightGBM model returning fixed probabilities
    mock_model = MagicMock()
    # Return 0.10 for candidate 1 ($2.00 EV), 0.30 for candidate 2 ($12.00 EV)
    mock_model.predict_proba.return_value = np.array(
        [
            [0.90, 0.10],
            [0.70, 0.30],
        ]
    )
    monkeypatch.setattr(main_module, "model_artifact", mock_model)

    payload = {
        "user": {
            "user_device": "desktop",
            "user_osName": "macOS",
            "user_browserName": "Chrome",
            "user_lat": 45.5017,
            "user_lng": -73.5673,
            "dest_lat": 40.7128,
            "dest_lng": -74.0060,
            "booked_flight": True,
        },
        "candidates": [
            {
                "subscriber_id": "SUB_LOW",
                "subscriber_name": "Low EV Partner",
                "subscriber_tier": "silver",
                "commission_rate": 0.10,
                "booking_rate": 200.0,  # EV = 0.10 * 0.10 * 200 = $2.00
            },
            {
                "subscriber_id": "SUB_HIGH",
                "subscriber_name": "High EV Partner",
                "subscriber_tier": "gold",
                "commission_rate": 0.20,
                "booking_rate": 200.0,  # EV = 0.30 * 0.20 * 200 = $12.00
            },
        ],
    }

    response = client.post("/route", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_fallback"] is False
    assert data["selected_subscriber_id"] in [
        "SUB_HIGH",
        "SUB_LOW",
    ]  # Accounts for epsilon exploration
    assert "propensity_score" in data
    assert 0.0 < data["propensity_score"] <= 1.0
    assert data["policy_version"] == "epsilon_greedy_v1"
    assert len(data["ranked_candidates"]) == 2

    # Verify decision logging function was called once per request
    assert mock_decision_logger.call_count == 1


def test_route_traffic_pydantic_validation_error(client):
    """Ensures Pydantic rejects invalid lat/lng bounds or missing fields."""
    invalid_payload = {
        "user": {
            "user_device": "mobile",
            "user_lat": 999.0,  # Invalid latitude (>90)
            "user_lng": -73.5673,
        },
        "candidates": [],  # Min length 1 required
    }

    response = client.post("/route", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
