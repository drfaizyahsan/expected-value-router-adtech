import sys

import requests


def test_endpoint(base_url: str = "http://localhost:8000"):
    payload = {
        "user_device": "mobile",
        "user_osName": "iOS",
        "user_browserName_clean": "safari",
        "subscriber_tier": "gold",
        "travel_distance_km": 1200.0,
        "is_long_haul": 1,
        "adr_clean": 220.0,
        "candidates": [
            {
                "partner_id": "partner_alpha",
                "cross_sell_score": 1.0,
                "mobile_ux_friction": 0,
                "expected_gross_commission": 35.0,
            },
            {
                "partner_id": "partner_beta",
                "cross_sell_score": 0.0,
                "mobile_ux_friction": 1,
                "expected_gross_commission": 15.0,
            },
        ],
    }

    try:
        response = requests.post(f"{base_url}/route", json=payload, timeout=5.0)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.json())
        assert response.status_code == 200
        print("\nSmoke test succeeded!")
    except (requests.RequestException, AssertionError) as e:
        print(f"\nSmoke test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_endpoint(url)
