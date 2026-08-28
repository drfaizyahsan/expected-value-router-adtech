import sys

import requests


def test_endpoint(base_url: str = "http://localhost:8000"):
    payload = {
        "user": {
            "user_device": "mobile",
            "user_osName": "iOS",
            "user_browserName": "Safari",
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
                "subscriber_id": "partner_alpha",
                "subscriber_name": "Partner Alpha",
                "subscriber_tier": "gold",
                "commission_rate": 0.12,
                "booking_rate": 220.00,
                "mobile_optimized": True,
            },
            {
                "subscriber_id": "partner_beta",
                "subscriber_name": "Partner Beta",
                "subscriber_tier": "silver",
                "commission_rate": 0.10,
                "booking_rate": 180.00,
                "mobile_optimized": False,
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
