# test_route.py
import requests

from src.utils import get_logger

logger = get_logger()


url = "http://127.0.0.1:8000/route"

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
            "subscriber_id": "SUB_EXPEDIA",
            "subscriber_name": "Expedia",
            "subscriber_tier": "platinum",
            "commission_rate": 0.12,
            "booking_rate": 350.00,
            "mobile_optimized": True,
        },
        {
            "subscriber_id": "SUB_BOOKING",
            "subscriber_name": "Booking.com",
            "subscriber_tier": "gold",
            "commission_rate": 0.15,
            "booking_rate": 280.00,
            "mobile_optimized": True,
        },
    ],
}

response = requests.post(url, json=payload)

logger.info(f"Status Code: {response.status_code}\n")
logger.info("Response JSON:")
logger.info(response.json())
