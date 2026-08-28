import logging
import os
import random

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

np.random.seed(42)
random.seed(42)


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculates approximate distance in kilometers between two lat/lng pairs."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def parse_coords_safely(coord_str: str) -> tuple[float, float]:
    """Safely extracts float (lat, lng) from raw string, returning (NaN, NaN) on failure."""
    try:
        parts = str(coord_str).split(",")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, TypeError):
        pass
    return np.nan, np.nan


def inject_dirty_data(df: pd.DataFrame) -> pd.DataFrame:
    """Injects missing values, numerical outliers, and ill-formed strings into strictly specified columns."""
    df = df.copy()
    n = len(df)

    # 1. Missing Data (NaNs)
    mask_browser = np.random.rand(n) < 0.04
    df.loc[mask_browser, "user_browserName"] = np.nan

    mask_rental = np.random.rand(n) < 0.03
    df.loc[mask_rental, "booked_rental"] = np.nan

    # 2. Ill-Formed Data
    dirty_browsers = [" chrome ", "CHROME_v118", "UNKNOWN_BROWSER", "safari_mobile"]
    mask_dirty_b = np.random.rand(n) < 0.02
    df.loc[mask_dirty_b, "user_browserName"] = np.random.choice(
        dirty_browsers, size=mask_dirty_b.sum()
    )

    malformed_coords = ["0.0,0.0", "INVALID_COORD", "45.5017;-73.5673", "NULL"]
    mask_coords = np.random.rand(n) < 0.02
    df.loc[mask_coords, "user_lat_lng"] = np.random.choice(
        malformed_coords, size=mask_coords.sum()
    )

    # 3. Numerical Outliers
    mask_outlier_dest = np.random.rand(n) < 0.01
    df.loc[mask_outlier_dest, "dest_lat_lng"] = "999.9999,-800.0000"

    return df


def generate_user_data(n_users: int = 5000) -> pd.DataFrame:
    """Generates synthetic user search sessions using strictly requested columns."""

    device_os_map = {
        "mobile": ["iOS", "Android"],
        "desktop": ["Windows", "macOS", "Linux"],
        "tablet": ["iOS", "Android"],
    }

    devices = np.random.choice(
        ["mobile", "desktop", "tablet"], size=n_users, p=[0.55, 0.35, 0.10]
    )
    os_names = [np.random.choice(device_os_map[d]) for d in devices]

    browser_map = {
        "iOS": ["Safari", "Chrome"],
        "Android": ["Chrome", "Samsung Internet"],
        "Windows": ["Chrome", "Edge", "Firefox"],
        "macOS": ["Safari", "Chrome", "Firefox"],
        "Linux": ["Firefox", "Chrome"],
    }
    browsers = [np.random.choice(browser_map[o]) for o in os_names]

    user_lat = np.round(np.random.uniform(25.0, 50.0, size=n_users), 4)
    user_lng = np.round(np.random.uniform(-120.0, -70.0, size=n_users), 4)
    user_lat_lng = [f"{lat},{lng}" for lat, lng in zip(user_lat, user_lng)]

    dest_lat = np.round(user_lat + np.random.uniform(-15.0, 15.0, size=n_users), 4)
    dest_lng = np.round(user_lng + np.random.uniform(-20.0, 20.0, size=n_users), 4)
    dest_lat_lng = [f"{lat},{lng}" for lat, lng in zip(dest_lat, dest_lng)]

    booked_flight = np.random.binomial(n=1, p=0.25, size=n_users).astype(float)
    booked_rental = np.random.binomial(n=1, p=0.12, size=n_users).astype(float)

    # Exact 8 columns specified
    df_users = pd.DataFrame(
        {
            "user_device": devices,
            "user_osName": os_names,
            "user_browserName": browsers,
            "user_lat_lng": user_lat_lng,
            "dest_lat_lng": dest_lat_lng,
            "booked_flight": booked_flight,
            "booked_hotel": 0,  # Placeholder, assigned by ground truth function below
            "booked_rental": booked_rental,
        }
    )

    return inject_dirty_data(df_users)


def generate_subscriber_data() -> pd.DataFrame:
    """Generates contractual terms for 5 hotel advertisers."""

    subscribers = [
        {
            "subscriber_id": "SUB_1",
            "subscriber_name": "BudgetInns Global",
            "subscriber_tier": 1,
            "avg_daily_rate": 85.0,
            "commission_rate": 0.10,
            "mobile_optimized": 1,
        },
        {
            "subscriber_id": "SUB_2",
            "subscriber_name": "Urban Stay Suites",
            "subscriber_tier": 2,
            "avg_daily_rate": 180.0,
            "commission_rate": 0.12,
            "mobile_optimized": 1,
        },
        {
            "subscriber_id": "SUB_3",
            "subscriber_name": "Grand Horizon Resorts",
            "subscriber_tier": 3,
            "avg_daily_rate": 450.0,
            "commission_rate": 0.15,
            "mobile_optimized": 0,
        },
        {
            "subscriber_id": "SUB_4",
            "subscriber_name": "EcoLodge Boutique",
            "subscriber_tier": 2,
            "avg_daily_rate": 220.0,
            "commission_rate": 0.14,
            "mobile_optimized": 1,
        },
        {
            "subscriber_id": "SUB_5",
            "subscriber_name": "Express Stop Motel",
            "subscriber_tier": 1,
            "avg_daily_rate": -99.0,
            "commission_rate": 0.08,
            "mobile_optimized": 0,
        },
    ]
    return pd.DataFrame(subscribers)


def apply_business_logic_and_ground_truth(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Computes log-odds and generates target 'booked_hotel' using strictly available features."""
    df = df_pairs.copy()

    # Base negative log-odds offset to hit ~3% overall conversion rate
    log_odds = -4.1

    # Signal 1: Cross-sell affinity
    flight_sig = df["booked_flight"].fillna(0)
    rental_sig = df["booked_rental"].fillna(0)
    log_odds += flight_sig * 0.90
    log_odds += rental_sig * 0.45

    # Signal 2: Travel Distance (Longer distance = higher hotel intent)
    distances = []
    for u_coord, d_coord in zip(df["user_lat_lng"], df["dest_lat_lng"]):
        u_lat, u_lng = parse_coords_safely(u_coord)
        d_lat, d_lng = parse_coords_safely(d_coord)
        if np.isnan(u_lat) or np.isnan(d_lat):
            distances.append(0.0)
        else:
            distances.append(calculate_haversine_distance(u_lat, u_lng, d_lat, d_lng))

    dist_km = np.array(distances)
    log_odds += (dist_km > 1000.0).astype(int) * 0.60

    # Signal 3: Mobile UX friction
    mobile_friction = (
        (df["user_device"] == "mobile") & (df["mobile_optimized"] == 0)
    ).astype(int)
    log_odds -= mobile_friction * 0.80

    # Signal 4: Desktop / Premium OS affinity for high-tier luxury resort (SUB_3)
    desktop_mac = (
        (df["user_osName"] == "macOS") & (df["subscriber_id"] == "SUB_3")
    ).astype(int)
    log_odds += desktop_mac * 0.70

    # Compute probabilities & sample target booked_hotel (0 or 1)
    p_conversion = 1.0 / (1.0 + np.exp(-log_odds))
    booked_hotel = np.random.binomial(n=1, p=p_conversion)

    df["p_conversion_ground_truth"] = p_conversion
    df["booked_hotel"] = booked_hotel

    return df


def main():
    logger = logging.getLogger(__name__)
    logger.info(" Generating synthetic user sessions & subscriber pairs...")

    df_users = generate_user_data(n_users=1000)
    df_subscribers = generate_subscriber_data()

    df_pairs = df_users.merge(df_subscribers, how="cross")
    df_dataset = apply_business_logic_and_ground_truth(df_pairs)

    os.makedirs("data/raw", exist_ok=True)
    df_dataset.to_csv("data/raw/session_subscriber_pairs.csv", index=False)

    logger.info("Raw dataset saved to `data/raw/session_subscriber_pairs.csv`")
    logger.info(f"Total Pair Records: {len(df_dataset):,}")
    logger.info(
        f"Target Imbalance (`booked_hotel`): {df_dataset['booked_hotel'].mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
