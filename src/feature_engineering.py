import math

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

# Explicitly import policy and schema contracts
from src.utils import get_logger

logger = get_logger()

# Feature Schema Contract
CAT_COLS = ["user_device", "user_osName", "user_browserName_clean", "subscriber_tier"]
NUMERIC_COLS = [
    "travel_distance_km",
    "is_long_haul",
    "adr_clean",
    "cross_sell_score",
    "mobile_ux_friction",
]
FEATURE_COLS = CAT_COLS + NUMERIC_COLS
TARGET_COL = "is_conversion"


def parse_coordinate(col_name: str, index: int):
    """Extracts latitude (index=1) or longitude (index=2) float from string coordinate."""
    pattern = r"(-?\d+\.\d+)[,\s;]+(-?\d+\.\d+)"
    val = F.regexp_extract(F.col(col_name), pattern, index).try_cast(DoubleType())
    if index == 1:
        return F.when((val >= -90.0) & (val <= 90.0), val).otherwise(None)
    return F.when((val >= -180.0) & (val <= 180.0), val).otherwise(None)


def compute_haversine_distance(lat1_col, lon1_col, lat2_col, lon2_col):
    """Computes great-circle distance between two points in kilometers using PySpark SQL math."""
    r = 6371.0
    phi1 = F.radians(lat1_col)
    phi2 = F.radians(lat2_col)
    delta_phi = F.radians(lat2_col - lat1_col)
    delta_lambda = F.radians(lon2_col - lon1_col)

    a = F.pow(F.sin(delta_phi / 2.0), 2) + F.cos(phi1) * F.cos(phi2) * F.pow(
        F.sin(delta_lambda / 2.0), 2
    )
    c = 2.0 * F.atan2(F.sqrt(a), F.sqrt(1.0 - a))
    return r * c


def engineer_features_spark(df):
    """Applies distributed feature engineering while guaranteeing all original input columns

    (including target and session metadata) are retained in the output DataFrame.
    """
    # Store original column names to verify no columns are lost
    original_cols = df.columns

    # 1. Clean Browser Name
    clean_browser_expr = (
        F.when(F.lower(F.trim(F.col("user_browserName"))).contains("chrome"), "chrome")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("safari"), "safari")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("firefox"), "firefox")
        .when(F.col("user_browserName").isNotNull(), "other")
        .otherwise("unknown")
    )

    df_working = df.withColumn("user_browserName_clean", clean_browser_expr)

    # Safely handle missing cross-sell columns if they don't exist in input schema
    booked_flight_col = (
        F.coalesce(F.col("booked_flight"), F.lit(0.0))
        if "booked_flight" in df_working.columns
        else F.lit(0.0)
    )
    booked_rental_col = (
        F.coalesce(F.col("booked_rental"), F.lit(0.0))
        if "booked_rental" in df_working.columns
        else F.lit(0.0)
    )

    # 2. Coordinates & Distance Computation
    df_coords = (
        df_working.withColumn("user_lat", parse_coordinate("user_lat_lng", 1))
        .withColumn("user_lng", parse_coordinate("user_lat_lng", 2))
        .withColumn("dest_lat", parse_coordinate("dest_lat_lng", 1))
        .withColumn("dest_lng", parse_coordinate("dest_lat_lng", 2))
    )

    df_dist = df_coords.withColumn(
        "raw_distance_km",
        compute_haversine_distance(
            F.col("user_lat"), F.col("user_lng"), F.col("dest_lat"), F.col("dest_lng")
        ),
    )

    # 3. Median Distance Calculation
    try:
        quantiles = df_dist.stat.approxQuantile("raw_distance_km", [0.5], 0.01)
        median_dist = quantiles[0] if quantiles and quantiles[0] is not None else 500.0
        if math.isnan(median_dist):
            median_dist = 500.0
    except TypeError, ValueError:
        median_dist = 500.0

    # 4. Engineer Final Features
    df_features = (
        df_dist.withColumn(
            "travel_distance_km",
            F.coalesce(F.col("raw_distance_km"), F.lit(float(median_dist))),
        )
        .withColumn(
            "is_long_haul", F.when(F.col("travel_distance_km") > 1000.0, 1).otherwise(0)
        )
        .withColumn(
            "adr_clean",
            F.when(F.col("avg_daily_rate") < 10.0, 10.0).otherwise(
                F.col("avg_daily_rate")
            ),
        )
        .withColumn("cross_sell_score", booked_flight_col + booked_rental_col)
        .withColumn(
            "mobile_ux_friction",
            F.when(
                (F.col("user_device") == "mobile") & (F.col("mobile_optimized") == 0), 1
            ).otherwise(0),
        )
    )

    # Drop strictly temporary intermediate parsing columns
    temp_cols_to_drop = [
        "raw_distance_km",
        "user_lat",
        "user_lng",
        "dest_lat",
        "dest_lng",
    ]
    cols_to_drop = [c for c in temp_cols_to_drop if c not in original_cols]

    return df_features.drop(*cols_to_drop)
