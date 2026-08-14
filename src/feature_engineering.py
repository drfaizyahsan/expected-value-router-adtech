import math
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType

from utils import get_logger

logger = get_logger()


def parse_coordinate(col_name: str, index: int):
    """
    Extracts latitude (index=1) or longitude (index=2) float from regex pattern match.
    Matches patterns like '45.5017,-73.5673' or '45.5017;-73.5673'.
    """
    pattern = r"(-?\d+\.\d+)[,\s;]+(-?\d+\.\d+)"
    val = F.regexp_extract(F.col(col_name), pattern, index).try_cast(DoubleType())

    if index == 1:
        return F.when((val >= -90.0) & (val <= 90.0), val).otherwise(None)
    else:
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
    """Applies distributed feature engineering transformations using PySpark DataFrame API."""

    # 1. Clean Browser Names & Impute Missing Binary Signals
    clean_browser_expr = (
        F.when(F.lower(F.trim(F.col("user_browserName"))).contains("chrome"), "chrome")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("safari"), "safari")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("firefox"), "firefox")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("edge"), "edge")
        .when(F.lower(F.trim(F.col("user_browserName"))).contains("samsung"), "samsung")
        .when(F.col("user_browserName").isNotNull(), "other")
        .otherwise("unknown")
    )

    df_cleaned = df.withColumn("user_browserName_clean", clean_browser_expr).fillna(
        {"booked_rental": 0.0, "booked_flight": 0.0}
    )

    # 2. Extract Spatial Coordinates
    df_coords = (
        df_cleaned.withColumn("user_lat", parse_coordinate("user_lat_lng", 1))
        .withColumn("user_lng", parse_coordinate("user_lat_lng", 2))
        .withColumn("dest_lat", parse_coordinate("dest_lat_lng", 1))
        .withColumn("dest_lng", parse_coordinate("dest_lat_lng", 2))
    )

    # 3. Compute Distance & Approximate Median Imputation
    df_dist = df_coords.withColumn(
        "raw_distance_km",
        compute_haversine_distance(
            F.col("user_lat"), F.col("user_lng"), F.col("dest_lat"), F.col("dest_lng")
        ),
    )

    # Approximate median over distributed partition
    median_dist = df_dist.stat.approxQuantile("raw_distance_km", [0.5], 0.01)[0]
    if median_dist is None or math.isnan(median_dist):
        median_dist = 500.0

    df_features = df_dist.withColumn(
        "travel_distance_km",
        F.coalesce(F.col("raw_distance_km"), F.lit(float(median_dist))),
    ).withColumn(
        "is_long_haul", F.when(F.col("travel_distance_km") > 1000.0, 1).otherwise(0)
    )

    # 4. Outlier Clipping & Domain Interaction Features
    df_final = (
        df_features.withColumn(
            "adr_clean",
            F.when(F.col("avg_daily_rate") < 10.0, 10.0).otherwise(
                F.col("avg_daily_rate")
            ),
        )
        .withColumn("cross_sell_score", F.col("booked_flight") + F.col("booked_rental"))
        .withColumn(
            "mobile_ux_friction",
            F.when(
                (F.col("user_device") == "mobile") & (F.col("mobile_optimized") == 0), 1
            ).otherwise(0),
        )
        .withColumn(
            "expected_gross_commission", F.col("adr_clean") * F.col("commission_rate")
        )
        .drop("raw_distance_km")
    )

    # Generate binary conversion target from synthetic ground truth probability
    if (
        "p_conversion_ground_truth" in df_final.columns
        and "is_conversion" not in df_final.columns
    ):
        df_final = df_final.withColumn(
            "is_conversion",
            F.when(F.rand(seed=42) < F.col("p_conversion_ground_truth"), 1).otherwise(
                0
            ),
        )

    return df_final


def main():
    raw_path = "data/raw/session_subscriber_pairs.csv"
    output_path = "data/processed/featured_pairs.parquet"

    spark = (
        SparkSession.builder.appName("ExpectedValueRouter-FeatureEngineering")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    df_raw = (
        spark.read.option("header", "true").option("inferSchema", "true").csv(raw_path)
    )
    logger.info(f"df_raw shape: {df_raw.count()} {len(df_raw.columns)}")

    df_featured = engineer_features_spark(df_raw)
    logger.info(f"processed df shape: {df_featured.count()} {len(df_featured.columns)}")

    os.makedirs("data/processed", exist_ok=True)
    df_featured.write.mode("overwrite").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    main()
