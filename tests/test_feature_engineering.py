import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.feature_engineering import engineer_features_spark


@pytest.fixture(scope="session")
def spark():
    """Session-wide SparkSession fixture for unit testing."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("TestFeatureEngineering")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_raw_df(spark):
    """Creates a small PySpark DataFrame covering normal, missing, and corrupted rows."""
    schema = StructType(
        [
            StructField("user_device", StringType(), True),
            StructField("user_osName", StringType(), True),
            StructField("user_browserName", StringType(), True),
            StructField("user_lat_lng", StringType(), True),
            StructField("dest_lat_lng", StringType(), True),
            StructField("booked_flight", DoubleType(), True),
            StructField("booked_rental", DoubleType(), True),
            StructField("avg_daily_rate", DoubleType(), True),
            StructField("commission_rate", DoubleType(), True),
            StructField("mobile_optimized", IntegerType(), True),
        ]
    )

    data = [
        # Row 0: Valid baseline (Montreal -> Toronto)
        ("desktop", "macOS", "Safari", "45.5017,-73.5673", "43.6532,-79.3832", 1.0, 0.0, 150.0, 0.12, 1),
        # Row 1: Dirty browser, missing rental signal, negative ADR anomaly, mobile UX friction
        ("mobile", "Android", " CHROME_v118 ", "34.0522,-118.2437", "36.1699,-115.1398", 0.0, None, -99.0, 0.10, 0),
        # Row 2: Malformed coords, missing browser (NaN)
        ("tablet", "iOS", None, "INVALID_COORD", "999.0,800.0", None, 1.0, 250.0, 0.15, 1),
    ]

    return spark.createDataFrame(data, schema)


def test_browser_cleaning_and_null_imputation(spark, sample_raw_df):
    """Tests browser string normalization and binary signal null imputation."""
    df_result = engineer_features_spark(sample_raw_df)
    results = df_result.select("user_browserName_clean", "booked_rental", "booked_flight").collect()

    # Row 0: Standard browser
    assert results[0]["user_browserName_clean"] == "safari"
    assert results[0]["booked_rental"] == 0.0

    # Row 1: Dirty browser ' CHROME_v118 ' -> 'chrome', None -> 0.0
    assert results[1]["user_browserName_clean"] == "chrome"
    assert results[1]["booked_rental"] == 0.0

    # Row 2: None browser -> 'unknown', None flight -> 0.0
    assert results[2]["user_browserName_clean"] == "unknown"
    assert results[2]["booked_flight"] == 0.0


def test_coordinate_parsing_and_distance(spark, sample_raw_df):
    """Tests spatial regex extraction, Haversine computation, and invalid coordinate fallback."""
    df_result = engineer_features_spark(sample_raw_df)
    results = df_result.select("travel_distance_km", "is_long_haul", "user_lat", "user_lng").collect()

    # Row 0: Valid distance Montreal -> Toronto (~500 km)
    assert 450.0 < results[0]["travel_distance_km"] < 550.0
    assert results[0]["is_long_haul"] == 0
    assert results[0]["user_lat"] == pytest.approx(45.5017, rel=1e-3)

    # Row 2: Invalid coordinate string should be imputed with median distance
    assert results[2]["user_lat"] is None
    assert results[2]["travel_distance_km"] > 0.0


def test_adr_clipping_and_financial_utility(spark, sample_raw_df):
    """Tests negative ADR clipping ($10.00 min bound) and expected gross commission math."""
    df_result = engineer_features_spark(sample_raw_df)
    results = df_result.select("adr_clean", "expected_gross_commission").collect()

    # Row 0: Normal ADR $150.00 * 0.12 commission = $18.00
    assert results[0]["adr_clean"] == 150.0
    assert results[0]["expected_gross_commission"] == pytest.approx(18.0)

    # Row 1: Negative ADR (-$99.00) clipped to $10.00 * 0.10 commission = $1.00
    assert results[1]["adr_clean"] == 10.0
    assert results[1]["expected_gross_commission"] == pytest.approx(1.0)


def test_domain_interaction_features(spark, sample_raw_df):
    """Tests cross-sell score sum and mobile UX friction flag logic."""
    df_result = engineer_features_spark(sample_raw_df)
    results = df_result.select("cross_sell_score", "mobile_ux_friction").collect()

    # Row 0: flight (1.0) + rental (0.0) = 1.0, desktop device = 0 friction
    assert results[0]["cross_sell_score"] == 1.0
    assert results[0]["mobile_ux_friction"] == 0

    # Row 1: flight (0.0) + rental (0.0 imputed) = 0.0, mobile + non-optimized = 1 friction
    assert results[1]["cross_sell_score"] == 0.0
    assert results[1]["mobile_ux_friction"] == 1