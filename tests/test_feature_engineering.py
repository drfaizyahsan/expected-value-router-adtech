import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

try:
    from feature_engineering import (
        FEATURE_COLS,
        TARGET_COL,
        engineer_features_spark,
    )
except ImportError:
    from src.feature_engineering import (
        FEATURE_COLS,
        TARGET_COL,
        engineer_features_spark,
    )


@pytest.fixture(scope="module")
def spark_session():
    """Provides a local SparkSession for pipeline testing."""
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test_feature_engineering")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def mock_session_data(spark_session):
    """Generates mock input session data matching the full raw schema contract expected by engineer_features_spark."""
    schema = StructType(
        [
            StructField("user_browserName", StringType(), True),
            StructField("user_osName", StringType(), True),
            StructField("user_lat_lng", StringType(), True),
            StructField("dest_lat_lng", StringType(), True),
            StructField("booked_flight", DoubleType(), True),
            StructField("booked_rental", DoubleType(), True),
            StructField("avg_daily_rate", DoubleType(), True),
            StructField("user_device", StringType(), True),
            StructField("mobile_optimized", IntegerType(), True),
            StructField("commission_rate", DoubleType(), True),
            StructField("p_conversion_ground_truth", DoubleType(), True),
            StructField("subscriber_tier", StringType(), True),
        ]
    )

    data = [
        (
            "Chrome 118.0",
            "iOS 16.1",
            "45.5017,-73.5673",
            "40.7128,-74.0060",
            1.0,
            0.0,
            120.0,
            "mobile",
            0,
            0.15,
            0.8,
            "gold",
        ),
        (
            "Safari 16.0",
            "macOS 13.0",
            "45.5017,-73.5673",
            "34.0522,-118.2437",
            0.0,
            1.0,
            250.0,
            "desktop",
            1,
            0.10,
            0.4,
            "silver",
        ),
    ]

    return spark_session.createDataFrame(data, schema=schema)


def test_feature_engineering_pipeline(mock_session_data):
    """Validates that engineer_features_spark correctly transforms raw input data and emits required schema columns."""
    output_df = engineer_features_spark(mock_session_data)
    result_columns = output_df.columns

    # Verify all target feature columns exist in output
    for expected_col in FEATURE_COLS:
        assert expected_col in result_columns, (
            f"Missing feature column in output: {expected_col}"
        )

    assert TARGET_COL in result_columns

    # Row value assertions
    first_row = output_df.first()
    assert first_row["user_browserName_clean"] == "chrome"
    assert first_row["cross_sell_score"] == 1.0
    assert first_row["expected_gross_commission"] == pytest.approx(18.0)
    assert first_row["mobile_ux_friction"] == 1
