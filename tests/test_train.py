import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train import (
    CAT_COLS,
    FEATURE_COLS,
    TARGET_COL,
    load_data,
    train_conversion_model,
)


@pytest.fixture
def sample_featured_df():
    """Generates a synthetic Pandas DataFrame with valid feature distributions for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "user_device": np.random.choice(["desktop", "mobile", "tablet"], n_samples),
        "user_osName": np.random.choice(
            ["macOS", "Windows", "Android", "iOS"], n_samples
        ),
        "user_browserName_clean": np.random.choice(
            ["chrome", "safari", "firefox"], n_samples
        ),
        "subscriber_tier": np.random.choice(
            ["bronze", "silver", "gold", "platinum"], n_samples
        ),  # Added column
        "travel_distance_km": np.random.uniform(50.0, 3000.0, n_samples),
        "is_long_haul": np.random.choice([0, 1], n_samples),
        "adr_clean": np.random.uniform(50.0, 500.0, n_samples),
        "cross_sell_score": np.random.choice([0.0, 1.0, 2.0], n_samples),
        "mobile_ux_friction": np.random.choice([0, 1], n_samples),
        "expected_gross_commission": np.random.uniform(5.0, 60.0, n_samples),
        TARGET_COL: np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }

    df = pd.DataFrame(data)
    return df


def test_load_data_missing_file():
    """Ensures load_data raises FileNotFoundError when input Parquet file is missing."""
    with pytest.raises(FileNotFoundError):
        load_data("data/processed/non_existent_file.parquet")


def test_load_data_categorical_conversion(tmp_path, sample_featured_df):
    """Ensures load_data correctly converts specified string columns to 'category' dtypes."""
    parquet_path = tmp_path / "test_featured.parquet"
    sample_featured_df.to_parquet(parquet_path)

    df_loaded = load_data(str(parquet_path))

    for cat_col in CAT_COLS:
        assert df_loaded[cat_col].dtype.name == "category"


def test_train_conversion_model_execution(sample_featured_df):
    """Verifies that LightGBM trains successfully and produces valid evaluation metrics."""
    # Cast categorical columns dynamically using CAT_COLS
    for col in CAT_COLS:
        sample_featured_df[col] = sample_featured_df[col].astype("category")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 10,
        "verbose": -1,
    }

    model, metrics = train_conversion_model(sample_featured_df, params=params)

    # Check metrics structure and bounds
    assert "val_roc_auc" in metrics
    assert "val_log_loss" in metrics
    assert 0.0 <= metrics["val_roc_auc"] <= 1.0
    assert metrics["val_log_loss"] >= 0.0

    # Test probability output bounds on sample input
    X_sample = sample_featured_df[FEATURE_COLS].iloc[:5]
    probs = model.predict_proba(X_sample)[:, 1]

    assert len(probs) == 5
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_model_artifact_persistence(tmp_path, sample_featured_df):
    """Verifies that the trained LightGBM model can be serialized and reloaded via joblib."""
    for col in CAT_COLS:
        sample_featured_df[col] = sample_featured_df[col].astype("category")

    model, _ = train_conversion_model(
        sample_featured_df,
        params={"objective": "binary", "n_estimators": 5, "verbose": -1},
    )

    artifact_path = tmp_path / "lgbm_conversion_model.pkl"
    joblib.dump(model, artifact_path)

    assert os.path.exists(artifact_path)

    # Reload model and verify prediction consistency
    reloaded_model = joblib.load(artifact_path)
    X_test = sample_featured_df[FEATURE_COLS].iloc[:2]

    original_preds = model.predict_proba(X_test)
    reloaded_preds = reloaded_model.predict_proba(X_test)

    np.testing.assert_array_almost_equal(original_preds, reloaded_preds)
