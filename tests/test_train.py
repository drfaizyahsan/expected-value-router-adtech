import os
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

import src.train as train_module
from src.train import train_and_evaluate


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mocks MLflow tracking functions to prevent disk writes/network calls during test runs."""
    mock_run = MagicMock()
    monkeypatch.setattr(train_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(
        train_module.mlflow, "start_run", MagicMock(return_value=mock_run)
    )
    monkeypatch.setattr(train_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(train_module.mlflow, "log_metrics", MagicMock())
    monkeypatch.setattr(train_module.mlflow, "log_artifact", MagicMock())
    return mock_run


def test_train_and_evaluate_synthetic_fallback(tmp_path, monkeypatch, mock_mlflow):
    """Verifies complete training execution using the synthetic data fallback when no CSV exists."""
    monkeypatch.chdir(tmp_path)

    train_and_evaluate()

    # Check model artifact creation
    expected_model_path = tmp_path / "models" / "lgbm_conversion_model.pkl"
    assert expected_model_path.exists()

    # Reload model to ensure validity
    model = joblib.load(expected_model_path)
    assert hasattr(model, "predict_proba")

    # Verify MLflow logging calls
    train_module.mlflow.log_params.assert_called_once()
    train_module.mlflow.log_metrics.assert_called_once()
    train_module.mlflow.log_artifact.assert_called_with(
        "models/lgbm_conversion_model.pkl"
    )

    # Inspect logged metrics to ensure model beats baseline
    logged_metrics = train_module.mlflow.log_metrics.call_args[0][0]
    assert logged_metrics["lgbm_pr_auc"] > logged_metrics["baseline_pr_auc"]
    assert logged_metrics["pr_auc_lift_percent"] > 0.0


def test_train_and_evaluate_with_csv(tmp_path, monkeypatch, mock_mlflow):
    """Verifies that train_and_evaluate correctly reads and processes an existing CSV file."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    # Generate synthetic CSV file with deterministic target relationship
    np.random.seed(42)
    n_samples = 500
    commission_rate = np.random.uniform(0.05, 0.25, n_samples)
    booked_flight = np.random.choice([0, 1], size=n_samples)
    distance_km = np.random.uniform(50, 1500, n_samples)

    # True probability linked to features so model learns real signal
    true_prob = 1 / (
        1
        + np.exp(
            -(-2.0 + 4.0 * commission_rate + 1.0 * booked_flight - 0.001 * distance_km)
        )
    )
    converted = np.random.binomial(1, true_prob)

    df = pd.DataFrame(
        {
            "distance_km": distance_km,
            "commission_rate": commission_rate,
            "booking_rate": np.random.uniform(100, 500, n_samples),
            "is_mobile": np.random.choice([0, 1], size=n_samples),
            "booked_flight": booked_flight,
            "converted": converted,
        }
    )
    df.to_csv("data/train_conversions.csv", index=False)

    train_and_evaluate()

    expected_model_path = tmp_path / "models" / "lgbm_conversion_model.pkl"
    assert expected_model_path.exists()

    logged_metrics = train_module.mlflow.log_metrics.call_args[0][0]
    assert "baseline_pr_auc" in logged_metrics
    assert "lgbm_pr_auc" in logged_metrics
    assert logged_metrics["lgbm_pr_auc"] > logged_metrics["baseline_pr_auc"]


def test_stratified_baseline_assertion_failure(tmp_path, monkeypatch, mock_mlflow):
    """Ensures an AssertionError is raised if LightGBM fails to beat the random baseline."""
    monkeypatch.chdir(tmp_path)

    # Mock average_precision_score to simulate model underperforming baseline
    def mock_average_precision(y_true, y_pred):
        if np.array_equal(y_pred, y_pred.astype(float)) and len(np.unique(y_pred)) > 2:
            return 0.10
        return 0.50

    with (
        patch("src.train.average_precision_score", side_effect=mock_average_precision),
        pytest.raises(AssertionError, match="failed to beat baseline"),
    ):
        train_and_evaluate()
