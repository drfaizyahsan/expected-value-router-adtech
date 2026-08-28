from unittest.mock import MagicMock

import numpy as np
import pytest

import src.train as train_module
from app.main import build_candidate_feature_matrix
from src.feature_engineering import CAT_COLS, FEATURE_COLS
from src.train import load_data, train_and_evaluate


@pytest.fixture
def mock_mlflow(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr(train_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(
        train_module.mlflow, "start_run", MagicMock(return_value=mock_run)
    )
    monkeypatch.setattr(train_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(train_module.mlflow, "log_metrics", MagicMock())
    monkeypatch.setattr(train_module.mlflow, "log_artifact", MagicMock())
    return mock_run


def test_load_data_schema(tmp_path):
    df = load_data("non_existent_file.parquet")
    for col in FEATURE_COLS:
        assert col in df.columns
    for cat_col in CAT_COLS:
        assert df[cat_col].dtype.name == "category"


def test_train_and_evaluate_execution(tmp_path, monkeypatch, mock_mlflow):
    monkeypatch.chdir(tmp_path)
    train_and_evaluate()

    expected_model_path = tmp_path / "models" / "lgbm_conversion_model.pkl"
    assert expected_model_path.exists()

    logged_metrics = train_module.mlflow.log_metrics.call_args[0][0]
    assert logged_metrics["lgbm_pr_auc"] >= logged_metrics["baseline_pr_auc"]


def test_trained_model_inference_compatibility_with_route(
    tmp_path, monkeypatch, mock_mlflow
):
    monkeypatch.chdir(tmp_path)
    model = train_and_evaluate()

    raw_request_payload = {
        "user_device": "mobile",
        "user_osName": "iOS",
        "user_browserName_clean": "safari",
        "subscriber_tier": "gold",
        "travel_distance_km": 450.0,
        "is_long_haul": 0,
        "adr_clean": 180.0,
        "candidates": [
            {
                "partner_id": "p_101",
                "cross_sell_score": 1.0,
                "mobile_ux_friction": 0,
                "expected_gross_commission": 25.0,
            },
            {
                "partner_id": "p_102",
                "cross_sell_score": 0.0,
                "mobile_ux_friction": 1,
                "expected_gross_commission": 15.0,
            },
        ],
    }

    X_candidate = build_candidate_feature_matrix(raw_request_payload)
    probs = model.predict_proba(X_candidate)[:, 1]

    assert len(probs) == 2
    assert np.all((probs >= 0.0) & (probs <= 1.0))
