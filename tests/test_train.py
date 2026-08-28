from unittest.mock import MagicMock

import numpy as np
import pytest

import src.train as train_module
from app.main import build_candidate_feature_matrix
from app.schemas import CandidateSubscriber, UserContext
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

    user = UserContext(
        user_device="mobile",
        user_osName="iOS",
        user_browserName="Safari",
        user_lat=45.5017,
        user_lng=-73.5673,
        dest_lat=40.7128,
        dest_lng=-74.0060,
        booked_flight=True,
        booked_hotel=False,
        booked_rental=False,
    )

    candidates = [
        CandidateSubscriber(
            subscriber_id="p_101",
            subscriber_name="Expedia",
            subscriber_tier="gold",
            commission_rate=0.10,
            booking_rate=250.0,
            mobile_optimized=True,
        ),
        CandidateSubscriber(
            subscriber_id="p_102",
            subscriber_name="Booking.com",
            subscriber_tier="silver",
            commission_rate=0.08,
            booking_rate=180.0,
            mobile_optimized=False,
        ),
    ]

    dist_km = 500.0
    is_long_haul = 0
    cross_sell_score = 1.0
    mobile_ux_friction = 1
    browser_clean = user.user_browserName.lower().strip()

    probs = []
    for cand in candidates:
        input_df = build_candidate_feature_matrix(
            user=user,
            candidate=cand,
            dist_km=dist_km,
            is_long_haul=is_long_haul,
            cross_sell_score=cross_sell_score,
            mobile_ux_friction=mobile_ux_friction,
            browser_clean=browser_clean,
        )
        p_conv = float(model.predict_proba(input_df)[0, 1])
        probs.append(p_conv)

    assert len(probs) == 2
    assert np.all((np.array(probs) >= 0.0) & (np.array(probs) <= 1.0))
