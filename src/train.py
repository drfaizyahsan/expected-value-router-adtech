import os

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import get_logger

logger = get_logger()


FEATURE_COLS = [
    "user_device",
    "user_osName",
    "user_browserName_clean",
    "subscriber_tier",
    "travel_distance_km",
    "is_long_haul",
    "adr_clean",
    "cross_sell_score",
    "mobile_ux_friction",
    "expected_gross_commission",
]

CAT_COLS = ["user_device", "user_osName", "user_browserName_clean", "subscriber_tier"]
TARGET_COL = "is_conversion"


def load_data(parquet_path: str) -> pd.DataFrame:
    """Loads processed parquet data into pandas for GBDT training."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Parquet file missing at {parquet_path}. Run feature engineering first."
        )

    df = pd.read_parquet(parquet_path)

    # Cast categorical columns to pandas 'category' dtype for LightGBM
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def train_conversion_model(
    df: pd.DataFrame,
    params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Trains LightGBM classifier and evaluates ROC-AUC and Log Loss metrics."""

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if params is None:
        params = {
            # "objective": "binary",
            # "metric": "binary_logloss",
            # "boosting_type": "gbdt",
            # "learning_rate": 0.05,
            # "num_leaves": 31,
            "n_estimators": 200,
            "random_state": random_state,
            "verbose": -1,
            "class_weight": "balanced",
        }

    model = lgb.LGBMClassifier(**params)

    # Enable MLflow autologging
    mlflow.lightgbm.autolog()

    with mlflow.start_run(run_name="lgbm_p_conversion"):
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        # Evaluate probabilities
        val_preds_prob = model.predict_proba(X_val)[:, 1]

        auc_score = float(roc_auc_score(y_val, val_preds_prob))
        aupr_score = float(average_precision_score(y_val, val_preds_prob))
        loss_val = float(log_loss(y_val, val_preds_prob))

        metrics = {
            "val_aupr": aupr_score,
            "val_roc_auc": auc_score,
            "val_log_loss": loss_val,
        }
        mlflow.log_metrics(metrics)

    return model, metrics


def main():
    data_path = "data/processed/featured_pairs.parquet"
    model_output_dir = "models"
    os.makedirs(model_output_dir, exist_ok=True)

    df = load_data(data_path)
    logger.info(f"df shape: {df.shape}")
    logger.info(f"df columns: {df.columns}")

    model, metrics = train_conversion_model(df)

    logger.info(f"metrics: {metrics}")

    # Save model artifact locally for FastAPI inference service
    artifact_path = os.path.join(model_output_dir, "lgbm_conversion_model.pkl")
    joblib.dump(model, artifact_path)


if __name__ == "__main__":
    main()
