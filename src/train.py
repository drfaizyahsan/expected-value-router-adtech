import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import CAT_COLS, FEATURE_COLS, TARGET_COL


def load_data(
    data_path: str = "data/processed/featured_pairs.parquet",
) -> pd.DataFrame:
    """Loads feature dataset or generates schema-compliant synthetic fallback data."""
    if os.path.exists(data_path):
        print(f"Loading feature dataset from {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        print(
            f"Warning: {data_path} not found. Generating schema-compliant synthetic fallback data..."
        )
        np.random.seed(42)
        n_samples = 3000

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
            ),
            "travel_distance_km": np.random.uniform(50.0, 3000.0, n_samples),
            "is_long_haul": np.random.choice([0, 1], n_samples),
            "adr_clean": np.random.uniform(50.0, 500.0, n_samples),
            "cross_sell_score": np.random.choice([0.0, 1.0, 2.0], n_samples),
            "mobile_ux_friction": np.random.choice([0, 1], n_samples),
        }

        tier_boost = np.where(data["subscriber_tier"] == "platinum", 1.5, 0.0)
        logits = -2.0 + tier_boost
        prob = 1 / (1 + np.exp(-logits))
        data[TARGET_COL] = np.random.binomial(1, prob)

        df = pd.DataFrame(data)

    # Cast string categoricals to pandas 'category' for native LightGBM handling
    for cat_col in CAT_COLS:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    return df


def train_and_evaluate(data_path: str = "data/processed/featured_pairs.parquet"):
    """Trains LightGBM conversion classifier, evaluates against stratified baseline, and logs artifacts to MLflow."""
    mlflow.set_experiment("ev_traffic_router")

    df = load_data(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].values

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "is_unbalance": True,
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lgb_conversion_baseline_comparison"):
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Stratified Random Prediction Baseline
        positive_class_ratio = float(np.mean(y_train))
        np.random.seed(42)
        y_pred_baseline = np.random.binomial(
            1, positive_class_ratio, size=len(y_test)
        ).astype(float)

        baseline_pr_auc = float(average_precision_score(y_test, y_pred_baseline))
        baseline_roc_auc = float(roc_auc_score(y_test, y_pred_baseline))

        # 2. Native LightGBM Classifier Fit
        model = LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)

        # Predict conversion probabilities
        y_pred_lgbm = model.predict_proba(X_test)[:, 1]

        # Classification Metrics
        lgbm_pr_auc = float(average_precision_score(y_test, y_pred_lgbm))
        lgbm_roc_auc = float(roc_auc_score(y_test, y_pred_lgbm))

        pr_auc_lift_abs = lgbm_pr_auc - baseline_pr_auc
        pr_auc_lift_pct = ((lgbm_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0

        print(
            f"Baseline PR-AUC: {baseline_pr_auc:.4f} | ROC-AUC: {baseline_roc_auc:.4f}"
        )
        print(f"LGBM PR-AUC:     {lgbm_pr_auc:.4f} | ROC-AUC: {lgbm_roc_auc:.4f}")
        print(f"PR-AUC Lift:     {pr_auc_lift_abs:+.4f} ({pr_auc_lift_pct:+.2f}%)")

        assert lgbm_pr_auc >= baseline_pr_auc, (
            f"LightGBM PR-AUC ({lgbm_pr_auc:.4f}) failed to beat baseline ({baseline_pr_auc:.4f})"
        )

        # MLflow Logging
        mlflow.log_params({**lgb_params, "feature_cols": FEATURE_COLS})
        mlflow.log_metrics(
            {
                "baseline_pr_auc": baseline_pr_auc,
                "baseline_roc_auc": baseline_roc_auc,
                "positive_class_ratio": positive_class_ratio,
                "lgbm_pr_auc": lgbm_pr_auc,
                "lgbm_roc_auc": lgbm_roc_auc,
                "pr_auc_lift_absolute": pr_auc_lift_abs,
                "pr_auc_lift_percent": pr_auc_lift_pct,
            }
        )

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        model_path = "models/lgbm_conversion_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

    return model


if __name__ == "__main__":
    train_and_evaluate()
