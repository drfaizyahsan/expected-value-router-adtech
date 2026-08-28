# src/train.py
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import candidate feature matrix builder or dataset loader


def train_and_evaluate():
    """Trains LightGBM conversion model, computes stratified random baseline,

    and validates PR-AUC performance margin.
    """
    mlflow.set_experiment("ev_traffic_router")

    with mlflow.start_run(run_name="lgb_conversion_baseline_comparison"):
        # 1. Load data or generate synthetic training dataset
        print("Loading training dataset...")
        # Replace with your actual dataset loading logic/filepath if using saved CSV/Parquet
        if os.path.exists("data/train_conversions.csv"):
            df = pd.read_csv("data/train_conversions.csv")
            X = df.drop(columns=["converted"])
            y = df["converted"].values
        else:
            # Fallback inline dummy data generation for pipeline sanity checks
            np.random.seed(42)
            n_samples = 5000
            X = pd.DataFrame(
                {
                    "distance_km": np.random.uniform(50, 1500, n_samples),
                    "commission_rate": np.random.uniform(0.05, 0.25, n_samples),
                    "booking_rate": np.random.uniform(100, 500, n_samples),
                    "is_mobile": np.random.choice([0, 1], size=n_samples),
                    "booked_flight": np.random.choice([0, 1], size=n_samples),
                }
            )
            # True probability dependent on commission & booking status
            true_prob = 1 / (
                1
                + np.exp(
                    -(
                        -2.0
                        + 3.0 * X["commission_rate"]
                        + 0.5 * X["booked_flight"]
                        - 0.001 * X["distance_km"]
                    )
                )
            )
            y = np.random.binomial(1, true_prob)

        # 2. Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ------------------------------------------------------------------
        # 3. Compute Stratified Random Baseline Metrics
        # ------------------------------------------------------------------
        positive_class_ratio = float(np.mean(y_train))

        # Stratified random predictor probabilities drawn from training distribution
        np.random.seed(42)
        y_pred_baseline = np.random.binomial(
            1, positive_class_ratio, size=len(y_test)
        ).astype(float)

        baseline_pr_auc = float(average_precision_score(y_test, y_pred_baseline))
        baseline_roc_auc = float(roc_auc_score(y_test, y_pred_baseline))

        print("\n--- Stratified Random Baseline ---")
        print(f"Positive Class Ratio (Prevalence): {positive_class_ratio:.4f}")
        print(f"Baseline PR-AUC:                  {baseline_pr_auc:.4f}")
        print(f"Baseline ROC-AUC:                 {baseline_roc_auc:.4f}")

        # ------------------------------------------------------------------
        # 4. Train LightGBM Model
        # ------------------------------------------------------------------
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # Predict probabilities on test set
        y_pred_lgbm = model.predict_proba(X_test)[:, 1]

        lgbm_pr_auc = float(average_precision_score(y_test, y_pred_lgbm))
        lgbm_roc_auc = float(roc_auc_score(y_test, y_pred_lgbm))

        # Absolute and Relative Margins
        pr_auc_lift_abs = lgbm_pr_auc - baseline_pr_auc
        pr_auc_lift_pct = ((lgbm_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0

        print("\n--- Trained LightGBM Model ---")
        print(f"LightGBM PR-AUC:                  {lgbm_pr_auc:.4f}")
        print(f"LightGBM ROC-AUC:                 {lgbm_roc_auc:.4f}")
        print(
            f"PR-AUC Margin Over Baseline:      +{pr_auc_lift_abs:.4f} ({pr_auc_lift_pct:.2f}% lift)\n"
        )

        # Assert performance criterion
        assert lgbm_pr_auc > baseline_pr_auc, (
            f"LightGBM PR-AUC ({lgbm_pr_auc:.4f}) failed to beat baseline ({baseline_pr_auc:.4f})"
        )

        # ------------------------------------------------------------------
        # 5. Log Everything to MLflow
        # ------------------------------------------------------------------
        mlflow.log_params(
            {
                "model_type": "LGBMClassifier",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

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

        # 6. Save Model Checkpoint
        os.makedirs("models", exist_ok=True)
        model_path = "models/lgbm_conversion_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
