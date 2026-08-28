import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import CAT_COLS, FEATURE_COLS

app = FastAPI(title="Expected Value Traffic Router", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_conversion_model.pkl")
model = None


class Candidate(BaseModel):
    partner_id: str
    cross_sell_score: float = 0.0
    mobile_ux_friction: int = 0
    expected_gross_commission: float = 0.0


class RouteRequest(BaseModel):
    user_device: str = "desktop"
    user_osName: str = "Windows"
    user_browserName_clean: str = "chrome"
    subscriber_tier: str = "bronze"
    travel_distance_km: float = 500.0
    is_long_haul: int = 0
    adr_clean: float = 150.0
    candidates: list[Candidate] = Field(..., min_items=1)


class CandidateRouteScore(BaseModel):
    partner_id: str
    p_conversion: float
    expected_gross_commission: float
    expected_value: float


class RouteResponse(BaseModel):
    selected_partner_id: str
    max_expected_value: float
    routing_scores: list[CandidateRouteScore]


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None


@app.on_event("startup")
def startup_event():
    load_model()


def build_candidate_feature_matrix(payload: dict) -> pd.DataFrame:
    """Transforms raw API route request payload into aligned pandas feature DataFrame."""
    rows = []
    candidates = payload.get("candidates", [])

    for cand in candidates:
        row = {
            "user_device": payload.get("user_device", "desktop"),
            "user_osName": payload.get("user_osName", "Windows"),
            "user_browserName_clean": payload.get("user_browserName_clean", "chrome"),
            "subscriber_tier": payload.get("subscriber_tier", "bronze"),
            "travel_distance_km": float(payload.get("travel_distance_km", 500.0)),
            "is_long_haul": int(payload.get("is_long_haul", 0)),
            "adr_clean": float(payload.get("adr_clean", 150.0)),
            "cross_sell_score": float(cand.get("cross_sell_score", 0.0)),
            "mobile_ux_friction": int(cand.get("mobile_ux_friction", 0)),
            "expected_gross_commission": float(
                cand.get("expected_gross_commission", 0.0)
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Enforce strict column ordering matching training contract
    df = df[FEATURE_COLS]

    # 2. Align categorical dtypes with LightGBM's trained category levels
    if model is not None and hasattr(model, "booster_"):
        trained_categories = model.booster_.pandas_categorical
        for idx, col in enumerate(FEATURE_COLS):
            if col in CAT_COLS and trained_categories[idx] is not None:
                categories = trained_categories[idx]
                df[col] = pd.Categorical(df[col], categories=categories)
    else:
        for col in CAT_COLS:
            df[col] = df[col].astype("category")

    return df


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/route", response_model=RouteResponse)
def route_traffic(request: RouteRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    try:
        payload = request.dict()
        X_candidates = build_candidate_feature_matrix(payload)

        # Predict conversion probabilities
        probs = model.predict_proba(X_candidates)[:, 1]

        routing_scores = []
        best_partner_id = None
        max_ev = -1.0

        for idx, cand in enumerate(request.candidates):
            p_conv = float(probs[idx])
            comm = float(cand.expected_gross_commission)
            ev = p_conv * comm

            score_entry = CandidateRouteScore(
                partner_id=cand.partner_id,
                p_conversion=p_conv,
                expected_gross_commission=comm,
                expected_value=ev,
            )
            routing_scores.append(score_entry)

            if ev > max_ev:
                max_ev = ev
                best_partner_id = cand.partner_id

        return RouteResponse(
            selected_partner_id=best_partner_id,
            max_expected_value=max_ev,
            routing_scores=routing_scores,
        )
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        raise HTTPException(
            status_code=500, detail=f"Inference execution failed: {e!s}"
        )
