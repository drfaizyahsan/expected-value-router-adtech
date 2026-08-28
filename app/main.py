import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.logging import log_routing_decision
from app.schemas import (
    CandidateSubscriber,
    RankedCandidateResponse,
    RoutingRequest,
    RoutingResponse,
    UserContext,
)
from src.feature_engineering import CAT_COLS, FEATURE_COLS
from src.policy import epsilon_greedy_policy

app = FastAPI(
    title="Expected Value Traffic Router API",
    description="Production inference endpoint executing p(conversion) prediction and Epsilon-Greedy expected value routing with decision logging.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_conversion_model.pkl")
model = None


@app.on_event("startup")
def load_model():
    """Loads trained LightGBM model artifact on startup."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading LightGBM model artifact from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Running uninitialized.")


def compute_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes Haversine distance in kilometers between two geo coordinates."""
    r = 6371.0  # Earth's radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def build_candidate_feature_matrix(
    user: UserContext,
    candidate: CandidateSubscriber,
    dist_km: float,
    is_long_haul: int,
    cross_sell_score: float,
    mobile_ux_friction: int,
    browser_clean: str,
) -> pd.DataFrame:
    """Transforms UserContext and CandidateSubscriber into a single-row DataFrame

    aligned strictly with FEATURE_COLS and CAT_COLS for LightGBM inference.
    """

    feature_dict = {
        "user_device": user.user_device,
        "user_osName": user.user_osName,
        "user_browserName_clean": browser_clean,
        "subscriber_tier": candidate.subscriber_tier,
        "travel_distance_km": dist_km,
        "is_long_haul": is_long_haul,
        "adr_clean": candidate.booking_rate,
        "cross_sell_score": cross_sell_score,
        "mobile_ux_friction": mobile_ux_friction,
    }

    # Align with strict feature order expected by trained model
    input_df = pd.DataFrame([feature_dict])[FEATURE_COLS]

    # Cast string categorical columns to categorical dtype for LightGBM
    for cat_col in CAT_COLS:
        if cat_col in input_df.columns:
            input_df[cat_col] = input_df[cat_col].astype("category")

    return input_df


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/route", response_model=RoutingResponse)
def route_traffic(payload: RoutingRequest):
    """Predicts conversion probabilities using schemas.py contracts, applies policy routing, and logs decisions."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Please train the model first.",
        )

    # Explicitly pull typed objects from request payload
    user: UserContext = payload.user
    candidates: list[CandidateSubscriber] = payload.candidates

    # Feature transformation from UserContext
    dist_km = compute_haversine_km(
        user.user_lat, user.user_lng, user.dest_lat, user.dest_lng
    )
    is_long_haul = 1 if dist_km > 1000.0 else 0
    cross_sell_score = float(int(user.booked_flight) + int(user.booked_rental))
    mobile_ux_friction = 1 if user.user_device in ["mobile", "tablet"] else 0
    browser_clean = user.user_browserName.lower().strip()

    evaluated_candidates = []
    ranked_candidates_response: list[RankedCandidateResponse] = []

    # Process each CandidateSubscriber instance
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

        # Cast string categorical columns using CAT_COLS for LightGBM
        for cat_col in CAT_COLS:
            if cat_col in input_df.columns:
                input_df[cat_col] = input_df[cat_col].astype("category")

        # Predict conversion probability
        try:
            p_conv = float(model.predict_proba(input_df)[0, 1])
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            raise HTTPException(
                status_code=500, detail=f"Model inference failed: {e!s}"
            )

        expected_value = float(p_conv * cand.booking_rate * cand.commission_rate)

        cand_record = {
            "subscriber_id": cand.subscriber_id,
            "subscriber_name": cand.subscriber_name,
            "p_conversion": p_conv,
            "commission_rate": cand.commission_rate,
            "booking_rate": cand.booking_rate,
            "expected_value": expected_value,
        }
        evaluated_candidates.append(cand_record)

        ranked_candidates_response.append(
            RankedCandidateResponse(
                subscriber_id=cand.subscriber_id,
                subscriber_name=cand.subscriber_name,
                p_conversion=p_conv,
                commission_rate=cand.commission_rate,
                booking_rate=cand.booking_rate,
                expected_value=expected_value,
            )
        )

    # Sort evaluated candidates by expected value descending
    evaluated_candidates.sort(key=lambda x: x["expected_value"], reverse=True)
    ranked_candidates_response.sort(key=lambda x: x.expected_value, reverse=True)

    # Execute policy routing
    selected_candidate, propensity, is_exploration = epsilon_greedy_policy(
        evaluated_candidates=evaluated_candidates,
        epsilon=0.10,
    )

    # Log routing decision
    try:
        log_routing_decision(
            user_context=user.model_dump(),
            candidates=[c.model_dump() for c in candidates],
            selected_subscriber_id=selected_candidate["subscriber_id"],
            selected_subscriber_name=selected_candidate["subscriber_name"],
            max_expected_value=selected_candidate["expected_value"],
            propensity_score=propensity,
            policy_version="epsilon_greedy_v1",
        )
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        print(f"Warning: Decision logging failed - {e!s}")

    # Construct strict RoutingResponse
    return RoutingResponse(
        selected_subscriber_id=selected_candidate["subscriber_id"],
        selected_subscriber_name=selected_candidate["subscriber_name"],
        max_expected_value=selected_candidate["expected_value"],
        is_fallback=is_exploration,
        propensity_score=propensity,
        policy_version="epsilon_greedy_v1",
        ranked_candidates=ranked_candidates_response,
    )
