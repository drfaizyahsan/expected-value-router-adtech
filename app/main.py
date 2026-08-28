import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from app.logging import log_routing_decision
from app.schemas import (
    CandidateSubscriber,
    RankedCandidateResponse,
    RoutingRequest,
    RoutingResponse,
    UserContext,
)
from src.policy import RoutingCandidate, rank_and_route_candidates

# Model path setup
MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_conversion_model.pkl")
model_artifact = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model artifact into memory on server startup."""
    global model_artifact
    if os.path.exists(MODEL_PATH):
        model_artifact = joblib.load(MODEL_PATH)
    else:
        # Fallback for testing environments where model artifact may be mocked/built dynamically
        model_artifact = None
    yield


app = FastAPI(
    title="Expected Value Traffic Router API",
    description="Real-time ad traffic routing service powered by LightGBM & Expected Value policy.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Feature Extraction Helpers
# ------------------------------------------------------------------


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculates geodesic distance between two points in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def build_candidate_feature_matrix(
    user: UserContext, candidates: list[CandidateSubscriber]
) -> pd.DataFrame:
    """Transforms raw request context into the exact feature matrix expected by LightGBM."""
    dist_km = calculate_haversine_distance(
        user.user_lat, user.user_lng, user.dest_lat, user.dest_lng
    )
    is_long_haul = 1 if dist_km >= 1000.0 else 0
    cross_sell_score = float(sum([user.booked_flight, user.booked_rental]))

    # Standardize browser name
    browser_clean = user.user_browserName.strip().lower()

    rows = []
    for c in candidates:
        # Calculate mobile friction (1 if on mobile and subscriber site is NOT mobile optimized)
        friction = (
            1
            if (user.user_device.lower() == "mobile" and not c.mobile_optimized)
            else 0
        )
        expected_gross_comm = c.commission_rate * c.booking_rate

        rows.append(
            {
                "user_device": user.user_device.lower(),
                "user_osName": user.user_osName,
                "user_browserName_clean": browser_clean,
                "subscriber_tier": c.subscriber_tier.lower(),
                "travel_distance_km": dist_km,
                "is_long_haul": is_long_haul,
                "adr_clean": c.booking_rate,
                "cross_sell_score": cross_sell_score,
                "mobile_ux_friction": friction,
                "expected_gross_commission": expected_gross_comm,
            }
        )

    df = pd.DataFrame(rows)

    # Cast categoricals matching train schema
    cat_cols = [
        "user_device",
        "user_osName",
        "user_browserName_clean",
        "subscriber_tier",
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "model_loaded": model_artifact is not None,
    }


@app.post("/route", response_model=RoutingResponse, status_code=status.HTTP_200_OK)
def route_traffic(request: RoutingRequest):
    """Predicts conversion probabilities across candidate subscribers and routes traffic to max EV target."""
    if model_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Train model or supply valid model checkpoint.",
        )

    # 1. Build feature matrix for all candidates in the batch
    X_features = build_candidate_feature_matrix(request.user, request.candidates)

    # 2. Vectorized LightGBM probability predictions
    try:
        p_conversions = model_artifact.predict_proba(X_features)[:, 1]
    except (ValueError, KeyError, AttributeError, RuntimeError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e!s}",
        )

    # 3. Construct RoutingCandidate instances
    routing_candidates = [
        RoutingCandidate(
            subscriber_id=candidate.subscriber_id,
            subscriber_name=candidate.subscriber_name,
            commission_rate=candidate.commission_rate,
            booking_rate=candidate.booking_rate,
            p_conversion=float(p_conversions[i]),
        )
        for i, candidate in enumerate(request.candidates)
    ]

    # 4. Rank, explore (epsilon-greedy), and compute propensity score in policy layer
    decision = rank_and_route_candidates(
        routing_candidates, epsilon=0.10, min_ev_threshold=0.05
    )

    # 5. Log decision context using propensity score from policy
    log_routing_decision(
        user_context=request.user.model_dump(),
        candidates=[c.model_dump() for c in request.candidates],
        selected_subscriber_id=decision.selected_subscriber_id,
        selected_subscriber_name=decision.selected_subscriber_name,
        max_expected_value=decision.max_expected_value,
        propensity_score=decision.propensity_score,  # <-- Extracted directly from decision
        policy_version="epsilon_greedy_v1",
    )

    # 6. Build response payload
    ranked_responses = [
        RankedCandidateResponse(
            subscriber_id=c["subscriber_id"],
            subscriber_name=c["subscriber_name"],
            p_conversion=c["p_conversion"],
            commission_rate=c["commission_rate"],
            booking_rate=c["booking_rate"],
            expected_value=c["expected_value"],
        )
        for c in decision.all_ranked_candidates
    ]

    return RoutingResponse(
        selected_subscriber_id=decision.selected_subscriber_id,
        selected_subscriber_name=decision.selected_subscriber_name,
        max_expected_value=decision.max_expected_value,
        is_fallback=decision.is_fallback,
        propensity_score=decision.propensity_score,  # <-- Passed directly to response
        policy_version="epsilon_greedy_v1",
        ranked_candidates=ranked_responses,
    )
