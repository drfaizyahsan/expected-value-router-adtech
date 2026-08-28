from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Pydantic Schemas (API Contracts - Pydantic v2 Compliant)
# ------------------------------------------------------------------


class UserContext(BaseModel):
    user_device: str = Field(..., examples=["mobile"])
    user_osName: str = Field(..., examples=["iOS"])
    user_browserName: str = Field(..., examples=["Safari"])
    user_lat: float = Field(..., ge=-90.0, le=90.0, examples=[45.5017])
    user_lng: float = Field(..., ge=-180.0, le=180.0, examples=[-73.5673])
    dest_lat: float = Field(..., ge=-90.0, le=90.0, examples=[40.7128])
    dest_lng: float = Field(..., ge=-180.0, le=180.0, examples=[-74.0060])
    booked_flight: bool = Field(default=False)
    booked_hotel: bool = Field(default=False)
    booked_rental: bool = Field(default=False)


class CandidateSubscriber(BaseModel):
    subscriber_id: str = Field(..., examples=["SUB_101"])
    subscriber_name: str = Field(..., examples=["Expedia"])
    subscriber_tier: str = Field(default="silver", examples=["gold"])
    commission_rate: float = Field(..., ge=0.0, le=1.0, examples=[0.12])
    booking_rate: float = Field(..., ge=0.0, examples=[250.00])
    mobile_optimized: bool = Field(default=True)


class RoutingRequest(BaseModel):
    user: UserContext
    candidates: list[CandidateSubscriber] = Field(..., min_length=1)


class RankedCandidateResponse(BaseModel):
    subscriber_id: str
    subscriber_name: str
    p_conversion: float
    commission_rate: float
    booking_rate: float
    expected_value: float


class RoutingResponse(BaseModel):
    selected_subscriber_id: str
    selected_subscriber_name: str
    max_expected_value: float
    is_fallback: bool
    propensity_score: float = Field(
        ...,
        description="Propensity score pi(a|x) logged for Inverse Propensity Scoring (IPS) evaluation.",
    )
    policy_version: str = Field(
        default="epsilon_greedy_v1", examples=["epsilon_greedy_v1"]
    )
    ranked_candidates: list[RankedCandidateResponse]
