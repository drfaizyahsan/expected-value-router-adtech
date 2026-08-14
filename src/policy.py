from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoutingCandidate:
    subscriber_id: str
    subscriber_name: str
    commission_rate: float  # e.g., 0.10 for 10% commission
    booking_rate: float  # Monetizable value per booking (e.g., average daily rate or basket size in $)
    p_conversion: float  # Model predicted conversion probability P(conversion | u, s)


@dataclass
class RoutingDecision:
    selected_subscriber_id: str
    selected_subscriber_name: str
    max_expected_value: float
    all_ranked_candidates: list[dict] = field(default_factory=list)
    is_fallback: bool = False


def calculate_expected_value(
    p_conversion: float,
    commission_rate: float,
    booking_rate: float,
) -> float:
    """Calculates Expected Value (EV) for an ad routing pair.

    Formula:
        EV = P(conversion | u, s) * commission_rate * booking_rate
    """
    ev = p_conversion * commission_rate * booking_rate
    return float(np.round(max(0.0, ev), 4))


def rank_and_route_candidates(
    candidates: list[RoutingCandidate],
    min_ev_threshold: float = 0.05,
    fallback_subscriber_id: str = "SUB_FALLBACK_DEFAULT",
) -> RoutingDecision:
    """Ranks candidate subscribers by Expected Value and selects the optimal target."""

    if not candidates:
        return RoutingDecision(
            selected_subscriber_id=fallback_subscriber_id,
            selected_subscriber_name="Default Fallback Partner",
            max_expected_value=0.0,
            all_ranked_candidates=[],
            is_fallback=True,
        )

    evaluated_candidates = []

    for c in candidates:
        ev = calculate_expected_value(
            p_conversion=c.p_conversion,
            commission_rate=c.commission_rate,
            booking_rate=c.booking_rate,
        )

        evaluated_candidates.append(
            {
                "subscriber_id": c.subscriber_id,
                "subscriber_name": c.subscriber_name,
                "commission_rate": c.commission_rate,
                "booking_rate": c.booking_rate,
                "p_conversion": c.p_conversion,
                "expected_value": ev,
            }
        )

    # Sort descending by Expected Value
    evaluated_candidates.sort(key=lambda x: x["expected_value"], reverse=True)
    top_candidate = evaluated_candidates[0]

    # Enforce minimum EV floor constraint
    if top_candidate["expected_value"] < min_ev_threshold:
        return RoutingDecision(
            selected_subscriber_id=fallback_subscriber_id,
            selected_subscriber_name="Default Fallback Partner",
            max_expected_value=top_candidate["expected_value"],
            all_ranked_candidates=evaluated_candidates,
            is_fallback=True,
        )

    return RoutingDecision(
        selected_subscriber_id=top_candidate["subscriber_id"],
        selected_subscriber_name=top_candidate["subscriber_name"],
        max_expected_value=top_candidate["expected_value"],
        all_ranked_candidates=evaluated_candidates,
        is_fallback=False,
    )
