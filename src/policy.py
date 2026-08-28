import random
from dataclasses import dataclass, field

import mlflow
import numpy as np


@dataclass
class RoutingCandidate:
    subscriber_id: str
    subscriber_name: str
    commission_rate: float  # e.g., 0.10 for 10% commission
    booking_rate: float  # Monetizable value per booking (e.g., ADR or basket size in $)
    p_conversion: float  # Model predicted conversion probability P(conversion | u, s)


@dataclass
class RoutingDecision:
    selected_subscriber_id: str
    selected_subscriber_name: str
    max_expected_value: float
    propensity_score: float  # pi(a|x) logged for Inverse Propensity Scoring (IPS)
    is_exploration: bool = False
    is_fallback: bool = False
    all_ranked_candidates: list[dict] = field(default_factory=list)


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
    epsilon: float = 0.10,
    min_ev_threshold: float = 0.05,
    fallback_subscriber_id: str = "SUB_FALLBACK_DEFAULT",
    log_to_mlflow: bool = False,
) -> RoutingDecision:
    """Ranks candidates by EV, applies Epsilon-Greedy exploration, computes propensity pi(a|x),

    and optionally logs selection metrics to MLflow.
    """
    if not candidates:
        fallback_decision = RoutingDecision(
            selected_subscriber_id=fallback_subscriber_id,
            selected_subscriber_name="Default Fallback Partner",
            max_expected_value=0.0,
            propensity_score=1.0,
            is_exploration=False,
            is_fallback=True,
            all_ranked_candidates=[],
        )
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("is_fallback", 1)
        return fallback_decision

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
    best_candidate = evaluated_candidates[0]

    # Enforce minimum EV floor constraint
    if best_candidate["expected_value"] < min_ev_threshold:
        fallback_decision = RoutingDecision(
            selected_subscriber_id=fallback_subscriber_id,
            selected_subscriber_name="Default Fallback Partner",
            max_expected_value=best_candidate["expected_value"],
            propensity_score=1.0,
            is_exploration=False,
            is_fallback=True,
            all_ranked_candidates=evaluated_candidates,
        )
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("is_fallback", 1)
        return fallback_decision

    # ------------------------------------------------------------------
    # Epsilon-Greedy Policy Selection & Propensity Calculation
    # ------------------------------------------------------------------
    num_candidates = len(evaluated_candidates)
    is_exploration = False

    if random.random() < epsilon and num_candidates > 1:
        # Explore: Choose uniformly at random across all candidates
        chosen_candidate = random.choice(evaluated_candidates)
        is_exploration = True
    else:
        # Exploit: Choose candidate with top Expected Value
        chosen_candidate = best_candidate

    # Propensity Score pi(a|x) Calculation
    if chosen_candidate["subscriber_id"] == best_candidate["subscriber_id"]:
        # Top arm propensity: (1 - epsilon) + (epsilon / K)
        propensity_score = (1.0 - epsilon) + (epsilon / num_candidates)
    else:
        # Non-top arm propensity: epsilon / K
        propensity_score = epsilon / num_candidates

    decision = RoutingDecision(
        selected_subscriber_id=chosen_candidate["subscriber_id"],
        selected_subscriber_name=chosen_candidate["subscriber_name"],
        max_expected_value=chosen_candidate["expected_value"],
        propensity_score=float(np.round(propensity_score, 6)),
        is_exploration=is_exploration,
        is_fallback=False,
        all_ranked_candidates=evaluated_candidates,
    )

    # ------------------------------------------------------------------
    # MLflow Tracking
    # ------------------------------------------------------------------
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metrics(
            {
                "propensity_score": decision.propensity_score,
                "selected_expected_value": decision.max_expected_value,
                "is_exploration": int(is_exploration),
                "is_fallback": 0,
                "num_candidates": num_candidates,
            }
        )

    return decision
