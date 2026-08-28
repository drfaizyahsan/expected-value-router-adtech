from unittest.mock import patch

from src.policy import (
    RoutingCandidate,
    calculate_expected_value,
    rank_and_route_candidates,
)


def test_calculate_expected_value():
    """Tests EV calculation: 0.20 P(conv) * 0.15 comm_rate * 200.0 booking_rate = $6.00 EV."""
    ev = calculate_expected_value(
        p_conversion=0.20,
        commission_rate=0.15,
        booking_rate=200.0,
    )
    assert ev == 6.00


def test_rank_and_route_candidates_pure_exploitation():
    """Verifies candidate with higher EV is chosen under pure exploitation (epsilon=0.0)."""
    candidates = [
        RoutingCandidate(
            subscriber_id="SUB_A",
            subscriber_name="High Conv Low Commission",
            p_conversion=0.30,
            commission_rate=0.05,
            booking_rate=100.0,  # EV = 0.30 * 0.05 * 100 = $1.50
        ),
        RoutingCandidate(
            subscriber_id="SUB_B",
            subscriber_name="Lower Conv High Commission",
            p_conversion=0.10,
            commission_rate=0.20,
            booking_rate=250.0,  # EV = 0.10 * 0.20 * 250 = $5.00
        ),
    ]

    # Force epsilon=0.0 (pure exploitation)
    decision = rank_and_route_candidates(candidates, epsilon=0.0)

    assert decision.is_fallback is False
    assert decision.is_exploration is False
    assert decision.selected_subscriber_id == "SUB_B"
    assert decision.max_expected_value == 5.00
    assert decision.propensity_score == 1.0  # (1 - 0) + (0/2)
    assert len(decision.all_ranked_candidates) == 2


def test_rank_and_route_propensity_score_greedy_choice():
    """Validates top arm propensity formula: pi(a*|x) = (1 - epsilon) + (epsilon / K)."""
    candidates = [
        RoutingCandidate("SUB_1", "Partner 1", 0.10, 100.0, 0.10),  # EV = 1.0
        RoutingCandidate("SUB_2", "Partner 2", 0.20, 100.0, 0.10),  # EV = 2.0 (Top)
    ]

    # Force random.random() to return > epsilon (0.10) to trigger exploitation
    with patch("random.random", return_value=0.50):
        decision = rank_and_route_candidates(candidates, epsilon=0.10)

    assert decision.selected_subscriber_id == "SUB_2"
    assert decision.is_exploration is False
    # pi(a*|x) = (1 - 0.10) + (0.10 / 2) = 0.90 + 0.05 = 0.95
    assert decision.propensity_score == 0.95


def test_rank_and_route_propensity_score_exploration_choice():
    """Validates exploratory arm propensity formula: pi(a|x) = epsilon / K."""
    candidates = [
        RoutingCandidate("SUB_1", "Partner 1", 0.10, 100.0, 0.10),  # EV = 1.0
        RoutingCandidate("SUB_2", "Partner 2", 0.20, 100.0, 0.10),  # EV = 2.0 (Top)
    ]

    # Force random.random() < 0.10 and random.choice() to pick non-top candidate SUB_1
    with (
        patch("random.random", return_value=0.05),
        patch(
            "random.choice",
            return_value={
                "subscriber_id": "SUB_1",
                "subscriber_name": "Partner 1",
                "expected_value": 1.0,
            },
        ),
    ):
        decision = rank_and_route_candidates(candidates, epsilon=0.10)

    assert decision.selected_subscriber_id == "SUB_1"
    assert decision.is_exploration is True
    # pi(a|x) = 0.10 / 2 = 0.05
    assert decision.propensity_score == 0.05


def test_rank_and_route_candidates_below_threshold():
    """Ensures fallback route triggers when top candidate EV is below min_ev_threshold."""
    candidates = [
        RoutingCandidate(
            subscriber_id="SUB_LOW",
            subscriber_name="Low Value Target",
            p_conversion=0.01,
            commission_rate=0.05,
            booking_rate=50.0,  # EV = 0.01 * 0.05 * 50 = $0.025 < $0.05 threshold
        )
    ]

    decision = rank_and_route_candidates(candidates, min_ev_threshold=0.05)

    assert decision.is_fallback is True
    assert decision.selected_subscriber_id == "SUB_FALLBACK_DEFAULT"
    assert decision.propensity_score == 1.0


def test_rank_and_route_empty_candidates():
    """Tests edge case where no candidate list is supplied."""
    decision = rank_and_route_candidates([])
    assert decision.is_fallback is True
    assert decision.max_expected_value == 0.0
    assert decision.propensity_score == 1.0


def test_mlflow_logging_integration():
    """Verifies metrics are logged to MLflow when log_to_mlflow=True and run is active."""
    candidates = [
        RoutingCandidate("SUB_1", "Partner 1", 0.10, 100.0, 0.20),
    ]

    with (
        patch("mlflow.active_run", return_value=True),
        patch("mlflow.log_metrics") as mock_log,
    ):
        rank_and_route_candidates(candidates, log_to_mlflow=True)
        assert mock_log.called
