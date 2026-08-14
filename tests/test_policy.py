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


def test_rank_and_route_candidates_selection():
    """Verifies candidate with higher EV is chosen even if raw conversion prob is lower."""
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

    decision = rank_and_route_candidates(candidates)

    assert decision.is_fallback is False
    assert decision.selected_subscriber_id == "SUB_B"
    assert decision.max_expected_value == 5.00
    assert len(decision.all_ranked_candidates) == 2


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


def test_rank_and_route_empty_candidates():
    """Tests edge case where no candidate list is supplied."""
    decision = rank_and_route_candidates([])
    assert decision.is_fallback is True
    assert decision.max_expected_value == 0.0
