# app/logging.py
import json
import os
from datetime import UTC, datetime
from typing import Any

LOG_FILE_PATH = os.getenv("DECISION_LOG_PATH", "logs/decision_logs.jsonl")


def log_routing_decision(
    user_context: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_subscriber_id: str,
    selected_subscriber_name: str,
    max_expected_value: float,
    propensity_score: float,
    policy_version: str = "epsilon_greedy_v1",
) -> None:
    """Appends decision context, candidates, and propensity scores to JSONL for off-policy evaluation."""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "user_context": user_context,
        "candidates": candidates,
        "selected_subscriber_id": selected_subscriber_id,
        "selected_subscriber_name": selected_subscriber_name,
        "max_expected_value": max_expected_value,
        "propensity_score": propensity_score,
        "policy_version": policy_version,
    }

    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
