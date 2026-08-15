# scripts/benchmark_policies.py
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np

# # to run this benchmark
# pdm run python scripts/benchmark_policies.py
# # to look results in mlflow
# pdm run mlflow ui --backend-store-uri sqlite:///mlflow.db

# 1. Force tracking to sqlite:///mlflow.db
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Set/create the experiment name
mlflow.set_experiment("adtech_routing_policy_benchmark")

# Fix seeds for reproducibility
np.random.seed(42)
random.seed(42)


# ==============================================================================
# 1. Synthetic Data & Environment Setup
# ==============================================================================
def generate_test_environment(num_requests: int = 2000):
    """Generates synthetic requests and subscriber arms with hidden cold-start dynamics."""
    subscribers = [
        {
            "id": "SUB_EXPEDIA",
            "name": "Expedia",
            "p_conv": 0.12,
            "booking_rate": 200.0,
            "commission_rate": 0.10,
        },  # EV = $2.40
        {
            "id": "SUB_BOOKING",
            "name": "Booking.com",
            "p_conv": 0.08,
            "booking_rate": 250.0,
            "commission_rate": 0.15,
        },  # EV = $3.00
        {
            "id": "SUB_AGODA",
            "name": "Agoda",
            "p_conv": 0.05,
            "booking_rate": 180.0,
            "commission_rate": 0.12,
        },  # EV = $1.08
        # COLD START ARM: Initially miscalibrated in static model, but highest true EV ($6.40)
        {
            "id": "SUB_BOUTIQUE",
            "name": "Boutique Partner",
            "p_conv": 0.16,
            "booking_rate": 200.0,
            "commission_rate": 0.20,
        },
    ]

    # Calculate true EV per arm
    for s in subscribers:
        s["true_ev"] = s["p_conv"] * s["booking_rate"] * s["commission_rate"]

    return subscribers, num_requests


# ==============================================================================
# 2. Policy Definitions
# ==============================================================================
def policy_p_conversion(candidates, predicted_p_conv):
    """Policy 1: Ranks purely by predicted P(conversion|u, s)."""
    best_idx = np.argmax(predicted_p_conv)
    return candidates[best_idx]


def policy_greedy_ev(candidates, predicted_p_conv):
    """Policy 2: Ranks by Expected Value = P(conversion) * booking_rate * commission_rate."""
    evs = [
        predicted_p_conv[i] * c["booking_rate"] * c["commission_rate"]
        for i, c in enumerate(candidates)
    ]
    best_idx = np.argmax(evs)
    return candidates[best_idx]


def policy_epsilon_greedy_ev(candidates, predicted_p_conv, epsilon=0.10):
    """Policy 3: Epsilon-Greedy Expected Value (10% exploration budget)."""
    if random.random() < epsilon and len(candidates) > 1:
        # Explore: pick a random arm
        return random.choice(candidates)

    # Exploit: pick highest estimated EV
    return policy_greedy_ev(candidates, predicted_p_conv)


# ==============================================================================
# 3. Simulation Execution & MLflow Logging
# ==============================================================================
def run_simulation():
    subscribers, num_requests = generate_test_environment(num_requests=2000)

    # Set up MLflow tracking
    mlflow.set_experiment("adtech_routing_policy_benchmark")

    # Initial model predictions (simulating cold-start miscalibration for Arm 3)
    # The model initially underestimates the new Boutique Partner (p_hat = 0.02 instead of true 0.16)
    initial_p_hat = np.array([0.12, 0.08, 0.05, 0.02])

    policies = {
        "P(Conversion) Only": policy_p_conversion,
        "Greedy Expected Value": policy_greedy_ev,
        "Epsilon-Greedy EV (Bandit)": policy_epsilon_greedy_ev,
    }

    results = {}

    for policy_name, policy_fn in policies.items():
        with mlflow.start_run(run_name=policy_name):
            total_realized_revenue = 0.0
            total_conversions = 0
            cold_start_impressions = 0

            # Dynamic model estimation state
            p_hat = initial_p_hat.copy()
            arm_impressions = np.zeros(len(subscribers))
            arm_conversions = np.zeros(len(subscribers))

            for req in range(num_requests):
                # 1. Policy selection
                if policy_name == "Epsilon-Greedy EV (Bandit)":
                    chosen = policy_fn(subscribers, p_hat, epsilon=0.10)
                else:
                    chosen = policy_fn(subscribers, p_hat)

                chosen_idx = subscribers.index(chosen)

                # 2. Simulate environment outcome using true conversion probability
                converted = random.random() < chosen["p_conv"]
                payout = (
                    chosen["booking_rate"] * chosen["commission_rate"]
                    if converted
                    else 0.0
                )

                total_realized_revenue += payout
                if converted:
                    total_conversions += 1
                if chosen["id"] == "SUB_BOUTIQUE":
                    cold_start_impressions += 1

                # 3. Online feedback learning update (updating estimated p_hat as traffic arrives)
                arm_impressions[chosen_idx] += 1
                if converted:
                    arm_conversions[chosen_idx] += 1

                # Update belief state after 15 observations
                if arm_impressions[chosen_idx] >= 15:
                    p_hat[chosen_idx] = (
                        arm_conversions[chosen_idx] / arm_impressions[chosen_idx]
                    )

            yield_per_1k = (total_realized_revenue / num_requests) * 1000

            # Log parameters and metrics to MLflow
            mlflow.log_param("policy_type", policy_name)
            mlflow.log_param("num_requests", num_requests)
            mlflow.log_metric("total_realized_revenue_usd", total_realized_revenue)
            mlflow.log_metric("yield_per_1k_requests_usd", yield_per_1k)
            mlflow.log_metric("total_conversions", total_conversions)
            mlflow.log_metric("cold_start_impressions", cold_start_impressions)

            results[policy_name] = {
                "total_revenue": total_realized_revenue,
                "yield_per_1k": yield_per_1k,
                "conversions": total_conversions,
                "cold_start_impressions": cold_start_impressions,
            }

            print(
                f"✅ Executed {policy_name}: Total Revenue = ${total_realized_revenue:,.2f}"
            )

    # Generate & log primary hero visualization
    generate_hero_plot(results)


# ==============================================================================
# 4. Generate Publication-Quality Hero Bar Plot
# ==============================================================================
def generate_hero_plot(results):
    os.makedirs("docs/assets", exist_ok=True)

    policies = list(results.keys())
    revenues = [results[p]["total_revenue"] for p in policies]

    # Styling settings
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    _, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    colors = ["#94a3b8", "#3b82f6", "#10b981"]  # Slate Gray, Royal Blue, Emerald Green
    bars = ax.bar(
        policies, revenues, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2
    )

    # Add numeric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"${height:,.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
        )

    # Highlight percent improvement
    base_rev = revenues[0]
    # greedy_rev = revenues[1]
    eps_rev = revenues[2]
    uplift = ((eps_rev - base_rev) / base_rev) * 100

    ax.set_title(
        "Total Realized Revenue ($) Across 2,000 Ad Requests",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Total Realized Revenue ($ USD)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(revenues) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtitle annotation inside plot
    ax.text(
        0.5,
        0.92,
        f"Key Business Takeaway: ε-Greedy EV delivered +{uplift:.1f}% higher revenue by discovering cold-start partner value",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontstyle="italic",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#ecfdf5",
            "edgecolor": "#10b981",
            "alpha": 0.9,
        },
    )

    plt.tight_layout()
    output_path = "docs/assets/hero_policy_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n🎉 Hero Plot successfully created at: {output_path}")

    # Log artifact to MLflow
    with mlflow.start_run(run_name="Hero_Visualization_Artifact"):
        mlflow.log_artifact(output_path)


if __name__ == "__main__":
    run_simulation()
