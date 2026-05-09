from __future__ import annotations

import numpy as np
import pandas as pd


BASE_OUTCOMES = np.array([0, 1, 2, 3, 4, 6])
BASE_PROBABILITIES = np.array([0.32, 0.32, 0.10, 0.02, 0.16, 0.08])


def derive_ball_outcome_distribution(deliveries: pd.DataFrame) -> pd.DataFrame:
    frame = deliveries.copy()
    runs = frame["batsman_runs"].clip(lower=0)
    wicket = frame.get("player_dismissed").notna().astype(int)
    outcome = pd.DataFrame({"runs": runs, "wicket": wicket})
    outcome_counts = outcome.groupby("runs").size().reset_index(name="count")
    total = outcome_counts["count"].sum() if len(outcome_counts) else 1
    outcome_counts["probability"] = outcome_counts["count"] / total
    return outcome_counts


def _adjust_probabilities(required_run_rate: float, current_run_rate: float, wickets_remaining: int) -> np.ndarray:
    pressure = required_run_rate / max(current_run_rate, 0.5)
    wicket_factor = max(0.0, 1.0 - wickets_remaining / 10.0)
    adjusted = BASE_PROBABILITIES.copy()
    adjusted[0] += 0.06 * pressure + 0.05 * wicket_factor
    adjusted[1] += 0.02 * pressure
    adjusted[4] -= 0.02 * pressure
    adjusted[5] -= 0.01 * pressure
    adjusted = np.clip(adjusted, 0.01, None)
    adjusted = adjusted / adjusted.sum()
    return adjusted


def simulate_match_scenario(current_score: int, target: int, balls_remaining: int, wickets_remaining: int, current_run_rate: float, required_run_rate: float, simulations: int = 500, random_state: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    wins = 0
    final_scores = []

    for _ in range(simulations):
        score = current_score
        wickets = wickets_remaining
        remaining_balls = balls_remaining
        while remaining_balls > 0 and wickets > 0 and score < target:
            probabilities = _adjust_probabilities(required_run_rate, current_run_rate, wickets)
            outcome = rng.choice(BASE_OUTCOMES, p=probabilities)
            score += int(outcome)
            wicket_chance = min(0.03 + (required_run_rate / 20.0), 0.18)
            if rng.random() < wicket_chance:
                wickets -= 1
            remaining_balls -= 1
        final_scores.append(score)
        if score >= target:
            wins += 1

    final_scores_array = np.array(final_scores)
    return {
        "win_probability": wins / simulations,
        "expected_final_score": float(final_scores_array.mean()),
        "score_p10": float(np.percentile(final_scores_array, 10)),
        "score_p50": float(np.percentile(final_scores_array, 50)),
        "score_p90": float(np.percentile(final_scores_array, 90)),
    }
