from __future__ import annotations

from typing import Any, Mapping

from .feature_engineering import build_live_feature_frame


def predict_win_probability(model: Any, live_state: dict[str, Any], venue_strength: float = 0.0) -> float:
    feature_frame = build_live_feature_frame(live_state, venue_strength=venue_strength)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    return max(0.0, min(1.0, probability))


def probability_label(probability: float, team_name: str = "Batting team") -> str:
    batting = probability * 100
    bowling = 100 - batting
    return f"{team_name}: {batting:.1f}% | Opponent: {bowling:.1f}%"


def live_state_from_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return dict(inputs)
