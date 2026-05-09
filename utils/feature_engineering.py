from __future__ import annotations

import numpy as np
import pandas as pd


def _sort_ball_data(deliveries: pd.DataFrame) -> pd.DataFrame:
    ordered = deliveries.copy()
    ordered = ordered.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)
    return ordered


def engineer_ball_by_ball_features(deliveries: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _sort_ball_data(deliveries)
    frame["legal_ball"] = ((frame.get("wide_runs", 0) == 0) & (frame.get("noball_runs", 0) == 0)).astype(int)
    if "is_wicket" in frame.columns:
        frame["wicket_fell"] = frame["is_wicket"].fillna(0).astype(int)
    else:
        frame["wicket_fell"] = frame.get("player_dismissed").notna().astype(int)

    grouped_innings = frame.groupby(["match_id", "inning"], sort=False)
    frame["innings_runs"] = grouped_innings["total_runs"].cumsum()
    frame["innings_wickets"] = grouped_innings["wicket_fell"].cumsum()
    frame["balls_bowled"] = grouped_innings["legal_ball"].cumsum()
    frame["balls_remaining"] = 120 - frame["balls_bowled"]
    frame["overs_bowled"] = frame["balls_bowled"] / 6.0
    frame["current_run_rate"] = np.where(frame["overs_bowled"] > 0, frame["innings_runs"] / frame["overs_bowled"], 0.0)

    if matches is not None and {"id", "target"}.issubset(matches.columns):
        match_targets = matches[["id", "target", "venue", "winner"]].rename(columns={"id": "match_id"})
        frame = frame.merge(match_targets, on="match_id", how="left", suffixes=("", "_match"))
    else:
        frame["target"] = np.nan

    frame["runs_needed"] = np.where(frame["inning"] == 2, frame["target"] - frame["innings_runs"], np.nan)
    frame["overs_remaining"] = frame["balls_remaining"] / 6.0
    # Calculate required_run_rate safely: avoid division by zero, fill small overs with conservative estimate
    frame["required_run_rate"] = np.where(
        frame["inning"] == 2,
        np.where(frame["overs_remaining"] > 0.1, frame["runs_needed"] / frame["overs_remaining"], frame["runs_needed"] * 6),
        np.nan
    )
    # Fill any remaining NaN with current_run_rate (neutral estimate)
    frame["required_run_rate"] = frame["required_run_rate"].fillna(frame["current_run_rate"])
    frame["wickets_remaining"] = 10 - frame["innings_wickets"]

    recent_runs = grouped_innings["total_runs"].rolling(window=6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    frame["recent_over_runs"] = recent_runs - frame["total_runs"]
    frame["recent_over_momentum"] = frame["recent_over_runs"] / 6.0

    wicket_segment = grouped_innings["wicket_fell"].cumsum()
    partnership_group = frame.groupby(["match_id", "inning", wicket_segment], sort=False)
    frame["partnership_runs"] = partnership_group["total_runs"].cumsum() - frame["total_runs"]
    frame["partnership_strength"] = frame["partnership_runs"].fillna(0)

    batter_group = frame.groupby(["match_id", "batsman"], sort=False)
    bowler_group = frame.groupby(["match_id", "bowler"], sort=False)
    frame["batter_balls_faced"] = batter_group["legal_ball"].cumsum() - frame["legal_ball"]
    frame["batter_runs_scored"] = batter_group["batsman_runs"].cumsum() - frame["batsman_runs"]
    frame["batter_strike_rate"] = np.where(frame["batter_balls_faced"] > 0, (frame["batter_runs_scored"] / frame["batter_balls_faced"]) * 100, 0.0)
    frame["bowler_balls"] = bowler_group["legal_ball"].cumsum() - frame["legal_ball"]
    frame["bowler_runs_conceded"] = bowler_group["total_runs"].cumsum() - frame["total_runs"]
    frame["bowler_economy"] = np.where(frame["bowler_balls"] > 0, frame["bowler_runs_conceded"] / (frame["bowler_balls"] / 6.0), 0.0)

    frame["pressure_index"] = np.where(
        frame["inning"] == 2,
        (frame["required_run_rate"].fillna(0) / (frame["current_run_rate"].replace(0, np.nan).fillna(0.5))) * (1 + (10 - frame["wickets_remaining"]) / 10.0),
        0.0,
    )

    # Additional cricket-specific features for improved prediction
    frame["balls_faced_pct"] = (frame["balls_bowled"] / 120.0 * 100).fillna(0)  # Percentage of innings played
    frame["target_gap"] = np.where(frame["inning"] == 2, (frame["target"] - frame["innings_runs"]).abs(), 0)  # Absolute runs needed
    frame["wicket_loss_rate"] = np.where(frame["overs_bowled"] > 0, frame["innings_wickets"] / frame["overs_bowled"], 0)  # Wickets per over
    frame["match_stage"] = np.where(frame["balls_bowled"] <= 36, 1, np.where(frame["balls_bowled"] <= 84, 2, 3))  # 1=powerplay, 2=middle, 3=death

    if matches is not None and "venue" in matches.columns:
        # Ensure deliveries have the venue info available by merging from matches when needed
        if "venue" not in frame.columns and "id" in matches.columns:
            venue_map = matches[["id", "venue"]].rename(columns={"id": "match_id"})
            frame = frame.merge(venue_map, on="match_id", how="left")

        venue_frame = matches.copy()
        first_innings = frame[frame["inning"] == 1][["match_id", "innings_runs"]].drop_duplicates("match_id")
        venue_frame = venue_frame.merge(first_innings, left_on="id", right_on="match_id", how="left")
        venue_strength = venue_frame.groupby("venue")["innings_runs"].mean()
        global_strength = float(venue_strength.mean()) if len(venue_strength) else 0.0
        frame["venue_advantage"] = frame["venue"].map(lambda venue: (venue_strength.get(venue, global_strength) - global_strength) / global_strength if global_strength else 0.0)
    else:
        frame["venue_advantage"] = 0.0

    frame["match_state_key"] = frame["match_id"].astype(str) + "_" + frame["inning"].astype(str)
    frame["target_reached"] = np.where((frame["inning"] == 2) & (frame["innings_runs"] >= frame["target"]), 1, 0)
    frame["win_target"] = np.where(frame["inning"] == 2, (frame["innings_runs"] >= frame["target"]).astype(int), np.nan)
    return frame


def build_training_frame(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_ball_by_ball_features(deliveries, matches)
    chase_frame = engineered[engineered["inning"] == 2].copy()
    # Safely merge match winners when available. Some datasets may omit 'winner' or use different id names.
    if isinstance(matches, pd.DataFrame) and not matches.empty:
        id_col = "id" if "id" in matches.columns else ("match_id" if "match_id" in matches.columns else None)
        if id_col is not None and "winner" in matches.columns:
            match_winners = matches[[id_col, "winner"]].rename(columns={id_col: "match_id"})
            chase_frame = chase_frame.merge(match_winners, on="match_id", how="left")
            # If winner values are missing, fall back to win_target
            if "winner" in chase_frame.columns and chase_frame["winner"].notna().any():
                chase_frame["target_win"] = (chase_frame["batting_team"] == chase_frame["winner"]).astype(int)
            else:
                chase_frame["target_win"] = chase_frame["win_target"].fillna(0).astype(int)
        else:
            chase_frame["target_win"] = chase_frame["win_target"].fillna(0).astype(int)
    else:
        chase_frame["target_win"] = chase_frame["win_target"].fillna(0).astype(int)
    return chase_frame


def build_live_feature_frame(state: dict[str, float | int | str], venue_strength: float = 0.0) -> pd.DataFrame:
    required_runs = max(float(state.get("target", 0)) - float(state.get("current_score", 0)), 0.0)
    balls_remaining = max(int(state.get("balls_remaining", 0)), 0)
    overs_remaining = balls_remaining / 6.0 if balls_remaining else 0.0
    wickets_remaining = max(10 - int(state.get("wickets_lost", 0)), 0)
    current_run_rate = float(state.get("current_score", 0)) / max((120 - balls_remaining) / 6.0, 0.1)
    required_run_rate = required_runs / max(overs_remaining, 0.1)
    pressure_index = (required_run_rate / max(current_run_rate, 0.5)) * (1 + (10 - wickets_remaining) / 10.0)

    return pd.DataFrame(
        [
            {
                "balls_remaining": balls_remaining,
                "overs_remaining": overs_remaining,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
                "wickets_remaining": wickets_remaining,
                "recent_over_momentum": float(state.get("recent_over_runs", 0)) / 6.0,
                "partnership_strength": float(state.get("partnership_runs", 0)),
                "pressure_index": pressure_index,
                "venue_advantage": venue_strength,
                "batter_strike_rate": float(state.get("batter_strike_rate", 0)),
                "bowler_economy": float(state.get("bowler_economy", 0)),
            }
        ]
    )
