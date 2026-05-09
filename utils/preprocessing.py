from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd


TEAM_NAMES = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

VENUES = [
    "Wankhede Stadium",
    "M. Chinnaswamy Stadium",
    "Eden Gardens",
    "MA Chidambaram Stadium",
    "Narendra Modi Stadium",
]

BATSMEN = ["Rohit Sharma", "Virat Kohli", "Suryakumar Yadav", "MS Dhoni", "Jos Buttler", "Shubman Gill", "KL Rahul", "Ravindra Jadeja"]
BOWLERS = ["Jasprit Bumrah", "Mohammed Siraj", "Rashid Khan", "Kuldeep Yadav", "Yuzvendra Chahal", "Arshdeep Singh", "Ravichandran Ashwin", "Varun Chakravarthy"]

SCHEMA_ALIASES = {
    "matchid": "match_id",
    "match_id": "match_id",
    "inning": "inning",
    "innings": "inning",
    "battingteam": "batting_team",
    "batting_team": "batting_team",
    "bowlingteam": "bowling_team",
    "bowling_team": "bowling_team",
    "batter": "batsman",
    "striker": "batsman",
    "batsman": "batsman",
    "nonstriker": "non_striker",
    "non_striker": "non_striker",
    "is_wicket": "is_wicket",
    "playerdismissed": "player_dismissed",
    "player_dismissed": "player_dismissed",
    "dismissalkind": "dismissal_kind",
    "dismissal_kind": "dismissal_kind",
    "wide": "wide_runs",
    "wide_runs": "wide_runs",
    "noball": "noball_runs",
    "noball_runs": "noball_runs",
    "bye": "bye_runs",
    "bye_runs": "bye_runs",
    "legbye": "legbye_runs",
    "legbye_runs": "legbye_runs",
    "batsmanruns": "batsman_runs",
    "batsman_runs": "batsman_runs",
    "extraruns": "extra_runs",
    "extra_runs": "extra_runs",
    "totalruns": "total_runs",
    "total_runs": "total_runs",
    "team1": "team1",
    "team2": "team2",
    "winner": "winner",
    "venue": "venue",
    "result": "result",
    "target": "target",
    "date": "date",
    "over": "over",
    "ball": "ball",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    cleaned_columns = []
    for column in renamed.columns:
        canonical = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        cleaned_columns.append(SCHEMA_ALIASES.get(canonical, canonical))
    renamed.columns = cleaned_columns
    return renamed


def load_tabular_data(source: Any) -> pd.DataFrame:
    if source is None:
        raise ValueError("No data source provided.")
    if isinstance(source, pd.DataFrame):
        return normalize_columns(source)
    if hasattr(source, "read"):
        name = str(getattr(source, "name", "")).lower()
        if name.endswith(".parquet"):
            return normalize_columns(pd.read_parquet(source))
        return normalize_columns(pd.read_csv(source))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return normalize_columns(pd.read_parquet(path))
    return normalize_columns(pd.read_csv(path))


def load_ipl_data(matches_source: Any = None, deliveries_source: Any = None) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    matches = load_tabular_data(matches_source) if matches_source is not None else None
    deliveries = load_tabular_data(deliveries_source)
    if matches is not None:
        matches = standardize_match_schema(matches)
    deliveries = standardize_delivery_schema(deliveries)
    return matches, deliveries


def standardize_match_schema(matches: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(matches)
    if "id" not in frame.columns and "match_id" in frame.columns:
        frame = frame.rename(columns={"match_id": "id"})
    if "target" not in frame.columns and "first_innings_total" in frame.columns:
        frame["target"] = frame["first_innings_total"] + 1
    return frame


def standardize_delivery_schema(deliveries: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(deliveries)
    rename_map = {}
    if "match_id" not in frame.columns and "id" in frame.columns:
        rename_map["id"] = "match_id"
    if "inning" not in frame.columns and "innings" in frame.columns:
        rename_map["innings"] = "inning"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "over" not in frame.columns and "over_number" in frame.columns:
        frame = frame.rename(columns={"over_number": "over"})
    if "ball" not in frame.columns and "ball_number" in frame.columns:
        frame = frame.rename(columns={"ball_number": "ball"})
    if "batsman" not in frame.columns and "batter" in frame.columns:
        frame = frame.rename(columns={"batter": "batsman"})
    for column in ["wide_runs", "noball_runs", "bye_runs", "legbye_runs", "batsman_runs", "extra_runs", "total_runs"]:
        if column not in frame.columns:
            frame[column] = 0
    if "is_wicket" not in frame.columns:
        frame["is_wicket"] = frame["player_dismissed"].notna().astype(int) if "player_dismissed" in frame.columns else 0
    if "player_dismissed" not in frame.columns:
        frame["player_dismissed"] = frame["batsman"].where(frame["is_wicket"].astype(int) == 1, other=pd.NA) if "batsman" in frame.columns else pd.NA
    return frame


def _sample_batsman(rng: np.random.Generator) -> str:
    return rng.choice(BATSMEN).item()


def _sample_bowler(rng: np.random.Generator) -> str:
    return rng.choice(BOWLERS).item()


def _generate_ball_outcome(rng: np.random.Generator, phase: str, pressure: float) -> tuple[int, int]:
    if phase == "powerplay":
        probs = np.array([0.30, 0.27, 0.14, 0.03, 0.15, 0.07, 0.04])
    elif phase == "death":
        probs = np.array([0.26, 0.18, 0.09, 0.02, 0.21, 0.14, 0.10])
    else:
        probs = np.array([0.35, 0.30, 0.13, 0.02, 0.12, 0.05, 0.03])

    probs = probs * np.array([1.0 + 0.05 * pressure, 1.0, 1.0, 1.0, 1.0 - 0.04 * pressure, 1.0 - 0.03 * pressure, 1.0 + 0.05 * pressure])
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()
    outcome = rng.choice([0, 1, 2, 3, 4, 6, -1], p=probs)
    if outcome == -1:
        return 0, 1
    return int(outcome), 0


def build_demo_data(random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    match_rows: list[dict[str, Any]] = []
    delivery_rows: list[dict[str, Any]] = []

    for match_id in range(1, 6):
        team1, team2 = rng.choice(TEAM_NAMES, size=2, replace=False)
        venue = rng.choice(VENUES).item()
        date = pd.Timestamp("2024-04-01") + pd.Timedelta(days=match_id)
        first_innings_total = 0
        second_innings_total = 0
        innings_wickets = {1: 0, 2: 0}

        for inning in (1, 2):
            batting_team = team1 if inning == 1 else team2
            bowling_team = team2 if inning == 1 else team1
            target = first_innings_total + 1 if inning == 2 else None
            current_score = 0
            wickets = 0
            legal_balls = 0
            batsman = _sample_batsman(rng)
            non_striker = _sample_batsman(rng)

            for ball_index in range(1, 121):
                over = (ball_index - 1) // 6
                ball_in_over = ((ball_index - 1) % 6) + 1
                phase = "powerplay" if over < 6 else "death" if over >= 16 else "middle"
                pressure = 0.0
                if inning == 2 and target is not None:
                    remaining = max(target - current_score, 0)
                    balls_left = max(120 - legal_balls, 1)
                    pressure = remaining / max(balls_left, 1)
                runs, wicket = _generate_ball_outcome(rng, phase, pressure)
                legal_balls += 1
                current_score += runs
                wickets += wicket

                delivery_rows.append(
                    {
                        "match_id": match_id,
                        "inning": inning,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over,
                        "ball": ball_in_over,
                        "batsman": batsman,
                        "non_striker": non_striker,
                        "bowler": _sample_bowler(rng),
                        "wide_runs": 0,
                        "noball_runs": 0,
                        "bye_runs": 0,
                        "legbye_runs": 0,
                        "batsman_runs": runs,
                        "extra_runs": 0,
                        "total_runs": runs,
                        "player_dismissed": batsman if wicket else None,
                        "dismissal_kind": "bowled" if wicket else None,
                    }
                )

                if wicket:
                    batsman = _sample_batsman(rng)
                if inning == 2 and target is not None and current_score >= target:
                    break
                if wickets >= 10:
                    break

            innings_wickets[inning] = wickets
            if inning == 1:
                first_innings_total = current_score
            else:
                second_innings_total = current_score

        if second_innings_total >= first_innings_total + 1:
            winner = team2
            result = f"{team2} won by {max(10 - innings_wickets[2], 1)} wickets"
        else:
            winner = team1
            result = f"{team1} won by {max(first_innings_total - second_innings_total, 1)} runs"

        match_rows.append(
            {
                "id": match_id,
                "date": date,
                "venue": venue,
                "team1": team1,
                "team2": team2,
                "winner": winner,
                "result": result,
                "target": first_innings_total + 1,
                "first_innings_total": first_innings_total,
                "second_innings_total": second_innings_total,
            }
        )

    matches = pd.DataFrame(match_rows)
    deliveries = pd.DataFrame(delivery_rows)
    return matches, deliveries


def safe_read_frame(source: Any, default: pd.DataFrame | None = None) -> pd.DataFrame | None:
    if source is None:
        return default
    if isinstance(source, pd.DataFrame):
        return normalize_columns(source)
    try:
        frame = load_tabular_data(source)
        if {"team1", "team2", "winner"}.intersection(frame.columns):
            return standardize_match_schema(frame)
        return standardize_delivery_schema(frame)
    except Exception:
        return default
