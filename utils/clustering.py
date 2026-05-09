from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_player_summary(deliveries: pd.DataFrame) -> pd.DataFrame:
    frame = deliveries.copy()
    frame["is_boundary_four"] = (frame["batsman_runs"] == 4).astype(int)
    frame["is_boundary_six"] = (frame["batsman_runs"] == 6).astype(int)
    frame["is_dot"] = (frame["total_runs"] == 0).astype(int)
    frame["is_out"] = frame.get("player_dismissed").notna().astype(int)

    summary = frame.groupby("batsman", as_index=False).agg(
        balls_faced=("total_runs", "size"),
        runs_scored=("batsman_runs", "sum"),
        fours=("is_boundary_four", "sum"),
        sixes=("is_boundary_six", "sum"),
        dots=("is_dot", "sum"),
        dismissals=("is_out", "sum"),
    )
    summary["strike_rate"] = np.where(summary["balls_faced"] > 0, (summary["runs_scored"] / summary["balls_faced"]) * 100, 0.0)
    summary["average"] = np.where(summary["dismissals"] > 0, summary["runs_scored"] / summary["dismissals"], summary["runs_scored"])
    summary["boundary_percentage"] = np.where(summary["balls_faced"] > 0, ((summary["fours"] + summary["sixes"]) / summary["balls_faced"]) * 100, 0.0)
    summary["dot_ball_percentage"] = np.where(summary["balls_faced"] > 0, (summary["dots"] / summary["balls_faced"]) * 100, 0.0)
    return summary


def choose_best_k(summary: pd.DataFrame, k_min: int = 2, k_max: int = 6, random_state: int = 42) -> int:
    numeric = summary[["strike_rate", "average", "boundary_percentage", "dot_ball_percentage"]].fillna(0)
    scaled = StandardScaler().fit_transform(numeric)
    scores: list[tuple[int, float]] = []
    upper = min(k_max, len(summary) - 1)
    if upper < k_min:
        return k_min
    for k in range(k_min, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) > 1:
            scores.append((k, silhouette_score(scaled, labels)))
    if not scores:
        return k_min
    return max(scores, key=lambda item: item[1])[0]


def label_cluster_profiles(cluster_summary: pd.DataFrame) -> pd.Series:
    labels = []
    for _, row in cluster_summary.iterrows():
        strike_rate = row.get("strike_rate", 0)
        average = row.get("average", 0)
        boundary = row.get("boundary_percentage", 0)
        dot_rate = row.get("dot_ball_percentage", 0)
        if strike_rate >= 150 and boundary >= 20:
            labels.append("Aggressive hitters")
        elif average >= 30 and strike_rate < 140:
            labels.append("Anchors")
        elif strike_rate >= 145 and average >= 20:
            labels.append("Finishers")
        elif dot_rate >= 45:
            labels.append("Powerplay specialists")
        else:
            labels.append("Balanced batters")
    return pd.Series(labels, index=cluster_summary.index)


def cluster_players(summary: pd.DataFrame, n_clusters: int | None = None, random_state: int = 42) -> tuple[pd.DataFrame, KMeans]:
    cluster_frame = summary.copy().reset_index(drop=True)
    feature_frame = cluster_frame[["strike_rate", "average", "boundary_percentage", "dot_ball_percentage"]].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)
    n_clusters = n_clusters or choose_best_k(cluster_frame, random_state=random_state)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_frame["cluster"] = model.fit_predict(scaled)
    cluster_summary = cluster_frame.groupby("cluster")[["strike_rate", "average", "boundary_percentage", "dot_ball_percentage"]].mean().reset_index()
    names = label_cluster_profiles(cluster_summary)
    cluster_name_map = {cluster: name for cluster, name in zip(cluster_summary["cluster"], names)}
    cluster_frame["cluster_name"] = cluster_frame["cluster"].map(cluster_name_map)
    return cluster_frame, model
