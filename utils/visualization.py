from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_run_progression(frame: pd.DataFrame):
    data = frame.copy()
    if "ball_number" not in data.columns:
        data["ball_number"] = range(1, len(data) + 1)
    if "innings_runs" not in data.columns and "total_runs" in data.columns:
        data["innings_runs"] = data["total_runs"].cumsum()
    figure = px.line(data, x="ball_number", y="innings_runs", title="Run Progression", markers=True)
    figure.update_layout(template="plotly_dark", xaxis_title="Ball", yaxis_title="Runs")
    return figure


def plot_manhattan(frame: pd.DataFrame):
    data = frame.copy()
    if "ball_number" not in data.columns:
        data["ball_number"] = range(1, len(data) + 1)
    figure = px.bar(data, x="ball_number", y="total_runs", color="total_runs", title="Manhattan Chart - Runs per Ball", color_continuous_scale="YlOrRd")
    figure.update_layout(template="plotly_dark", xaxis_title="Ball", yaxis_title="Runs")
    return figure


def plot_worm(frame: pd.DataFrame):
    data = frame.copy()
    if "over" not in data.columns:
        data["over"] = data.get("ball_number", range(1, len(data) + 1)) / 6.0
    if "innings_runs" not in data.columns:
        data["innings_runs"] = data["total_runs"].cumsum()
    figure = px.line(data, x="over", y="innings_runs", title="Worm Chart - Score by Over", markers=True)
    figure.update_layout(template="plotly_dark", xaxis_title="Over", yaxis_title="Cumulative Runs")
    return figure


def plot_win_probability(probability_history: pd.DataFrame):
    data = probability_history.copy()
    figure = px.line(data, x=data.columns[0], y=data.columns[1], title="Live Win Probability", markers=True)
    figure.update_layout(template="plotly_dark", xaxis_title=data.columns[0], yaxis_title="Win Probability")
    return figure


def plot_probability_gauge(probability: float, title: str = "Win Probability"):
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#F4B400"},
                "steps": [
                    {"range": [0, 40], "color": "#5b1a1a"},
                    {"range": [40, 60], "color": "#5f4b0a"},
                    {"range": [60, 100], "color": "#0d5a3a"},
                ],
            },
        )
    )
    figure.update_layout(template="plotly_dark", height=300)
    return figure


def plot_player_comparison(summary: pd.DataFrame, players: list[str] | None = None):
    data = summary.copy()
    if players:
        data = data[data["batsman"].isin(players)]
    melted = data.melt(id_vars="batsman", value_vars=["strike_rate", "average", "boundary_percentage", "dot_ball_percentage"], var_name="metric", value_name="value")
    figure = px.bar(melted, x="batsman", y="value", color="metric", barmode="group", title="Player Comparison")
    figure.update_layout(template="plotly_dark", xaxis_title="Player", yaxis_title="Value")
    return figure


def plot_venue_heatmap(frame: pd.DataFrame):
    pivot = frame.pivot_table(index="venue", columns="batting_team", values="total_runs", aggfunc="mean", fill_value=0)
    figure = px.imshow(pivot, text_auto=True, aspect="auto", title="Venue Heatmap")
    figure.update_layout(template="plotly_dark")
    return figure


def plot_cluster_scatter(summary: pd.DataFrame):
    data = summary.copy()
    figure = px.scatter(
        data,
        x="strike_rate",
        y="average",
        color="cluster_name" if "cluster_name" in data.columns else "cluster",
        size="boundary_percentage",
        hover_name="batsman",
        title="Player Cluster Map",
    )
    figure.update_layout(template="plotly_dark", xaxis_title="Strike Rate", yaxis_title="Average")
    return figure
