"""IC analysis visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import apply_dark, IC_POS, IC_NEG, annotate


def plot_ic_bar(ic_series: pd.Series, rolling_ic: pd.Series | None = None, title: str = "IC Time Series") -> go.Figure:
    """Colored IC bar chart with optional rolling mean overlay."""
    colors = [IC_POS if v >= 0 else IC_NEG for v in ic_series]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ic_series.index,
            y=ic_series.values,
            marker_color=colors,
            name="IC",
            hovertemplate="%{x|%Y-%m}<br>IC: %{y:.3f}<extra></extra>",
        )
    )

    if rolling_ic is not None and len(rolling_ic.dropna()) > 0:
        fig.add_trace(
            go.Scatter(
                x=rolling_ic.index,
                y=rolling_ic.values,
                mode="lines",
                line=dict(color="white", width=2, dash="dot"),
                name="Rolling IC (12m)",
                hovertemplate="%{x|%Y-%m}<br>Rolling IC: %{y:.3f}<extra></extra>",
            )
        )

    # Zero line
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    # Mean IC annotation
    mean_ic = ic_series.mean()
    icir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else float("nan")
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.95,
        text=f"Mean IC: {mean_ic:.3f}  |  ICIR: {icir:.2f}  |  n={len(ic_series)}",
        showarrow=False,
        font=dict(size=12, color="#e0e0e0"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4,
        align="left",
    )

    apply_dark(fig, title=title, height=350)
    fig.update_yaxes(title_text="Spearman IC")
    return fig


def plot_ic_distribution(ic_series: pd.Series, title: str = "IC Distribution") -> go.Figure:
    """Histogram of IC values with normal reference."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=ic_series.values,
            nbinsx=25,
            marker_color=IC_POS,
            opacity=0.75,
            name="IC",
            hovertemplate="IC: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )
    # Mean line
    mean_ic = ic_series.mean()
    fig.add_vline(x=mean_ic, line_color="white", line_width=2, line_dash="dash",
                  annotation_text=f"μ={mean_ic:.3f}", annotation_position="top right")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    apply_dark(fig, title=title, height=300)
    fig.update_xaxes(title_text="IC")
    fig.update_yaxes(title_text="Frequency")
    return fig


def plot_ic_decay(decay_df: pd.DataFrame, title: str = "IC Decay by Horizon") -> go.Figure:
    """Bar chart of mean IC at increasing forward horizons."""
    decay = decay_df.reset_index()
    label_map = {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m", 126: "6m"}
    decay["label"] = decay["horizon_days"].map(lambda x: label_map.get(x, f"{x}d"))

    colors = [IC_POS if v >= 0 else IC_NEG for v in decay["IC_mean"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=decay["label"],
            y=decay["IC_mean"],
            error_y=dict(type="data", array=decay["IC_std"].fillna(0), visible=True, color="rgba(255,255,255,0.4)"),
            marker_color=colors,
            name="Mean IC",
            hovertemplate="Horizon: %{x}<br>IC: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    apply_dark(fig, title=title, height=320)
    fig.update_xaxes(title_text="Forward Horizon")
    fig.update_yaxes(title_text="Mean IC")
    return fig


def plot_cumulative_ic(ic_series: pd.Series, title: str = "Cumulative IC") -> go.Figure:
    """Running sum of IC — useful for detecting regime changes."""
    cum_ic = ic_series.cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_ic.index,
            y=cum_ic.values,
            mode="lines",
            line=dict(color=IC_POS, width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 136, 0.15)",
            name="Cumulative IC",
            hovertemplate="%{x|%Y-%m}<br>Cumulative IC: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    apply_dark(fig, title=title, height=300)
    fig.update_yaxes(title_text="Cumulative IC")
    return fig
