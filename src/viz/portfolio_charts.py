"""Portfolio and backtest visualizations."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import (
    apply_dark, QUINTILE_COLORS, LS_COLOR, BENCH_COLOR, DRAWDOWN_COLOR, annotate
)


def plot_quintile_fans(
    cumulative: dict[int, pd.Series],
    title: str = "Quintile Portfolio Cumulative Returns",
) -> go.Figure:
    """5-line equity curves colored from red (Q1) to blue (Q5)."""
    n = len(cumulative)
    colors = QUINTILE_COLORS if n == 5 else [QUINTILE_COLORS[int(i * 4 / (n - 1))] for i in range(n)]

    fig = go.Figure()
    for q, cum in sorted(cumulative.items()):
        if len(cum) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=f"Q{q}",
                line=dict(color=colors[q - 1], width=2),
                hovertemplate=f"Q{q}<br>%{{x|%Y-%m-%d}}<br>Return: %{{y:.2f}}x<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line_color="rgba(255,255,255,0.2)", line_width=1, line_dash="dot")
    apply_dark(fig, title=title, height=400)
    fig.update_yaxes(title_text="Growth of $1")
    return fig


def plot_cumulative_ls(
    ls_cum: pd.Series,
    bench_cum: pd.Series | None = None,
    title: str = "Long-Short Strategy vs Benchmark",
) -> go.Figure:
    """L/S cumulative return with optional SPY overlay."""
    fig = go.Figure()

    if bench_cum is not None and len(bench_cum) > 0:
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode="lines",
                name="SPY",
                line=dict(color=BENCH_COLOR, width=1.5, dash="dot"),
                hovertemplate="SPY<br>%{x|%Y-%m-%d}<br>%{y:.2f}x<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=ls_cum.index,
            y=ls_cum.values,
            mode="lines",
            name="L/S Strategy",
            line=dict(color=LS_COLOR, width=2.5),
            hovertemplate="L/S<br>%{x|%Y-%m-%d}<br>%{y:.2f}x<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_color="rgba(255,255,255,0.2)", line_width=1, line_dash="dot")
    apply_dark(fig, title=title, height=380)
    fig.update_yaxes(title_text="Growth of $1")
    return fig


def plot_drawdown(drawdown: pd.Series, title: str = "L/S Strategy Drawdown") -> go.Figure:
    """Filled drawdown chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor=DRAWDOWN_COLOR,
            line=dict(color="#ff4b5c", width=1.5),
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1f}%<extra></extra>",
        )
    )
    apply_dark(fig, title=title, height=250)
    fig.update_yaxes(title_text="Drawdown (%)")
    return fig


def plot_annual_returns(annual_df: pd.DataFrame, title: str = "Annual Returns") -> go.Figure:
    """Grouped bar chart of annual returns by strategy vs SPY."""
    fig = go.Figure()
    colors = [LS_COLOR, BENCH_COLOR]
    for i, col in enumerate(annual_df.columns):
        fig.add_trace(
            go.Bar(
                x=annual_df.index.astype(str),
                y=(annual_df[col] * 100).round(1),
                name=col,
                marker_color=colors[i % len(colors)],
                hovertemplate=f"{col}<br>Year: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(barmode="group")
    apply_dark(fig, title=title, height=320)
    fig.update_yaxes(title_text="Return (%)")
    return fig
