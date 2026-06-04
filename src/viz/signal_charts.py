"""Visualizations for Signal Lab — basket scan and single-ticker analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.viz.theme import apply_dark, IC_POS, IC_NEG, BENCH_COLOR, QUINTILE_COLORS

_TOP_BG = "rgba(86, 201, 164, 0.18)"
_BOT_BG = "rgba(232, 98, 76, 0.18)"


# ---------------------------------------------------------------------------
# Score history (single ticker)
# ---------------------------------------------------------------------------

def plot_score_history(
    score_panel: pd.DataFrame,
    ticker: str,
    title: str = "",
) -> go.Figure:
    """Bar chart of a single ticker's factor score over rebalance dates."""
    if ticker not in score_panel.columns:
        return go.Figure()
    s = score_panel[ticker].dropna()

    colors = [IC_POS if v >= 0 else IC_NEG for v in s.values]

    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.add_trace(go.Bar(
        x=s.index, y=s.values,
        marker_color=colors,
        hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.4f}<extra></extra>",
        name="Score",
    ))

    apply_dark(fig, title=title or f"{ticker} — Factor Score History", height=300)
    fig.update_yaxes(title_text="Factor Score")
    return fig


# ---------------------------------------------------------------------------
# Cross-sectional percentile rank (single ticker)
# ---------------------------------------------------------------------------

def plot_rank_history(
    score_panel: pd.DataFrame,
    ticker: str,
    title: str = "",
) -> go.Figure:
    """Cross-sectional percentile rank (0–100) of one ticker over time."""
    if ticker not in score_panel.columns:
        return go.Figure()

    def _pct_rank(row):
        valid = row.dropna()
        if len(valid) < 2 or ticker not in valid.index:
            return np.nan
        return float(valid.rank(pct=True)[ticker] * 100)

    rank_s = score_panel.apply(_pct_rank, axis=1).dropna()

    fig = go.Figure()
    fig.add_hrect(y0=80, y1=100, fillcolor=_TOP_BG, line_width=0,
                  annotation_text="Top 20%", annotation_position="top left",
                  annotation_font=dict(color="#56c9a4", size=10))
    fig.add_hrect(y0=0, y1=20, fillcolor=_BOT_BG, line_width=0,
                  annotation_text="Bottom 20%", annotation_position="bottom left",
                  annotation_font=dict(color="#e8624c", size=10))
    fig.add_trace(go.Scatter(
        x=rank_s.index, y=rank_s.values,
        mode="lines+markers",
        line=dict(color="#4c9be8", width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(76,155,232,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Rank: %{y:.1f}th pct<extra></extra>",
        name="Percentile Rank",
    ))

    apply_dark(fig, title=title or f"{ticker} — Cross-Sectional Rank Percentile", height=280)
    fig.update_yaxes(title_text="Percentile Rank", range=[0, 108])
    return fig


# ---------------------------------------------------------------------------
# Price chart with quintile signal shading
# ---------------------------------------------------------------------------

def plot_price_with_signal(
    daily_prices: pd.DataFrame,
    score_panel: pd.DataFrame,
    ticker: str,
    n_quantiles: int = 5,
    title: str = "",
) -> go.Figure:
    """Price line with background bands showing top/bottom quantile membership.

    Green band = ticker in top quantile that period.
    Red band   = ticker in bottom quantile that period.
    """
    if ticker not in score_panel.columns or ticker not in daily_prices.columns:
        return go.Figure()

    price_s = daily_prices[ticker].dropna()

    # Compute quantile label for the ticker at each rebalance date
    def _qlabel(row):
        valid = row.dropna()
        if len(valid) < n_quantiles or ticker not in valid.index:
            return np.nan
        labels = pd.qcut(valid.rank(method="first"), n_quantiles,
                         labels=list(range(1, n_quantiles + 1)))
        return float(labels[ticker])

    quant_s = score_panel.apply(_qlabel, axis=1).dropna()

    fig = go.Figure()

    # Vertical bands between rebalance dates
    rebal_dates = list(quant_s.index)
    for i, rdate in enumerate(rebal_dates):
        q = quant_s.loc[rdate]
        if np.isnan(q):
            continue
        x0 = rdate
        x1 = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else price_s.index[-1]
        if q == n_quantiles:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=_TOP_BG, line_width=0,
                          annotation_text=f"Q{int(q)}" if i == 0 else "")
        elif q == 1:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=_BOT_BG, line_width=0,
                          annotation_text="Q1" if i == 0 else "")

    fig.add_trace(go.Scatter(
        x=price_s.index, y=price_s.values,
        mode="lines",
        line=dict(color=BENCH_COLOR, width=1.8),
        name="Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    t = title or f"{ticker} — Price with Signal Overlay"
    apply_dark(fig, title=t, height=360)
    fig.update_yaxes(title_text="Price ($)")
    fig.add_annotation(
        text="<b style='color:#56c9a4'>■</b> Top quintile &nbsp;&nbsp;"
             "<b style='color:#e8624c'>■</b> Bottom quintile",
        xref="paper", yref="paper", x=0.01, y=1.07,
        showarrow=False, font=dict(size=10, color="#aaaaaa"),
    )
    return fig


# ---------------------------------------------------------------------------
# Factor score vs forward return scatter (single ticker over time)
# ---------------------------------------------------------------------------

def plot_signal_scatter(
    score_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    ticker: str,
    title: str = "",
) -> go.Figure:
    """Scatter of this ticker's factor score vs its subsequent forward return."""
    if ticker not in score_panel.columns or ticker not in forward_returns.columns:
        return go.Figure()

    scores = score_panel[ticker].dropna()
    fwds = forward_returns[ticker].dropna()
    both = pd.concat([scores.rename("score"), fwds.rename("fwd")], axis=1).dropna()

    if both.empty:
        return go.Figure()

    colors = [IC_POS if v >= 0 else IC_NEG for v in both["fwd"]]

    x, y = both["score"].values, both["fwd"].values
    if len(x) >= 3:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = m * x_line + b
    else:
        x_line, y_line = [], []

    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_trace(go.Scatter(
        x=both["score"], y=both["fwd"],
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.85),
        text=both.index.strftime("%Y-%m"),
        hovertemplate="Period: %{text}<br>Score: %{x:.4f}<br>Fwd Ret: %{y:.2%}<extra></extra>",
        name=ticker,
    ))
    if len(x_line):
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
            name="Trend",
            hoverinfo="skip",
        ))

    apply_dark(fig, title=title or f"{ticker} — Score vs Forward Return", height=340)
    fig.update_xaxes(title_text="Factor Score")
    fig.update_yaxes(title_text="Forward Return", tickformat=".1%")
    return fig


# ---------------------------------------------------------------------------
# Basket: cross-section scatter (score vs realised return, all tickers)
# ---------------------------------------------------------------------------

def plot_basket_scatter(
    scores: pd.Series,
    returns: pd.Series,
    ticker_sector: dict[str, str],
    sector_colors: dict[str, str],
    title: str = "",
) -> go.Figure:
    """Scatter of current factor scores vs a return window, coloured by sector."""
    both = pd.concat([scores.rename("score"), returns.rename("ret")], axis=1).dropna()
    if both.empty:
        return go.Figure()

    # Trend line
    x, y = both["score"].values, both["ret"].values
    if len(x) >= 3:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = m * x_line + b
    else:
        x_line, y_line = [], []

    from src.viz.theme import SECTOR_COLORS
    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    # One trace per sector for legend grouping
    sectors = sorted(set(ticker_sector.get(t, "Unknown") for t in both.index))
    for sector in sectors:
        tickers = [t for t in both.index if ticker_sector.get(t, "Unknown") == sector]
        if not tickers:
            continue
        sub = both.loc[tickers]
        color = sector_colors.get(sector, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=sub["score"], y=sub["ret"],
            mode="markers+text",
            text=sub.index.tolist(),
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(color=color, size=8, opacity=0.85),
            name=sector,
            hovertemplate=f"%{{text}}<br>Score: %{{x:.4f}}<br>Return: %{{y:.2%}}<extra>{sector}</extra>",
        ))

    if len(x_line):
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
            name="Trend", hoverinfo="skip",
        ))

    apply_dark(fig, title=title or "Factor Score vs Return — Basket", height=460)
    fig.update_xaxes(title_text="Factor Score")
    fig.update_yaxes(title_text="Return", tickformat=".1%")
    return fig
