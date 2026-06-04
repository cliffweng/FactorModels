"""Visualizations specific to the multi-factor model page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.viz.theme import apply_dark, LS_COLOR, BENCH_COLOR, IC_POS, IC_NEG

# Palette for per-factor lines — distinct colours for up to 8 factors
FACTOR_PALETTE = [
    "#4c9be8", "#e8624c", "#56c9a4", "#e8a84c",
    "#9b59b6", "#f39c12", "#27ae60", "#e74c3c",
]


# ---------------------------------------------------------------------------
# Weight allocation charts
# ---------------------------------------------------------------------------

def plot_weight_comparison(
    results: list,   # list[OptimizeResult]
    title: str = "Factor Weights by Optimisation Method",
) -> go.Figure:
    """Grouped bar chart: x=factor names, one bar group per method."""
    if not results:
        return go.Figure()

    methods = [r.method for r in results]
    factor_names = list(results[0].weights.keys())

    fig = go.Figure()
    for i, result in enumerate(results):
        color = FACTOR_PALETTE[i % len(FACTOR_PALETTE)]
        fig.add_trace(go.Bar(
            name=result.method,
            x=factor_names,
            y=[result.weights.get(f, 0) * 100 for f in factor_names],
            marker_color=color,
            hovertemplate=f"<b>{result.method}</b><br>%{{x}}<br>Weight: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(barmode="group")
    apply_dark(fig, title=title, height=380)
    fig.update_yaxes(title_text="Weight (%)")
    fig.update_xaxes(tickangle=-20)
    return fig


def plot_weights_radar(
    weights_dict: dict[str, float],
    title: str = "Factor Weight Allocation",
) -> go.Figure:
    """Radar / spider chart of factor weights."""
    names = list(weights_dict.keys())
    values = [weights_dict[n] * 100 for n in names]
    # Close the polygon
    names_closed = names + [names[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=names_closed,
        fill="toself",
        fillcolor="rgba(76,155,232,0.2)",
        line=dict(color="#4c9be8", width=2),
        hovertemplate="%{theta}<br>%{r:.1f}%<extra></extra>",
    ))
    apply_dark(fig, title=title, height=380)
    fig.update_layout(
        polar=dict(
            bgcolor="#111520",
            radialaxis=dict(visible=True, range=[0, max(values) * 1.15],
                            gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Factor contribution stacked bar
# ---------------------------------------------------------------------------

def plot_factor_contributions(
    contrib_df: pd.DataFrame,
    title: str = "Factor Contributions to Composite Score",
) -> go.Figure:
    """Stacked bar per ticker showing each factor's weighted z-score contribution.

    Positive contributions stack above zero; negative below (barmode=relative).
    Tickers are pre-sorted ascending in contrib_df (caller's responsibility).
    """
    fig = go.Figure()
    factors = contrib_df.columns.tolist()

    for i, factor_label in enumerate(factors):
        color = FACTOR_PALETTE[i % len(FACTOR_PALETTE)]
        vals = contrib_df[factor_label].values
        fig.add_trace(go.Bar(
            name=factor_label,
            x=contrib_df.index.tolist(),
            y=vals,
            marker_color=color,
            hovertemplate=f"<b>{factor_label}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(barmode="relative")
    apply_dark(fig, title=title, height=420)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Weighted Z-Score")
    return fig


# ---------------------------------------------------------------------------
# Multi-factor IC comparison
# ---------------------------------------------------------------------------

def plot_ic_comparison(
    ic_series_dict: dict[str, pd.Series],   # {factor_label: ic_series}
    composite_ic: pd.Series | None = None,
    title: str = "IC Time Series — Individual Factors vs Composite",
) -> go.Figure:
    """Multi-line IC chart: one line per factor + heavier line for composite."""
    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)

    for i, (label, ic_s) in enumerate(ic_series_dict.items()):
        color = FACTOR_PALETTE[i % len(FACTOR_PALETTE)]
        fig.add_trace(go.Scatter(
            x=ic_s.index, y=ic_s.rolling(3, min_periods=1).mean(),
            mode="lines", name=label,
            line=dict(color=color, width=1.5),
            opacity=0.7,
            hovertemplate=f"{label}<br>%{{x|%Y-%m}}<br>IC: %{{y:.3f}}<extra></extra>",
        ))

    if composite_ic is not None and len(composite_ic) > 0:
        fig.add_trace(go.Scatter(
            x=composite_ic.index, y=composite_ic.rolling(3, min_periods=1).mean(),
            mode="lines", name="Composite",
            line=dict(color="white", width=3),
            hovertemplate="Composite<br>%{x|%Y-%m}<br>IC: %{y:.3f}<extra></extra>",
        ))

    apply_dark(fig, title=title, height=380)
    fig.update_yaxes(title_text="3-Period Rolling IC")
    return fig


def plot_ic_summary_bars(
    summary_rows: list[dict],   # [{name, IC_mean, ICIR}, ...]
    title: str = "IC Summary — Individual vs Composite",
) -> go.Figure:
    """Side-by-side bars: Mean IC and ICIR per factor + composite."""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Mean IC", "ICIR"],
                        shared_yaxes=False)

    ic_colors = [IC_POS if v >= 0 else IC_NEG for v in df["IC_mean"]]
    icir_colors = [IC_POS if v >= 0 else IC_NEG for v in df["ICIR"]]

    fig.add_trace(go.Bar(
        x=df["name"], y=df["IC_mean"],
        marker_color=ic_colors,
        name="Mean IC",
        hovertemplate="%{x}<br>Mean IC: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["name"], y=df["ICIR"],
        marker_color=icir_colors,
        name="ICIR",
        hovertemplate="%{x}<br>ICIR: %{y:.2f}<extra></extra>",
    ), row=1, col=2)

    for col in [1, 2]:
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=1, col=col)

    apply_dark(fig, title=title, height=340)
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Backtest overlay
# ---------------------------------------------------------------------------

def plot_backtest_overlay(
    cumulative_dict: dict[str, pd.Series],   # {label: cumulative_return_series}
    benchmark: pd.Series | None = None,
    highlight: str | None = None,            # label of the series to emphasise
    title: str = "Long-Short Cumulative Returns",
) -> go.Figure:
    """Multi-line cumulative return chart for comparing factor strategies."""
    fig = go.Figure()
    fig.add_hline(y=1.0, line_color="rgba(255,255,255,0.15)", line_width=1, line_dash="dot")

    if benchmark is not None and len(benchmark) > 0:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values,
            mode="lines", name="SPY",
            line=dict(color=BENCH_COLOR, width=1.5, dash="dot"),
            hovertemplate="SPY<br>%{x|%Y-%m-%d}<br>%{y:.2f}x<extra></extra>",
        ))

    for i, (label, cum) in enumerate(cumulative_dict.items()):
        if len(cum) == 0:
            continue
        is_composite = label == highlight
        color = "white" if is_composite else FACTOR_PALETTE[i % len(FACTOR_PALETTE)]
        width = 3.0 if is_composite else 1.5
        opacity = 1.0 if is_composite else 0.65
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=label,
            line=dict(color=color, width=width),
            opacity=opacity,
            hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}x<extra></extra>",
        ))

    apply_dark(fig, title=title, height=420)
    fig.update_yaxes(title_text="Growth of $1")
    return fig


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    named_points: list[dict] | None = None,  # [{method, IC_mean, IC_std, ICIR}]
    title: str = "IC Efficient Frontier",
) -> go.Figure:
    """IC_std vs IC_mean frontier coloured by ICIR, with named method markers."""
    fig = go.Figure()

    if not frontier_df.empty:
        # Frontier curve
        fig.add_trace(go.Scatter(
            x=frontier_df["IC_std"],
            y=frontier_df["IC_mean"],
            mode="lines+markers",
            marker=dict(
                color=frontier_df["ICIR"],
                colorscale="RdYlGn",
                size=6,
                colorbar=dict(title="ICIR", tickfont=dict(color="#e0e0e0"), x=1.02),
                showscale=True,
            ),
            line=dict(color="rgba(255,255,255,0.25)", width=1),
            name="Frontier",
            hovertemplate="IC Std: %{x:.4f}<br>IC Mean: %{y:.4f}<br>ICIR: %{marker.color:.2f}<extra></extra>",
        ))

    # Named method markers
    if named_points:
        marker_colors = FACTOR_PALETTE[: len(named_points)]
        symbols = ["circle", "diamond", "square", "cross", "star"]
        for j, pt in enumerate(named_points):
            fig.add_trace(go.Scatter(
                x=[pt["IC_std"]], y=[pt["IC_mean"]],
                mode="markers+text",
                marker=dict(
                    color=marker_colors[j % len(marker_colors)],
                    size=14,
                    symbol=symbols[j % len(symbols)],
                    line=dict(color="white", width=1.5),
                ),
                text=[pt["method"]],
                textposition="top center",
                textfont=dict(size=11, color="#e0e0e0"),
                name=pt["method"],
                hovertemplate=f"<b>{pt['method']}</b><br>IC Std: {pt['IC_std']:.4f}<br>IC Mean: {pt['IC_mean']:.4f}<br>ICIR: {pt['ICIR']:.2f}<extra></extra>",
            ))

    apply_dark(fig, title=title, height=440)
    fig.update_xaxes(title_text="IC Standard Deviation (risk)")
    fig.update_yaxes(title_text="Mean IC (expected predictive power)")
    return fig


# ---------------------------------------------------------------------------
# Weight sensitivity tornado chart
# ---------------------------------------------------------------------------

def plot_sensitivity(
    base_icir: float,
    sensitivities: dict[str, float],   # {factor_name: delta_icir when weight +10%}
    title: str = "ICIR Sensitivity to +10% Weight Increase",
) -> go.Figure:
    """Horizontal bar chart showing how ICIR changes when each factor's weight is nudged."""
    df = pd.Series(sensitivities).sort_values()
    colors = [IC_POS if v >= 0 else IC_NEG for v in df.values]

    fig = go.Figure(go.Bar(
        y=df.index.tolist(),
        x=df.values,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>ΔICIR: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    apply_dark(fig, title=f"{title} (base ICIR: {base_icir:.2f})", height=max(260, len(df) * 40))
    fig.update_xaxes(title_text="Change in Composite ICIR")
    return fig
