"""Factor cross-section and correlation visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.viz.theme import apply_dark, SECTOR_COLORS


def plot_factor_bar(
    scores: pd.Series,
    ticker_sector: dict[str, str] | None = None,
    title: str = "Factor Scores (Cross-Section)",
) -> go.Figure:
    """Horizontal bar chart of factor scores sorted descending, colored by sector."""
    sorted_scores = scores.sort_values(ascending=True)

    if ticker_sector:
        colors = [SECTOR_COLORS.get(ticker_sector.get(t, "Unknown"), "#7f8c8d") for t in sorted_scores.index]
    else:
        # Color by positive/negative
        colors = ["#4575b4" if v >= 0 else "#d73027" for v in sorted_scores.values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=sorted_scores.index,
            x=sorted_scores.values,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    apply_dark(fig, title=title, height=max(400, len(sorted_scores) * 14))
    fig.update_xaxes(title_text="Factor Score")
    return fig


def plot_factor_scatter(
    scores: pd.Series,
    fwd_returns: pd.Series,
    ticker_sector: dict[str, str] | None = None,
    title: str = "Factor Score vs Forward Return",
) -> go.Figure:
    """Scatter plot of factor score vs forward 1-month return."""
    df = pd.DataFrame({"score": scores, "return": fwd_returns}).dropna()
    if ticker_sector:
        df["sector"] = df.index.map(lambda t: ticker_sector.get(t, "Unknown"))
    else:
        df["sector"] = "All"

    fig = go.Figure()
    for sector in df["sector"].unique():
        mask = df["sector"] == sector
        color = SECTOR_COLORS.get(sector, "#7f8c8d")
        sub = df[mask]
        fig.add_trace(
            go.Scatter(
                x=sub["score"],
                y=sub["return"] * 100,
                mode="markers",
                name=sector,
                marker=dict(color=color, size=8, opacity=0.75),
                hovertemplate="<b>%{text}</b><br>Score: %{x:.4f}<br>Fwd Ret: %{y:.1f}%<extra></extra>",
                text=sub.index,
            )
        )

    # Add trend line
    if len(df) >= 5:
        import numpy as np
        coef = np.polyfit(df["score"], df["return"] * 100, 1)
        x_range = np.linspace(df["score"].min(), df["score"].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=np.polyval(coef, x_range),
                mode="lines",
                line=dict(color="white", width=1.5, dash="dash"),
                name="Trend",
                hoverinfo="skip",
            )
        )

    apply_dark(fig, title=title, height=400)
    fig.update_xaxes(title_text="Factor Score")
    fig.update_yaxes(title_text="Forward Return (%)")
    return fig


def plot_factor_distribution(
    scores: pd.Series,
    ticker_sector: dict[str, str] | None = None,
    title: str = "Factor Score Distribution",
) -> go.Figure:
    """Box plot of factor scores, optionally grouped by sector."""
    fig = go.Figure()
    if ticker_sector:
        df = pd.DataFrame({"score": scores, "sector": pd.Series(scores.index).map(ticker_sector).values})
        df.index = scores.index
        for sector in sorted(df["sector"].unique()):
            sub = df[df["sector"] == sector]["score"]
            color = SECTOR_COLORS.get(sector, "#7f8c8d")
            fig.add_trace(
                go.Box(
                    y=sub.values,
                    name=sector,
                    marker_color=color,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=scores.values,
                nbinsx=20,
                marker_color="#4c9be8",
                opacity=0.75,
                name="Scores",
            )
        )
    apply_dark(fig, title=title, height=380)
    return fig


def plot_correlation_matrix(
    corr: pd.DataFrame,
    title: str = "Factor Correlation Matrix",
) -> go.Figure:
    """Annotated heatmap of factor correlations."""
    labels = corr.columns.tolist()
    z = corr.values.round(2)

    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation", tickfont=dict(color="#e0e0e0")),
            hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    apply_dark(fig, title=title, height=450)
    fig.update_xaxes(side="bottom", tickangle=-30)
    return fig


def plot_universe_correlation(
    price_corr: pd.DataFrame,
    ticker_sector: dict[str, str],
    title: str = "Return Correlation — Universe",
) -> go.Figure:
    """Heatmap of stock return correlations, tickers sorted by sector."""
    # Sort by sector
    sorted_tickers = sorted(price_corr.columns, key=lambda t: ticker_sector.get(t, "ZZ"))
    sub = price_corr.loc[sorted_tickers, sorted_tickers]

    fig = go.Figure(
        go.Heatmap(
            z=sub.values.round(2),
            x=sorted_tickers,
            y=sorted_tickers,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="ρ", tickfont=dict(color="#e0e0e0")),
            hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    apply_dark(fig, title=title, height=550)
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig
