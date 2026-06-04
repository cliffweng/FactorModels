"""Shared dark theme and color palettes for all Plotly charts."""
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Layout defaults
# ---------------------------------------------------------------------------

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#111520",
    font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)


def apply_dark(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply dark theme and optional title/height to a figure."""
    layout = dict(DARK_LAYOUT)
    layout["height"] = height
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color="#e0e0e0"), x=0.01)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Quintile fan: Q1 (worst) → Q5 (best)
QUINTILE_COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]

# IC positive / negative
IC_POS = "#00cc88"
IC_NEG = "#ff4b5c"

# Sector palette (11 GICS sectors)
SECTOR_COLORS = {
    "Technology":              "#4c9be8",
    "Healthcare":              "#56c9a4",
    "Financials":              "#e8a84c",
    "Consumer Discretionary":  "#e8624c",
    "Consumer Staples":        "#a4c84c",
    "Communication Services":  "#9b59b6",
    "Industrials":             "#5dade2",
    "Energy":                  "#e8784c",
    "Materials":               "#27ae60",
    "Utilities":               "#f39c12",
    "Real Estate":             "#e74c3c",
    "Unknown":                 "#7f8c8d",
}

# Long-short strategy and benchmark
LS_COLOR = "#4c9be8"
BENCH_COLOR = "#aaaaaa"
DRAWDOWN_COLOR = "rgba(255, 75, 92, 0.35)"


# ---------------------------------------------------------------------------
# Shared annotation helper
# ---------------------------------------------------------------------------

def annotate(fig: go.Figure, x, y, text: str, **kwargs) -> go.Figure:
    fig.add_annotation(
        x=x, y=y, text=text,
        showarrow=False,
        font=dict(size=11, color="#e0e0e0"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=3,
        **kwargs,
    )
    return fig
