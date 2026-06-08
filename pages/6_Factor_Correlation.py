"""Factor Correlation — cross-factor diversification and correlation analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import src.factors  # noqa: register factors

from src.data.loader import load_prices, get_fundamentals
from src.data.universe import get_universe, get_ticker_sector, BENCHMARK
UNIVERSE      = st.session_state.get("filtered_universe") or get_universe()
TICKER_SECTOR = get_ticker_sector()
st.session_state["_ue_page_run"] = 0
from src.factors.base import get_registry
from src.viz.factor_charts import plot_correlation_matrix
from src.viz.theme import apply_dark, SECTOR_COLORS

st.set_page_config(page_title="Factor Correlation", page_icon="🔗", layout="wide")
st.title("Factor Correlation")
st.caption("How diversified is the factor library? Identify factor clusters and redundancies.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = get_registry()
_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_active = st.session_state.get("active_factors", _DEFAULT_ACTIVE)
all_factors = [
    (n, f) for n, f in registry.items()
    if not f.requires_fundamentals and not f.requires_edgar and n in _active
]
price_factors_names = [n for n, f in all_factors]

with st.sidebar:
    st.header("Settings")
    if not all_factors:
        st.warning("No active price-based factors. Go to **Factor Lab** to enable some.")
        st.stop()
    _active_labels = [f.label for _, f in all_factors]
    selected_factor_names = st.multiselect(
        "Factors to include",
        options=_active_labels,
        default=_active_labels,
    )
    lookback_years = st.slider("Price History (years)", 1, 5, 3)
    force_refresh = st.button("Refresh Data")
    st.markdown("---")
    _sectors = st.session_state.get("selected_sectors") or st.session_state.get("_sectors_shadow")
    st.caption(
        f"Universe: {', '.join(_sectors)} ({len(UNIVERSE)} tickers)"
        if _sectors else f"Universe: all sectors ({len(UNIVERSE)} tickers)"
    )
    st.markdown("---")
    st.markdown(
        """
        **Reading the Matrix**
        - ρ ≈ 0: factors measure different things → diversification
        - ρ > 0.7: high overlap → likely redundant
        - ρ < -0.5: opposing signals → may cancel out in composite
        """
    )

if not selected_factor_names:
    st.warning("Select at least 2 factors.")
    st.stop()

# Map label → registry name
label_to_name = {f.label: name for name, f in all_factors}
selected_names = [label_to_name[lbl] for lbl in selected_factor_names if lbl in label_to_name]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=86_400, show_spinner="Loading prices...")
def load_prices_cached(tickers, start, end, _force=False):
    return load_prices(tickers, start, end, force_refresh=_force)

with st.spinner("Loading data..."):
    try:
        prices = load_prices_cached(tickers, start_date, end_date, _force=force_refresh)
    except Exception as e:
        st.error(f"Price data error: {e}")
        st.stop()

stock_cols = [t for t in UNIVERSE if t in prices.columns]
prices_stocks = prices[stock_cols]

# ---------------------------------------------------------------------------
# Compute all factor cross-sections
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Computing factor scores...")
def compute_all_factors(factor_names, stock_cols, _prices):
    from src.factors.base import get_factor
    scores = {}
    for name in factor_names:
        try:
            f = get_factor(name)
            s = f.compute(_prices[stock_cols])
            if not s.empty:
                scores[f.label] = s
        except Exception:
            pass
    return pd.DataFrame(scores)

factor_scores_df = compute_all_factors(selected_names, stock_cols, prices_stocks)

if factor_scores_df.shape[1] < 2:
    st.error("Could not compute enough factors for correlation. Try loading more data.")
    st.stop()

# Correlation of cross-sectional factor scores
factor_corr = factor_scores_df.corr(method="spearman")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Factor Scatter", "Panel Correlation"])

with tab1:
    fig_corr = plot_correlation_matrix(
        factor_corr,
        title="Factor Score Correlation (Spearman, snapshot cross-section)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("High Correlations")
    pairs = []
    for i in range(len(factor_corr.columns)):
        for j in range(i + 1, len(factor_corr.columns)):
            rho = factor_corr.iloc[i, j]
            pairs.append({
                "Factor A": factor_corr.columns[i],
                "Factor B": factor_corr.columns[j],
                "Spearman ρ": round(rho, 3),
                "Overlap": "High" if abs(rho) > 0.7 else "Moderate" if abs(rho) > 0.4 else "Low",
            })

    pairs_df = pd.DataFrame(pairs).sort_values("Spearman ρ", key=abs, ascending=False)
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

with tab2:
    if factor_scores_df.shape[1] >= 2:
        col_x = st.selectbox("X-axis factor", factor_scores_df.columns.tolist(), index=0)
        col_y = st.selectbox("Y-axis factor", factor_scores_df.columns.tolist(), index=min(1, len(factor_scores_df.columns) - 1))

        scatter_df = factor_scores_df[[col_x, col_y]].dropna()
        scatter_df["sector"] = scatter_df.index.map(lambda t: TICKER_SECTOR.get(t, "Unknown"))

        fig_scatter = go.Figure()
        for sector in scatter_df["sector"].unique():
            mask = scatter_df["sector"] == sector
            sub = scatter_df[mask]
            color = SECTOR_COLORS.get(sector, "#7f8c8d")
            fig_scatter.add_trace(go.Scatter(
                x=sub[col_x],
                y=sub[col_y],
                mode="markers",
                name=sector,
                marker=dict(color=color, size=9, opacity=0.8),
                text=sub.index,
                hovertemplate="<b>%{text}</b><br>%{x:.4f}, %{y:.4f}<extra></extra>",
            ))

        # Trend
        clean = scatter_df[[col_x, col_y]].dropna()
        if len(clean) >= 5:
            coef = np.polyfit(clean[col_x], clean[col_y], 1)
            x_rng = np.linspace(clean[col_x].min(), clean[col_x].max(), 50)
            fig_scatter.add_trace(go.Scatter(
                x=x_rng, y=np.polyval(coef, x_rng),
                mode="lines", name="Trend",
                line=dict(color="white", dash="dash", width=1.5),
                hoverinfo="skip",
            ))

        apply_dark(fig_scatter, title=f"{col_x} vs {col_y}", height=420)
        fig_scatter.update_xaxes(title_text=col_x)
        fig_scatter.update_yaxes(title_text=col_y)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("Cross-factor IC correlation over time — based on rolling monthly factor panels.")

    @st.cache_data(ttl=3600, show_spinner="Computing rolling factor panels...")
    def compute_panel_corr(factor_names, _prices_key, _prices, stock_cols):
        from src.factors.base import get_factor
        panels = {}
        for name in factor_names:
            try:
                f = get_factor(name)
                if not f.requires_fundamentals:
                    panel = f.compute_panel(_prices[stock_cols])
                    if not panel.empty:
                        panels[f.label] = panel
            except Exception:
                pass
        if len(panels) < 2:
            return pd.DataFrame()

        # For each rebalance date, compute cross-section scores and correlate
        all_dates = list(panels.values())[0].index
        ic_corr_data = {}
        for dt in all_dates:
            row = {}
            for label, panel in panels.items():
                if dt in panel.index:
                    row[label] = panel.loc[dt]
            if len(row) >= 2:
                combined = pd.DataFrame(row).dropna()
                if len(combined) >= 10:
                    ic_corr_data[dt] = combined.corr(method="spearman").stack()

        if not ic_corr_data:
            return pd.DataFrame()

        # Average correlation over time
        avg = pd.DataFrame(ic_corr_data).T.mean()
        avg_matrix = avg.unstack()
        return avg_matrix

    panel_corr = compute_panel_corr(
        selected_names,
        f"panel_{lookback_years}",
        prices_stocks,
        stock_cols,
    )

    if panel_corr.empty:
        st.info("Not enough factor panel data for time-averaged correlation.")
    else:
        fig_panel_corr = plot_correlation_matrix(
            panel_corr,
            title="Time-Averaged Factor Correlation (Monthly Panel, Spearman)",
        )
        st.plotly_chart(fig_panel_corr, use_container_width=True)
        st.caption(
            "Average Spearman correlation computed across all monthly cross-sections. "
            "More stable than point-in-time snapshot correlations."
        )
