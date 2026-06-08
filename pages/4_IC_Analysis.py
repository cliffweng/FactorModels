"""IC Analysis — Information Coefficient time series, rolling IC, decay curve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import streamlit as st
import pandas as pd
import src.factors  # noqa: register factors

from src.data.loader import load_prices
from src.data.universe import get_universe, BENCHMARK
UNIVERSE = st.session_state.get("filtered_universe") or get_universe()
st.session_state["_ue_page_run"] = 0
from src.factors.base import get_registry
from src.analysis.ic import (
    compute_ic_series, compute_rolling_ic, compute_ic_decay,
    compute_icir, compute_forward_returns
)
from src.viz.ic_charts import (
    plot_ic_bar, plot_ic_distribution, plot_ic_decay, plot_cumulative_ic
)

st.title("IC Analysis")
st.caption("Information Coefficient: how well does the factor rank tomorrow's winners?")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = get_registry()
_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_active = st.session_state.get("active_factors", _DEFAULT_ACTIVE)
# Only price-based, non-EDGAR factors that are active support panel / time-series IC
price_factors = {
    f.label: name for name, f in registry.items()
    if not f.requires_fundamentals and not f.requires_edgar and name in _active
}

with st.sidebar:
    st.header("IC Settings")
    if not price_factors:
        st.warning("No active price-based factors. Go to **Factor Lab** to enable some.")
        st.stop()
    selected_label = st.selectbox("Factor", list(price_factors.keys()))
    factor_name = price_factors[selected_label]
    factor = registry[factor_name]

    horizon = st.select_slider(
        "IC Horizon",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(x, f"{x}d"),
    )
    rebal_freq = st.selectbox("Rebalance Frequency", ["ME", "W-FRI", "QE"], index=0,
                               format_func=lambda x: {"ME": "Monthly", "W-FRI": "Weekly", "QE": "Quarterly"}[x])
    rolling_window = st.slider("Rolling IC Window (periods)", 3, 24, 12)
    lookback_years = st.slider("Price History (years)", 2, 5, 3)
    min_obs = st.slider("Min Cross-Section Obs", 5, 30, 10)
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    _sectors = st.session_state.get("selected_sectors") or st.session_state.get("_sectors_shadow")
    st.caption(
        f"Universe: {', '.join(_sectors)} ({len(UNIVERSE)} tickers)"
        if _sectors else f"Universe: all sectors ({len(UNIVERSE)} tickers)"
    )
    st.markdown("---")
    st.markdown(f"**Factor:** {factor.description}")

# ---------------------------------------------------------------------------
# Data loading + panel computation
# ---------------------------------------------------------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=86_400, show_spinner="Loading prices...")
def load_prices_cached(tickers, start, end, _force=False):
    return load_prices(tickers, start, end, force_refresh=_force)

@st.cache_data(ttl=3600, show_spinner="Computing factor panel...")
def compute_panel(factor_name, tickers, start, end, freq, _force=False):
    from src.factors.base import get_factor
    prices = load_prices(tickers, start, end, force_refresh=_force)
    stock_cols = [t for t in prices.columns if t != BENCHMARK]
    f = get_factor(factor_name)
    return f.compute_panel(prices[stock_cols], freq=freq), prices

with st.spinner("Computing factor panel..."):
    try:
        factor_panel, prices = compute_panel(
            factor_name, tickers, start_date, end_date, rebal_freq, _force=force_refresh
        )
    except Exception as e:
        st.error(f"Panel computation error: {e}")
        st.stop()

stock_cols = [t for t in prices.columns if t != BENCHMARK]
daily_returns = prices[stock_cols].pct_change().dropna(how="all")

# Drop all-NaN rows
factor_panel = factor_panel.dropna(how="all")

if factor_panel.empty:
    st.error("No factor panel data. Try a longer history or different factor.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute IC metrics
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Computing IC series...")
def get_ic_metrics(factor_name, rebal_freq, horizon, tickers, min_obs, rolling_window, _factor_panel, _daily_returns):
    ic = compute_ic_series(_factor_panel, _daily_returns, horizon_days=horizon, min_obs=min_obs)
    rolling = compute_rolling_ic(ic, window=rolling_window)
    return ic, rolling

ic_series, rolling_ic = get_ic_metrics(
    factor_name, rebal_freq, horizon, tickers, min_obs, rolling_window,
    factor_panel, daily_returns
)

if len(ic_series) < 3:
    st.warning("Not enough IC observations. Try a longer lookback or smaller horizon.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
icir = compute_icir(ic_series)
pct_pos = (ic_series > 0).mean()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Mean IC", f"{ic_series.mean():.4f}", help="Average Spearman rank correlation")
m2.metric("ICIR", f"{icir:.2f}", help="IC / std(IC) — higher = more consistent")
m3.metric("IC Std", f"{ic_series.std():.4f}")
m4.metric("% Positive IC", f"{pct_pos:.0%}")
m5.metric("Observations", len(ic_series))

# Signal interpretation
if abs(ic_series.mean()) < 0.02:
    st.warning("Mean IC < 0.02 — this factor shows weak predictive power over this period.")
elif abs(ic_series.mean()) < 0.05:
    st.info(f"Mean IC {ic_series.mean():.3f} — modest signal. ICIR {icir:.2f}.")
else:
    st.success(f"Strong factor signal: Mean IC {ic_series.mean():.3f}, ICIR {icir:.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["IC Time Series", "IC Distribution", "IC Decay", "Cumulative IC"])

with tab1:
    horizon_label = {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(horizon, f"{horizon}d")
    fig_ic = plot_ic_bar(
        ic_series, rolling_ic,
        title=f"{factor.label} — IC ({horizon_label} horizon, {rebal_freq} rebalance)"
    )
    st.plotly_chart(fig_ic, use_container_width=True)
    st.caption(
        "Each bar = Spearman rank correlation between factor scores and subsequent returns. "
        "Green = factor correctly ranked winners above losers. Dotted line = rolling mean IC."
    )

with tab2:
    fig_dist = plot_ic_distribution(ic_series, title=f"{factor.label} — IC Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Histogram of IC values. A good factor has a distribution shifted right of zero.")

with tab3:
    with st.spinner("Computing IC decay..."):
        @st.cache_data(ttl=3600)
        def get_ic_decay(factor_name, rebal_freq, tickers, _panel, _rets):
            return compute_ic_decay(_panel, _rets)

        decay_df = get_ic_decay(factor_name, rebal_freq, tickers, factor_panel, daily_returns)

    fig_decay = plot_ic_decay(decay_df, title=f"{factor.label} — IC Decay by Horizon")
    st.plotly_chart(fig_decay, use_container_width=True)
    st.caption(
        "IC at increasing forward horizons. Momentum factors typically peak at 1–6 months then reverse. "
        "Reversal factors peak at 1 week. Structural factors (value) may persist longer."
    )
    st.dataframe(
        decay_df.rename(columns={"IC_mean": "Mean IC", "IC_std": "Std", "ICIR": "ICIR", "t_stat": "t-stat"}).round(3),
        use_container_width=True,
    )

with tab4:
    fig_cum = plot_cumulative_ic(ic_series, title=f"{factor.label} — Cumulative IC")
    st.plotly_chart(fig_cum, use_container_width=True)
    st.caption(
        "Cumulative IC (running sum). An upward trend confirms persistent factor efficacy; "
        "flat or declining sections indicate regime changes or factor decay."
    )

# ---------------------------------------------------------------------------
# Factor panel heatmap
# ---------------------------------------------------------------------------
with st.expander("Factor Panel Heatmap (dates × tickers)"):
    import plotly.graph_objects as go
    from src.viz.theme import apply_dark
    # Show last 24 rows
    panel_display = factor_panel.tail(24)
    fig_heat = go.Figure(go.Heatmap(
        z=panel_display.values,
        x=panel_display.columns.tolist(),
        y=panel_display.index.strftime("%Y-%m-%d").tolist(),
        colorscale="RdYlGn",
        zmid=panel_display.median().median(),
        hovertemplate="%{y}<br>%{x}<br>Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Score", tickfont=dict(color="#e0e0e0")),
    ))
    apply_dark(fig_heat, title="Factor Score Panel (last 24 periods)", height=500)
    fig_heat.update_xaxes(tickangle=-90, tickfont=dict(size=9))
    st.plotly_chart(fig_heat, use_container_width=True)
