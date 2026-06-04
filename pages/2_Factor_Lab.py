"""Factor Lab — cross-sectional factor scores, distributions, and scatter analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import streamlit as st
import pandas as pd
import src.factors  # noqa: register factors

from src.data.loader import load_prices, get_fundamentals
from src.data.universe import get_universe, get_ticker_sector, BENCHMARK, get_download_tickers
UNIVERSE = get_universe()
TICKER_SECTOR = get_ticker_sector()
from src.factors.base import get_registry
from src.analysis.ic import compute_forward_returns
from src.viz.factor_charts import plot_factor_bar, plot_factor_scatter, plot_factor_distribution

st.set_page_config(page_title="Factor Lab", page_icon="🔬", layout="wide")
st.title("Factor Lab")
st.caption("Compute and explore cross-sectional factor scores for the current snapshot")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = get_registry()
factor_options = {f.label: name for name, f in registry.items()}

with st.sidebar:
    st.header("Factor Settings")
    selected_label = st.selectbox("Factor", list(factor_options.keys()))
    factor_name = factor_options[selected_label]
    factor = registry[factor_name]

    winsorize = st.checkbox("Winsorize (1%–99%)", value=True)
    standardize = st.checkbox("Standardize (z-score)", value=False)
    lookback_years = st.slider("Price History (years)", 1, 5, 3)
    fwd_horizon = st.select_slider(
        "Forward Return Horizon",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(x, f"{x}d"),
    )
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    st.markdown(f"**Category:** {factor.category}")
    st.markdown(f"**Direction:** {'Higher is better' if factor.direction == 1 else 'Lower is better'}")
    st.markdown(f"**Description:** {factor.description}")
    if factor.requires_fundamentals:
        st.info("This is a snapshot fundamental factor. No historical panel available.")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=86_400, show_spinner="Loading prices...")
def load_prices_cached(tickers, start, end, _force=False):
    return load_prices(tickers, start, end, force_refresh=_force)

@st.cache_data(ttl=86_400, show_spinner="Loading fundamentals...")
def load_fundamentals_cached(tickers, _force=False):
    return get_fundamentals(tickers, force_refresh=_force)

with st.spinner("Loading data..."):
    try:
        prices = load_prices_cached(tickers, start_date, end_date, _force=force_refresh)
    except Exception as e:
        st.error(f"Price data error: {e}")
        st.stop()

    fundamentals = None
    if factor.requires_fundamentals:
        try:
            stock_tickers = tuple(sorted(t for t in UNIVERSE if t in prices.columns))
            fundamentals = load_fundamentals_cached(stock_tickers, _force=force_refresh)
        except Exception as e:
            st.warning(f"Could not load fundamentals: {e}")

# ---------------------------------------------------------------------------
# Compute factor scores
# ---------------------------------------------------------------------------
stock_tickers_available = [t for t in UNIVERSE if t in prices.columns]
prices_stocks = prices[stock_tickers_available]

kwargs = {}
if fundamentals is not None:
    kwargs["fundamentals"] = fundamentals.reindex(stock_tickers_available)

try:
    scores = factor.compute(prices_stocks, **kwargs)
except Exception as e:
    st.error(f"Factor computation error: {e}")
    st.stop()

if scores.empty:
    st.warning("Factor returned no scores. Not enough history or data unavailable.")
    st.stop()

# Post-processing
if winsorize:
    scores = factor.winsorize(scores)
if standardize:
    scores = factor.z_score(scores)

# Compute forward returns for scatter.
# The most recent date has no future prices, so fwd_rets there are NaN.
# Find the last date where forward returns are actually available, recompute
# factor scores at that historical reference date, and pair the two.
rets_daily = prices_stocks.pct_change().dropna(how="all")
last_date = prices_stocks.index[-1]
fwd_rets = pd.Series(dtype=float)
scatter_scores = scores
scatter_ref_date = last_date

try:
    fwd_rets_panel = compute_forward_returns(rets_daily, horizon_days=fwd_horizon)
    valid_dates = fwd_rets_panel.index[fwd_rets_panel.notna().any(axis=1)]
    if len(valid_dates) > 0:
        ref_date = valid_dates[-1]
        fwd_rets = fwd_rets_panel.loc[ref_date].dropna()
        # Recompute factor scores at the reference date so they align causally
        prices_at_ref = prices_stocks.loc[:ref_date]
        try:
            scatter_scores = factor.compute(prices_at_ref, **kwargs)
            if winsorize:
                scatter_scores = factor.winsorize(scatter_scores)
            if standardize:
                scatter_scores = factor.z_score(scatter_scores)
        except Exception:
            pass  # fall back to current scores
        scatter_ref_date = ref_date
except Exception:
    pass

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Coverage", f"{len(scores)} / {len(stock_tickers_available)} tickers")
m2.metric("Mean Score", f"{scores.mean():.4f}")
m3.metric("Std Dev", f"{scores.std():.4f}")

top5 = scores.nlargest(5)
m4.metric("Top Ticker", f"{top5.index[0]} ({top5.iloc[0]:.3f})")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Factor Scores", "Distribution", "Score vs. Return"])

with tab1:
    fig_bar = plot_factor_bar(
        scores,
        ticker_sector=TICKER_SECTOR,
        title=f"{factor.label} — Cross-Section ({last_date.strftime('%Y-%m-%d')})",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    fig_dist = plot_factor_distribution(
        scores,
        ticker_sector=TICKER_SECTOR,
        title=f"{factor.label} — Distribution by Sector",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    if fwd_rets.empty:
        st.info("Forward return data not available (insufficient future data for this horizon).")
    else:
        horizon_label = {1: "1-day", 5: "1-week", 10: "2-week", 21: "1-month", 42: "2-month", 63: "3-month"}.get(fwd_horizon, f"{fwd_horizon}-day")
        fig_scatter = plot_factor_scatter(
            scatter_scores,
            fwd_rets,
            ticker_sector=TICKER_SECTOR,
            title=f"{factor.label} vs {horizon_label} Forward Return (scores at {scatter_ref_date.strftime('%Y-%m-%d')})",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            f"Factor scores computed at {scatter_ref_date.strftime('%Y-%m-%d')} — "
            f"the last date with complete {horizon_label} forward return data."
        )

# ---------------------------------------------------------------------------
# Top / Bottom table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Top & Bottom 10 Stocks")
c_top, c_bot = st.columns(2)

score_df = pd.DataFrame({
    "Score": scores,
    "Sector": pd.Series({t: TICKER_SECTOR.get(t, "?") for t in scores.index}),
})

with c_top:
    st.markdown(f"**Top 10** (highest {factor.label})")
    st.dataframe(score_df.nlargest(10, "Score").round(4), use_container_width=True)

with c_bot:
    st.markdown(f"**Bottom 10** (lowest {factor.label})")
    st.dataframe(score_df.nsmallest(10, "Score").round(4), use_container_width=True)
