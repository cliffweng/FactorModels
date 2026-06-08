"""Backtest — quintile portfolios and long-short factor strategy."""
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
from src.analysis.quantile import form_quantile_portfolios
from src.analysis.backtest import run_backtest
from src.analysis.stats import summary_stats
from src.viz.portfolio_charts import (
    plot_quintile_fans, plot_cumulative_ls, plot_drawdown, plot_annual_returns
)

st.title("Factor Backtest")
st.caption("Quintile portfolios and long-short performance across time")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
registry = get_registry()
_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_active = st.session_state.get("active_factors", _DEFAULT_ACTIVE)
price_factors = {
    f.label: name for name, f in registry.items()
    if not f.requires_fundamentals and not f.requires_edgar and name in _active
}

with st.sidebar:
    st.header("Backtest Settings")
    if not price_factors:
        st.warning("No active price-based factors. Go to **Factor Lab** to enable some.")
        st.stop()
    selected_label = st.selectbox("Factor", list(price_factors.keys()))
    factor_name = price_factors[selected_label]
    factor = registry[factor_name]

    n_quantiles = st.slider("Number of Portfolios", 3, 10, 5)
    rebal_freq = st.selectbox("Rebalance Frequency", ["ME", "W-FRI", "QE"],
                               format_func=lambda x: {"ME": "Monthly", "W-FRI": "Weekly", "QE": "Quarterly"}[x])
    tc_bps = st.slider("Transaction Cost (bps, round-trip)", 0, 50, 10)
    lookback_years = st.slider("Backtest Period (years)", 1, 5, 3)
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    _sectors = st.session_state.get("selected_sectors") or st.session_state.get("_sectors_shadow")
    st.caption(
        f"Universe: {', '.join(_sectors)} ({len(UNIVERSE)} tickers)"
        if _sectors else f"Universe: all sectors ({len(UNIVERSE)} tickers)"
    )
    st.markdown("---")
    st.caption(f"Direction: {'Higher score → Long leg' if factor.direction == 1 else 'Lower score → Long leg'}")
    st.caption(f"Description: {factor.description}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=3600, show_spinner="Running backtest...")
def run_bt(factor_name, tickers, start, end, n_quantiles, rebal_freq, tc_bps, direction, _force=False):
    from src.factors.base import get_factor
    prices = load_prices(tickers, start, end, force_refresh=_force)
    stock_cols = [t for t in prices.columns if t != BENCHMARK]
    f = get_factor(factor_name)

    panel = f.compute_panel(prices[stock_cols], freq=rebal_freq)
    daily_returns = prices.pct_change().dropna(how="all")
    result = run_backtest(panel, daily_returns, n_quantiles, direction, tc_bps)
    result.factor_name = f.label
    quant_result = form_quantile_portfolios(panel, daily_returns[stock_cols], n_quantiles, direction)
    return result, quant_result

with st.spinner("Running backtest..."):
    try:
        bt_result, quant_result = run_bt(
            factor_name, tickers, start_date, end_date,
            n_quantiles, rebal_freq, tc_bps, factor.direction,
            _force=force_refresh,
        )
    except Exception as e:
        st.error(f"Backtest error: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
stats_df = bt_result.stats()

m1, m2, m3, m4, m5 = st.columns(5)
ls_stats = stats_df.loc["L/S Strategy"] if "L/S Strategy" in stats_df.index else {}
m1.metric("L/S Ann. Return", ls_stats.get("Ann. Return", "N/A"))
m2.metric("L/S Sharpe", ls_stats.get("Sharpe", "N/A"))
m3.metric("L/S Max DD", ls_stats.get("Max DD", "N/A"))
m4.metric("L/S Calmar", ls_stats.get("Calmar", "N/A"))

# Spread between top and bottom quintile annualized
top_q = quant_result.portfolio_returns[n_quantiles]
bot_q = quant_result.portfolio_returns[1]
if len(top_q) > 0 and len(bot_q) > 0:
    spread_cum = (
        (1 + top_q.reindex(bt_result.ls_returns.index).fillna(0)).prod() /
        (1 + bot_q.reindex(bt_result.ls_returns.index).fillna(0)).prod() - 1
    )
    m5.metric("Total Q5/Q1 Spread", f"{spread_cum:.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Quintile Fans", "L/S Strategy", "Drawdown", "Annual Returns"])

with tab1:
    cum_returns = quant_result.cumulative
    # Start all at 1.0
    fig_fans = plot_quintile_fans(
        cum_returns,
        title=f"{factor.label} — {n_quantiles} Quantile Portfolios",
    )
    st.plotly_chart(fig_fans, use_container_width=True)
    st.caption(
        f"Q{n_quantiles} (blue) = highest factor scores; Q1 (red) = lowest factor scores. "
        "Clear separation confirms factor efficacy."
    )

with tab2:
    fig_ls = plot_cumulative_ls(
        bt_result.cumulative_ls,
        bt_result.cumulative_benchmark,
        title=f"{factor.label} — Long-Short Strategy",
    )
    st.plotly_chart(fig_ls, use_container_width=True)

    # Stats table
    st.markdown("**Performance Statistics**")
    st.dataframe(stats_df, use_container_width=True)

with tab3:
    fig_dd = plot_drawdown(bt_result.drawdown, title=f"{factor.label} L/S — Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

with tab4:
    annual_df = bt_result.annual_returns()
    if not annual_df.empty:
        fig_ann = plot_annual_returns(annual_df, title=f"{factor.label} — Annual Returns by Year")
        st.plotly_chart(fig_ann, use_container_width=True)
    else:
        st.info("Not enough data for annual breakdown.")

# ---------------------------------------------------------------------------
# Quintile stats table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Quantile Portfolio Statistics")
qt_stats = quant_result.stats_table()
st.dataframe(qt_stats, use_container_width=True)
