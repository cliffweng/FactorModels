"""Signal Lab — run any factor or strategy against the full basket or a single ticker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime

import numpy as np
import pandas as pd
import streamlit as st
import src.factors  # noqa: register all factors

from src.data.loader import load_prices
from src.data.universe import get_universe, get_ticker_sector, BENCHMARK
UNIVERSE      = st.session_state.get("filtered_universe") or get_universe()
TICKER_SECTOR = get_ticker_sector()
st.session_state["_ue_page_run"] = 0

from src.factors.base import get_registry
from src.factors.composite import CompositeModel

from src.analysis.ic import compute_forward_returns, compute_ic_series, compute_icir
from src.analysis.backtest import run_backtest
from src.analysis.quantile import form_quantile_portfolios
from src.analysis.stats import summary_stats
from src.analysis.strategy_store import list_strategies

from src.viz.factor_charts import plot_factor_bar
from src.viz.portfolio_charts import (
    plot_quintile_fans, plot_cumulative_ls, plot_drawdown,
)
from src.viz.signal_charts import (
    plot_score_history, plot_rank_history,
    plot_price_with_signal, plot_signal_scatter,
    plot_basket_scatter,
)
from src.viz.theme import SECTOR_COLORS

st.set_page_config(page_title="Signal Lab", page_icon="🔬", layout="wide")
st.title("Signal Lab")
st.caption(
    "Run any factor or saved strategy against the full basket or drill into a single ticker."
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
registry = get_registry()
_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_active = st.session_state.get("active_factors", _DEFAULT_ACTIVE)
price_factors = {
    name: f for name, f in registry.items()
    if not f.requires_fundamentals and name in _active
}
all_factors = {
    name: f for name, f in registry.items()
    if name in _active
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Signal Settings")

    signal_type = st.radio("Signal type", ["Factor", "Saved Strategy"], horizontal=True)

    if signal_type == "Factor":
        factor_label_map = {f.label: n for n, f in all_factors.items()}
        if not factor_label_map:
            st.warning("No active factors. Go to **Factor Lab** to enable some.")
            st.stop()
        chosen_label = st.selectbox(
            "Factor",
            options=list(factor_label_map.keys()),
        )
        chosen_factor_name = factor_label_map[chosen_label]
        chosen_factor = registry[chosen_factor_name]
        is_price_based = chosen_factor_name in price_factors
        signal_label = chosen_factor.label
        signal_direction = chosen_factor.direction
    else:
        saved = list_strategies()
        if not saved:
            st.warning("No strategies saved yet. Go to the **Multi-Factor Model** page to create one.")
            st.stop()
        strategy_names = [s["name"] for s in saved]
        chosen_strategy_name = st.selectbox("Strategy", options=strategy_names)
        chosen_strategy = next(s for s in saved if s["name"] == chosen_strategy_name)
        is_price_based = True   # strategies only contain price-based factors
        signal_label = chosen_strategy_name
        signal_direction = 1

    st.markdown("---")
    mode = st.radio("Analysis mode", ["Full Basket", "Single Ticker"], horizontal=True)

    if mode == "Single Ticker":
        chosen_ticker = st.selectbox("Ticker", sorted(UNIVERSE))

    st.markdown("---")
    rebal_freq = st.selectbox(
        "Rebalance Frequency",
        ["ME", "W-FRI", "QE"],
        format_func=lambda x: {"ME": "Monthly", "W-FRI": "Weekly", "QE": "Quarterly"}[x],
    )
    ic_horizon = st.select_slider(
        "Forward return horizon",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(x, f"{x}d"),
    )
    n_quantiles = st.slider("Quantiles", 3, 10, 5)
    lookback_years = st.slider("Price History (years)", 2, 5, 3)
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    _sectors = st.session_state.get("selected_sectors") or st.session_state.get("_sectors_shadow")
    st.caption(
        f"Universe: {', '.join(_sectors)} ({len(UNIVERSE)} tickers)"
        if _sectors else f"Universe: all sectors ({len(UNIVERSE)} tickers)"
    )

    if signal_type == "Factor" and not is_price_based:
        st.info(
            "This is a **fundamental factor** — only the current cross-section is available. "
            "Historical analysis requires a price-based factor."
        )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date   = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers    = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=86_400, show_spinner="Loading prices...")
def load_prices_cached(tickers, start, end, _force=False):
    return load_prices(tickers, start, end, force_refresh=_force)

with st.spinner("Loading price data..."):
    try:
        prices = load_prices_cached(tickers, start_date, end_date, _force=force_refresh)
    except Exception as e:
        st.error(f"Price data error: {e}")
        st.stop()

stock_cols          = [t for t in UNIVERSE if t in prices.columns]
prices_stocks       = prices[stock_cols]
daily_returns       = prices.pct_change().dropna(how="all")
daily_returns_stocks = prices_stocks.pct_change().dropna(how="all")

# ---------------------------------------------------------------------------
# Build the signal object (factor or composite) and compute panel / scores
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Computing signal panel...")
def build_panel(signal_type, chosen_name, strategy_factors_json,
                stock_cols_tuple, freq, _prices):
    """Returns (panel_df, current_scores_series, supports_panel: bool)."""
    cols = list(stock_cols_tuple)
    p = _prices[cols]

    if signal_type == "Factor":
        f = registry[chosen_name]
        if f.requires_fundamentals:
            # Snapshot only
            try:
                scores = f.compute(p)
            except Exception:
                scores = pd.Series(dtype=float)
            return pd.DataFrame(), scores, False
        else:
            try:
                panel = f.compute_panel(p, freq=freq)
            except Exception:
                panel = pd.DataFrame()
            scores = panel.iloc[-1] if not panel.empty else pd.Series(dtype=float)
            return panel, scores, True
    else:
        # Strategy → CompositeModel
        factor_names = [n for n in strategy_factors_json if n in registry]
        if not factor_names:
            return pd.DataFrame(), pd.Series(dtype=float), False
        total_w = sum(strategy_factors_json.get(n, 0) for n in factor_names)
        weights = [
            strategy_factors_json[n] / total_w if total_w > 1e-12 else 1.0 / len(factor_names)
            for n in factor_names
        ]
        m = CompositeModel([registry[n] for n in factor_names], weights)
        try:
            panel = m.compute_panel(p, freq=freq)
        except Exception:
            panel = pd.DataFrame()
        scores = m.compute_scores(p) if panel.empty else panel.iloc[-1].dropna()
        return panel, scores, not panel.empty

# Serialise strategy factors for cache key
strat_factors_json = chosen_strategy["factors"] if signal_type == "Saved Strategy" else {}

with st.spinner("Computing signal..."):
    panel, current_scores, supports_panel = build_panel(
        signal_type,
        chosen_factor_name if signal_type == "Factor" else "",
        strat_factors_json,
        tuple(stock_cols),
        rebal_freq,
        prices_stocks,
    )

if current_scores.empty:
    st.error("Could not compute signal scores. Try a different factor or longer history.")
    st.stop()

# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------
b1, b2, b3, b4 = st.columns(4)
b1.metric("Signal", signal_label)
b2.metric("Tickers scored", int(current_scores.notna().sum()))
b3.metric(
    "Top stock",
    current_scores.idxmax() if signal_direction == 1 else current_scores.idxmin(),
)
b4.metric(
    "History available",
    f"{len(panel)} periods" if supports_panel else "Snapshot only",
)
st.markdown("---")

# ===========================================================================
# FULL BASKET MODE
# ===========================================================================
if mode == "Full Basket":

    # ---- Current cross-section ----
    st.subheader("Current Cross-Section")

    fig_bar = plot_factor_bar(
        current_scores,
        ticker_sector=TICKER_SECTOR,
        title=f"{signal_label} — Scores (latest cross-section)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Top / Bottom tables
    direction_scores = current_scores * signal_direction
    top_n    = direction_scores.nlargest(10).rename("Score")
    bottom_n = direction_scores.nsmallest(10).rename("Score")

    tbl_col_l, tbl_col_r = st.columns(2)
    with tbl_col_l:
        st.markdown("**Top 10**")
        top_df = top_n.reset_index()
        top_df.columns = ["Ticker", "Score"]
        top_df["Sector"] = top_df["Ticker"].map(lambda t: TICKER_SECTOR.get(t, "—"))
        st.dataframe(top_df.set_index("Ticker"), use_container_width=True)
    with tbl_col_r:
        st.markdown("**Bottom 10**")
        bot_df = bottom_n.reset_index()
        bot_df.columns = ["Ticker", "Score"]
        bot_df["Sector"] = bot_df["Ticker"].map(lambda t: TICKER_SECTOR.get(t, "—"))
        st.dataframe(bot_df.set_index("Ticker"), use_container_width=True)

    # Score vs recent return scatter
    if not daily_returns_stocks.empty:
        ret_window = ic_horizon
        recent_ret = (
            (1 + daily_returns_stocks.tail(ret_window)).prod() - 1
        ).reindex(current_scores.index)
        st.markdown("---")
        st.subheader(f"Score vs {ret_window}-Day Return (current basket)")
        fig_scat = plot_basket_scatter(
            current_scores, recent_ret,
            TICKER_SECTOR, SECTOR_COLORS,
            title=f"{signal_label} Score vs {ret_window}d Return",
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ---- Historical performance ----
    if not supports_panel:
        st.info("Historical quintile analysis requires a price-based factor.")
    else:
        st.markdown("---")
        st.subheader("Historical Quintile Performance")

        @st.cache_data(ttl=3600, show_spinner="Running backtest...")
        def cached_backtest(direction, n_q, tickers, _panel, _daily_returns, tx_bps=10):
            bt  = run_backtest(_panel, _daily_returns, direction=direction,
                               n_quantiles=n_q, transaction_cost_bps=tx_bps)
            stock_cols_bt = [c for c in _daily_returns.columns if c != "SPY"]
            qr  = form_quantile_portfolios(
                _panel, _daily_returns[stock_cols_bt], n_q, direction
            )
            return bt, qr

        bt, qr = cached_backtest(signal_direction, n_quantiles, tickers, panel, daily_returns)

        # Quintile fan + L/S chart
        fan_col, ls_col = st.columns(2)
        with fan_col:
            fig_fan = plot_quintile_fans(
                qr.cumulative,
                title=f"{signal_label} — Quintile Fan",
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        with ls_col:
            fig_ls = plot_cumulative_ls(
                bt.cumulative_ls,
                bench_cum=bt.cumulative_benchmark,
                title=f"{signal_label} — L/S vs SPY",
            )
            st.plotly_chart(fig_ls, use_container_width=True)

        fig_dd = plot_drawdown(bt.drawdown, title=f"{signal_label} L/S Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Performance table
        st.markdown("---")
        st.subheader("Performance Statistics")
        stats_df = qr.stats_table()
        st.dataframe(stats_df, use_container_width=True)

        # IC summary
        with st.expander("IC Analysis (basket-level)"):
            @st.cache_data(ttl=3600)
            def cached_ic(horizon, tickers, _panel, _returns):
                return compute_ic_series(_panel, _returns, horizon_days=horizon)

            ic_s = cached_ic(ic_horizon, tickers, panel, daily_returns_stocks)
            if len(ic_s) >= 3:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean IC", f"{ic_s.mean():.4f}")
                m2.metric("ICIR", f"{compute_icir(ic_s):.3f}")
                m3.metric("IC Std", f"{ic_s.std():.4f}")
                m4.metric("% Positive", f"{(ic_s > 0).mean():.1%}")
            else:
                st.info("Not enough history to compute IC.")

# ===========================================================================
# SINGLE TICKER MODE
# ===========================================================================
else:
    if not supports_panel:
        st.warning(
            "Time-series analysis for a single ticker requires a price-based factor "
            "or a strategy with price-based factors."
        )
        st.stop()

    if chosen_ticker not in panel.columns:
        st.warning(f"{chosen_ticker} has no signal data in this panel. Try a longer lookback.")
        st.stop()

    ticker_sector_label = TICKER_SECTOR.get(chosen_ticker, "Unknown")
    st.subheader(f"{chosen_ticker} · {ticker_sector_label}")

    # Forward returns at rebalance frequency
    @st.cache_data(ttl=3600)
    def get_fwd_returns(_returns, horizon, _panel_index):
        return compute_forward_returns(
            _returns, horizon_days=horizon, rebal_dates=_panel_index
        )

    fwd_returns = get_fwd_returns(daily_returns_stocks, ic_horizon, panel.index)

    # ---- Score history + rank ----
    hist_col, rank_col = st.columns(2)
    with hist_col:
        fig_sh = plot_score_history(
            panel, chosen_ticker,
            title=f"{chosen_ticker} — {signal_label} Score",
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    with rank_col:
        fig_rh = plot_rank_history(
            panel, chosen_ticker,
            title=f"{chosen_ticker} — Cross-Sectional Rank",
        )
        st.plotly_chart(fig_rh, use_container_width=True)

    # ---- Price with signal overlay ----
    st.markdown("---")
    st.subheader("Price & Signal Overlay")
    st.caption(
        f"Green shading = {chosen_ticker} in top quintile that period. "
        f"Red shading = bottom quintile."
    )
    fig_pw = plot_price_with_signal(
        prices_stocks, panel, chosen_ticker,
        n_quantiles=n_quantiles,
        title=f"{chosen_ticker} — Price with {signal_label} Signal",
    )
    st.plotly_chart(fig_pw, use_container_width=True)

    # ---- Score vs forward return scatter ----
    st.markdown("---")
    st.subheader("Signal Predictiveness")
    fig_ss = plot_signal_scatter(
        panel, fwd_returns, chosen_ticker,
        title=f"{chosen_ticker} — {signal_label} Score vs {ic_horizon}d Forward Return",
    )
    st.plotly_chart(fig_ss, use_container_width=True)

    # ---- Summary stats ----
    st.markdown("---")
    st.subheader("Signal Summary")

    ticker_scores = panel[chosen_ticker].dropna()
    ticker_fwd    = fwd_returns[chosen_ticker].dropna() if chosen_ticker in fwd_returns.columns else pd.Series(dtype=float)
    both_df       = pd.concat([ticker_scores.rename("score"), ticker_fwd.rename("fwd")], axis=1).dropna()

    # Quantile membership over time
    def _qlabel(row):
        valid = row.dropna()
        if len(valid) < n_quantiles or chosen_ticker not in valid.index:
            return np.nan
        labels = pd.qcut(valid.rank(method="first"), n_quantiles,
                         labels=list(range(1, n_quantiles + 1)))
        return float(labels[chosen_ticker])

    quant_history = panel.apply(_qlabel, axis=1).dropna()
    n_periods     = len(quant_history)
    pct_top       = (quant_history == n_quantiles).mean() if n_periods else np.nan
    pct_bot       = (quant_history == 1).mean() if n_periods else np.nan

    # Hit rate: when in top quintile, did it deliver positive forward return?
    top_periods = quant_history[quant_history == n_quantiles].index
    top_fwd     = ticker_fwd.reindex(top_periods).dropna()
    hit_rate    = (top_fwd > 0).mean() if len(top_fwd) else np.nan

    # Avg return by quintile
    q_avg_ret = {}
    for q in range(1, n_quantiles + 1):
        q_dates = quant_history[quant_history == q].index
        q_fwd   = ticker_fwd.reindex(q_dates).dropna()
        if len(q_fwd):
            q_avg_ret[f"Q{q}"] = q_fwd.mean()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Periods tracked", n_periods)
    s2.metric(f"% in Top Q (Q{n_quantiles})", f"{pct_top:.1%}" if not np.isnan(pct_top) else "—")
    s3.metric("% in Bottom Q (Q1)", f"{pct_bot:.1%}" if not np.isnan(pct_bot) else "—")
    s4.metric("Hit rate (top → +ret)", f"{hit_rate:.1%}" if not np.isnan(hit_rate) else "—")

    if q_avg_ret:
        st.markdown("**Average forward return by quintile**")
        q_ret_df = pd.DataFrame.from_dict(
            q_avg_ret, orient="index", columns=["Avg Fwd Return"]
        )
        q_ret_df["Avg Fwd Return"] = q_ret_df["Avg Fwd Return"].map("{:.2%}".format)
        st.dataframe(q_ret_df, use_container_width=False)

    # Period-by-period log
    with st.expander("Period log"):
        if not both_df.empty:
            quant_col = quant_history.reindex(both_df.index)
            log_df = both_df.copy()
            log_df["Quintile"] = quant_col.map(lambda q: f"Q{int(q)}" if not np.isnan(q) else "—")
            log_df["score"] = log_df["score"].map("{:.4f}".format)
            log_df["fwd"] = log_df["fwd"].map("{:.2%}".format)
            log_df.columns = ["Score", "Fwd Return", "Quintile"]
            st.dataframe(log_df.sort_index(ascending=False), use_container_width=True)
        else:
            st.info("No period data available.")
