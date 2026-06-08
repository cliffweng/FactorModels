"""Factor Lab — cross-sectional factor scores, distributions, and scatter analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import streamlit as st
import pandas as pd
import src.factors  # noqa: register factors

from src.data.loader import load_prices, get_fundamentals
from src.data.universe import get_universe, get_ticker_sector, get_sector_map, BENCHMARK
UNIVERSE      = st.session_state.get("filtered_universe") or get_universe()
TICKER_SECTOR = get_ticker_sector()
st.session_state["_ue_page_run"] = 0
from src.factors.base import get_registry
from src.analysis.ic import compute_forward_returns
from src.viz.factor_charts import plot_factor_bar, plot_factor_scatter, plot_factor_distribution

st.set_page_config(page_title="Factor Lab", page_icon="🔬", layout="wide")
st.title("Factor Lab")
st.caption("Compute and explore cross-sectional factor scores for the current snapshot")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
registry = get_registry()

# ---------------------------------------------------------------------------
# Header-toggle ↔ active_factors sync  (snapshot-based)
#
# The header toggle for the currently-viewed factor and the per-factor toggles
# on the Factor Library page both write to st.session_state["active_factors"].
# We keep them consistent via a per-factor snapshot:
#
#   _fl_hdr_snaps[name] = value of active_factors for that factor on our last render
#
# On each rerun:
#   • toggle differs from snapshot  → user just clicked → update active_factors
#   • toggle equals snapshot but active_factors differs → external change (Factor Library)
#                                                       → force header toggle key
#   Either way, always write the final value back to the key before rendering.
# ---------------------------------------------------------------------------

_af = set(st.session_state.get("active_factors", set(registry.keys())))
_fl_pre  = st.session_state.get("factor_lab_selected")   # set by on_change before rerun
_snaps   = st.session_state.get("_fl_hdr_snaps", {})     # {factor_name: bool}

if _fl_pre and _fl_pre in registry:
    _hdr_key = f"_hdr_active_{_fl_pre}"

    if _hdr_key in st.session_state and _fl_pre in _snaps:
        _cur  = st.session_state[_hdr_key]
        _snap = _snaps[_fl_pre]
        if _cur != _snap:
            # User clicked the header toggle → propagate to active_factors
            _af = set(_af)
            _af.add(_fl_pre) if _cur else _af.discard(_fl_pre)
            st.session_state["active_factors"] = _af
        # External change is handled below by the unconditional force.
    # Always force the key so it reflects the current (possibly externally updated) state
    st.session_state[_hdr_key] = _fl_pre in _af
    _snaps[_fl_pre] = _fl_pre in _af
    st.session_state["_fl_hdr_snaps"] = _snaps

active_factors = _af   # local alias; mutated above if user clicked

# ---------------------------------------------------------------------------
# Sidebar — factor selector (all factors, category headers as dividers)
# ---------------------------------------------------------------------------

_by_cat: dict[str, list] = {}
for _n, _f in registry.items():
    _by_cat.setdefault(_f.category, []).append((_n, _f))

_HDR = "__hdr__"
_all_opts: list[str] = []
for _cat in sorted(_by_cat.keys()):
    _all_opts.append(f"{_HDR}{_cat}")
    for _n, _f in _by_cat[_cat]:
        _all_opts.append(_n)

_first_factor: str = next(o for o in _all_opts if not o.startswith(_HDR))

if "factor_lab_selected" not in st.session_state or \
        st.session_state["factor_lab_selected"] not in registry:
    st.session_state["factor_lab_selected"] = _first_factor


def _fmt_opt(x: str) -> str:
    if x.startswith(_HDR):
        return f"── {x[len(_HDR):].upper()} ──"
    return f"  {registry[x].label}"


def _on_factor_select() -> None:
    v = st.session_state["_fl_select"]
    if v.startswith(_HDR):
        st.session_state["_fl_select"] = st.session_state["factor_lab_selected"]
    else:
        st.session_state["factor_lab_selected"] = v


_desired = st.session_state["factor_lab_selected"]
if st.session_state.get("_fl_select", _desired) != _desired:
    st.session_state["_fl_select"] = _desired

with st.sidebar:
    st.header("Factor Lab")

    st.selectbox(
        "Factor",
        _all_opts,
        index=_all_opts.index(_desired),
        format_func=_fmt_opt,
        key="_fl_select",
        on_change=_on_factor_select,
    )

    factor_name: str = st.session_state["factor_lab_selected"]
    factor = registry[factor_name]

    # Ensure the header-toggle key exists for a newly-selected factor
    if f"_hdr_active_{factor_name}" not in st.session_state:
        st.session_state[f"_hdr_active_{factor_name}"] = factor_name in active_factors

    st.markdown("---")
    st.subheader("Settings")
    winsorize      = st.checkbox("Winsorize (1%–99%)", value=True)
    standardize    = st.checkbox("Standardize (z-score)", value=False)
    lookback_years = st.slider("Price History (years)", 1, 5, 3)
    fwd_horizon    = st.select_slider(
        "Forward Return Horizon",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(x, f"{x}d"),
    )
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    _sectors = st.session_state.get("selected_sectors") or st.session_state.get("_sectors_shadow")
    if _sectors and len(_sectors) < len(list(get_sector_map())):
        st.caption(f"Universe filtered to: {', '.join(_sectors)}")
    else:
        st.caption(f"Universe: all sectors ({len(UNIVERSE)} tickers)")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date   = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers    = tuple(sorted(set(UNIVERSE + [BENCHMARK])))


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

if winsorize:
    scores = factor.winsorize(scores)
if standardize:
    scores = factor.z_score(scores)

# Forward returns for scatter tab
rets_daily       = prices_stocks.pct_change().dropna(how="all")
last_date        = prices_stocks.index[-1]
fwd_rets         = pd.Series(dtype=float)
scatter_scores   = scores
scatter_ref_date = last_date

try:
    fwd_rets_panel = compute_forward_returns(rets_daily, horizon_days=fwd_horizon)
    valid_dates = fwd_rets_panel.index[fwd_rets_panel.notna().any(axis=1)]
    if len(valid_dates) > 0:
        ref_date = valid_dates[-1]
        fwd_rets = fwd_rets_panel.loc[ref_date].dropna()
        prices_at_ref = prices_stocks.loc[:ref_date]
        try:
            scatter_scores = factor.compute(prices_at_ref, **kwargs)
            if winsorize:
                scatter_scores = factor.winsorize(scatter_scores)
            if standardize:
                scatter_scores = factor.z_score(scatter_scores)
        except Exception:
            pass
        scatter_ref_date = ref_date
except Exception:
    pass

# ---------------------------------------------------------------------------
# Factor header — name + active toggle + metadata
# ---------------------------------------------------------------------------
_col_name, _col_toggle = st.columns([7, 1])

with _col_name:
    st.subheader(factor.label)

with _col_toggle:
    # Key was already set by the sync block at the top of this script.
    st.toggle("Active", key=f"_hdr_active_{factor_name}")

_dir_text = "Higher is better (+)" if factor.direction == 1 else "Lower is better (−)"
_m1, _m2, _m3 = st.columns(3)
_m1.markdown(f"**Category:** {factor.category}")
_m2.markdown(f"**Direction:** {_dir_text}")
_m3.markdown(f"**Description:** {factor.description}")
if factor.requires_fundamentals:
    st.info("Snapshot fundamental factor — no historical panel available.")

st.markdown("---")

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
        _hlabel = {1: "1-day", 5: "1-week", 10: "2-week", 21: "1-month", 42: "2-month", 63: "3-month"}.get(fwd_horizon, f"{fwd_horizon}-day")
        fig_scatter = plot_factor_scatter(
            scatter_scores,
            fwd_rets,
            ticker_sector=TICKER_SECTOR,
            title=f"{factor.label} vs {_hlabel} Forward Return (scores at {scatter_ref_date.strftime('%Y-%m-%d')})",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            f"Factor scores computed at {scatter_ref_date.strftime('%Y-%m-%d')} — "
            f"the last date with complete {_hlabel} forward return data."
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
