"""Factor Lab — cross-sectional factor scores, distributions, and scatter analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import numpy as np
import streamlit as st
import pandas as pd
from scipy.stats import spearmanr
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
# ---------------------------------------------------------------------------

_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_af = set(st.session_state.get("active_factors", _DEFAULT_ACTIVE))
_fl_pre  = st.session_state.get("factor_lab_selected")
_snaps   = st.session_state.get("_fl_hdr_snaps", {})

if _fl_pre and _fl_pre in registry:
    _hdr_key = f"_hdr_active_{_fl_pre}"
    if _hdr_key in st.session_state and _fl_pre in _snaps:
        _cur  = st.session_state[_hdr_key]
        _snap = _snaps[_fl_pre]
        if _cur != _snap:
            _af = set(_af)
            _af.add(_fl_pre) if _cur else _af.discard(_fl_pre)
            st.session_state["active_factors"] = _af
    st.session_state[_hdr_key] = _fl_pre in _af
    _snaps[_fl_pre] = _fl_pre in _af
    st.session_state["_fl_hdr_snaps"] = _snaps

active_factors = _af

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

# Forward returns for scatter + IC
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

# Pre-compute IC (used in both metrics row and scatter tab)
_ic_val = _ic_tstat = _ic_pval = _ic_n = None
_horizon_label = {
    1: "1-day", 5: "1-week", 10: "2-week",
    21: "1-month", 42: "2-month", 63: "3-month",
}.get(fwd_horizon, f"{fwd_horizon}-day")

if not fwd_rets.empty:
    _common = scatter_scores.index.intersection(fwd_rets.index)
    if len(_common) > 10:
        _ic_val, _ic_pval = spearmanr(scatter_scores[_common], fwd_rets[_common])
        _ic_n = len(_common)
        _ic_tstat = _ic_val * np.sqrt((_ic_n - 2) / max(1 - _ic_val ** 2, 1e-9))

# ---------------------------------------------------------------------------
# Factor header — name + active toggle
# ---------------------------------------------------------------------------
_col_name, _col_toggle = st.columns([7, 1])
with _col_name:
    st.subheader(factor.label)
with _col_toggle:
    st.toggle("Active", key=f"_hdr_active_{factor_name}")

# Row 1: Category | Direction | Description
_dir_text = "Higher is better (+)" if factor.direction == 1 else "Lower is better (−)"
_mc1, _mc2, _mc3 = st.columns(3)
_mc1.markdown(f"**Category:** {factor.category}")
_mc2.markdown(f"**Direction:** {_dir_text}")
_mc3.markdown(f"**Description:** {factor.description}")

# Row 2: Formula | Reference  (only shown when populated)
if factor.formula or factor.academic_ref:
    _mf1, _mf2 = st.columns(2)
    if factor.formula:
        _mf1.markdown(f"**Formula:** `{factor.formula}`")
    if factor.academic_ref:
        _mf2.markdown(f"**Reference:** {factor.academic_ref}")

# Educational expander: plain-English interpretation
if factor.interpretation:
    with st.expander("About this factor", expanded=False):
        st.markdown(factor.interpretation)

if factor.requires_fundamentals:
    st.info("Snapshot fundamental factor — no historical panel available for IC / Backtest.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Metrics — two rows of four
# ---------------------------------------------------------------------------
_desc = scores.describe(percentiles=[0.25, 0.5, 0.75])

# Row 1: basic distribution stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Coverage",   f"{len(scores)} / {len(stock_tickers_available)} tickers")
m2.metric("Mean Score", f"{_desc['mean']:.4f}")
m3.metric("Median",     f"{_desc['50%']:.4f}")
m4.metric("Std Dev",    f"{_desc['std']:.4f}")

# Row 2: tail shape + predictive IC
m5, m6, m7, m8 = st.columns(4)
m5.metric("Min",      f"{_desc['min']:.4f}")
m6.metric("Max",      f"{_desc['max']:.4f}")
m7.metric(
    "Skewness",
    f"{scores.skew():.3f}",
    help="0 = symmetric  |  +ve = right tail (few very high scores)  |  −ve = left tail.",
)
if _ic_val is not None:
    m8.metric(
        f"Rank IC ({_horizon_label})",
        f"{_ic_val:.4f}",
        help=(
            f"Spearman rank correlation between scores and {_horizon_label} forward returns "
            f"(n={_ic_n}).  |IC| > 0.05 is considered meaningful in practice; "
            f"|IC| > 0.10 is strong.  t-stat = {_ic_tstat:.2f},  p = {_ic_pval:.3f}."
        ),
    )
else:
    top5 = scores.nlargest(5)
    m8.metric("Top Ticker", f"{top5.index[0]} ({top5.iloc[0]:.3f})")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Factor Scores", "Distribution", "Score vs. Return"])

# ── Tab 1: bar chart ────────────────────────────────────────────────────────
with tab1:
    fig_bar = plot_factor_bar(
        scores,
        ticker_sector=TICKER_SECTOR,
        title=f"{factor.label} — Cross-Section ({last_date.strftime('%Y-%m-%d')})",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 2: distribution + statistics table ──────────────────────────────────
with tab2:
    fig_dist = plot_factor_distribution(
        scores,
        ticker_sector=TICKER_SECTOR,
        title=f"{factor.label} — Distribution by Sector",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("**Score distribution statistics**")
    _iqr = _desc["75%"] - _desc["25%"]
    _stats_rows = [
        ("Count",             f"{int(_desc['count'])}"),
        ("Minimum",           f"{_desc['min']:.4f}"),
        ("25th Percentile",   f"{_desc['25%']:.4f}"),
        ("Median",            f"{_desc['50%']:.4f}"),
        ("75th Percentile",   f"{_desc['75%']:.4f}"),
        ("Maximum",           f"{_desc['max']:.4f}"),
        ("Mean",              f"{_desc['mean']:.4f}"),
        ("Std Dev",           f"{_desc['std']:.4f}"),
        ("IQR (75th − 25th)", f"{_iqr:.4f}"),
        ("Skewness",          f"{scores.skew():.4f}"),
        ("Excess Kurtosis",   f"{scores.kurt():.4f}"),
    ]
    st.dataframe(
        pd.DataFrame(_stats_rows, columns=["Statistic", "Value"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "**Skewness:** 0 = symmetric; positive = right-tailed (a few very high scores pull the mean above median); "
        "negative = left-tailed.  "
        "**Excess kurtosis:** 0 = normal distribution; positive = fatter tails than normal (more extreme outliers)."
    )

# ── Tab 3: IC statistics + scatter ──────────────────────────────────────────
with tab3:
    if fwd_rets.empty:
        st.info("Forward return data not available (insufficient future data for this horizon).")
    else:
        if _ic_val is not None:
            # IC stats row
            _ci1, _ci2, _ci3, _ci4 = st.columns(4)
            _ci1.metric(
                "Rank IC",
                f"{_ic_val:.4f}",
                help="Spearman rank correlation between factor scores and forward returns.",
            )
            _ci2.metric(
                "t-statistic",
                f"{_ic_tstat:.2f}",
                help="|t| > 2 suggests the IC is statistically distinguishable from zero at ~5% significance.",
            )
            _ci3.metric(
                "p-value",
                f"{_ic_pval:.3f}",
                help="Probability of seeing this IC by chance if the true IC were zero. < 0.05 = significant.",
            )
            _ci4.metric(
                "Pairs (N)",
                f"{_ic_n}",
                help="Number of ticker–return pairs used to compute the IC.",
            )
            _sig_str = "✓ significant at 5%" if _ic_pval < 0.05 else "✗ not significant at 5%"
            st.caption(
                f"**IC interpretation guide:** |IC| < 0.03 = negligible, 0.03–0.05 = weak, "
                f"0.05–0.10 = moderate, > 0.10 = strong.  "
                f"This factor's IC is **{_sig_str}** based on "
                f"scores at {scatter_ref_date.strftime('%Y-%m-%d')}."
            )
            st.markdown("---")

        fig_scatter = plot_factor_scatter(
            scatter_scores,
            fwd_rets,
            ticker_sector=TICKER_SECTOR,
            title=f"{factor.label} vs {_horizon_label} Forward Return "
                  f"(scores at {scatter_ref_date.strftime('%Y-%m-%d')})",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            f"Factor scores computed at {scatter_ref_date.strftime('%Y-%m-%d')} — "
            f"the last date with complete {_horizon_label} forward return data."
        )

# ---------------------------------------------------------------------------
# Top / Bottom table  (with percentile rank)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Top & Bottom 10 Stocks")
c_top, c_bot = st.columns(2)

_pct_ranks = scores.rank(pct=True)
score_df = pd.DataFrame({
    "Score":      scores,
    "Percentile": _pct_ranks.map(lambda x: f"{x:.0%}"),
    "Sector":     pd.Series({t: TICKER_SECTOR.get(t, "?") for t in scores.index}),
})

with c_top:
    st.markdown(f"**Top 10** — highest {factor.label} scores")
    st.dataframe(score_df.nlargest(10, "Score").round({"Score": 4}), use_container_width=True)

with c_bot:
    st.markdown(f"**Bottom 10** — lowest {factor.label} scores")
    st.dataframe(score_df.nsmallest(10, "Score").round({"Score": 4}), use_container_width=True)

st.caption(
    "**Percentile** = score rank within today's cross-section.  "
    "100% = highest score in the universe; 1% = lowest.  "
    "For direction = +1 factors, top-10 stocks are long candidates; "
    "for direction = −1, bottom-10 are the long candidates."
)
