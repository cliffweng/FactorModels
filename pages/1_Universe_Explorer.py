"""Universe Explorer — sector breakdown, return correlations, YTD performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import src.factors  # noqa: register factors

from src.data.loader import load_prices, get_fundamentals
from src.data.universe import get_universe, get_sector_map, get_ticker_sector, BENCHMARK, get_download_tickers
UNIVERSE = get_universe()
SECTOR_MAP = get_sector_map()
TICKER_SECTOR = get_ticker_sector()
from src.viz.theme import apply_dark, SECTOR_COLORS
from src.viz.factor_charts import plot_universe_correlation

st.set_page_config(page_title="Universe Explorer", page_icon="🌐", layout="wide")
st.title("Universe Explorer")
st.caption("Explore the investment universe: sector composition, return history, and correlations")

# ---------------------------------------------------------------------------
# Background EDGAR cache warmup — fires once per session on first page visit
# so fundamental data is ready by the time the user reaches Multi-Factor Model.
# ---------------------------------------------------------------------------
import threading
from src.data.edgar import get_edgar_fundamentals_panel, edgar_cache_info

_all_tickers = tuple(sorted(UNIVERSE))
_cache_info  = edgar_cache_info(_all_tickers)
_n_cached    = len(_cache_info["cached"])
_n_total     = len(_all_tickers)
_n_missing   = len(_cache_info["missing"])

if _n_missing > 0 and not st.session_state.get("edgar_warmup_started"):
    from loguru import logger as _log

    def _warmup_edgar(tickers, n_missing):
        _log.info(f"EDGAR warmup: starting background load for {n_missing} uncached tickers")
        try:
            get_edgar_fundamentals_panel(tickers)
            _log.info("EDGAR warmup: complete")
        except Exception as e:
            _log.error(f"EDGAR warmup: failed — {type(e).__name__}: {e}")

    threading.Thread(
        target=_warmup_edgar, args=(_all_tickers, _n_missing), daemon=True
    ).start()
    st.session_state["edgar_warmup_started"] = True

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
all_sectors = list(SECTOR_MAP.keys())

# Stable-default pattern:
# Streamlit keys its keyless widgets on hash(label, options, default). If default changes
# between reruns, the widget is considered new and default is re-applied, wiping the user's
# selection. So we compute _ue_widget_default ONCE per page session (on fresh navigation when
# _ue_page_run == 0) and never change it mid-session. Other pages reset _ue_page_run to 0.
_ue_page_run = st.session_state.get("_ue_page_run", 0)
st.session_state["_ue_page_run"] = _ue_page_run + 1
if _ue_page_run == 0:
    st.session_state["_ue_widget_default"] = st.session_state.get("_sectors_shadow") or all_sectors

with st.sidebar:
    st.header("Universe Settings")
    selected_sectors = st.multiselect(
        "Sectors", all_sectors,
        default=st.session_state["_ue_widget_default"],
        help="Filter the universe by sector",
    )
    lookback_years = st.slider("Price History (years)", 1, 5, 3)
    force_refresh = st.button("Refresh Data", help="Force re-download from yfinance")

    st.markdown("---")
    if _n_missing == 0:
        st.success(f"EDGAR ready — {_n_total} tickers cached")
    elif st.session_state.get("edgar_warmup_started"):
        st.info(
            f"Loading EDGAR fundamental data in background "
            f"({_n_cached}/{_n_total} tickers cached). "
            "Multi-Factor Model fundamentals will be available once complete."
        )
    else:
        st.warning(f"EDGAR data not cached ({_n_missing} tickers missing).")

# Persist for cross-navigation and for other pages to read.
st.session_state["_sectors_shadow"] = selected_sectors
st.session_state["selected_sectors"] = selected_sectors

# Filter universe by sector
filtered = [t for t in UNIVERSE if TICKER_SECTOR.get(t) in selected_sectors]
if not filtered:
    st.warning("No tickers selected. Pick at least one sector.")
    st.stop()

# When sectors change, reset stock selection to all stocks in the new sector set
_curr_sectors = frozenset(selected_sectors)
if frozenset(st.session_state.get("_prev_sectors", [])) != _curr_sectors:
    st.session_state["selected_tickers"] = sorted(filtered)
st.session_state["_prev_sectors"] = list(selected_sectors)

# Ensure stock selection contains only tickers currently in the filtered universe
if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = sorted(filtered)
else:
    _valid = sorted([t for t in st.session_state["selected_tickers"] if t in filtered])
    if not _valid:
        st.session_state["selected_tickers"] = sorted(filtered)
    elif _valid != sorted(st.session_state["selected_tickers"]):
        st.session_state["selected_tickers"] = _valid

with st.sidebar:
    selected_tickers = st.multiselect(
        "Stocks",
        sorted(filtered),
        key="selected_tickers",
        help="Refine to specific tickers (defaults to all sector-filtered stocks)",
    )
    if not selected_tickers:
        selected_tickers = sorted(filtered)
        st.session_state["selected_tickers"] = selected_tickers

# Propagate to all pages
st.session_state["filtered_universe"] = selected_tickers

# Date range
import datetime
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
download_tickers = tuple(sorted(set(selected_tickers + [BENCHMARK])))

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86_400, show_spinner="Loading price data...")
def load_prices_cached(tickers, start, end, _force=False):
    return load_prices(tickers, start, end, force_refresh=_force)

with st.spinner("Loading data..."):
    try:
        prices = load_prices_cached(download_tickers, start_date, end_date, _force=force_refresh)
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        st.stop()

# Keep only tickers that loaded successfully
available = [t for t in selected_tickers if t in prices.columns]
if not available:
    st.error("No price data available for selected tickers.")
    st.stop()

prices_u = prices[available]
spy_prices = prices[BENCHMARK] if BENCHMARK in prices.columns else None

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
returns = prices_u.pct_change().dropna(how="all")
ytd_start = str(datetime.date.today().year) + "-01-01"
ytd_rets = prices_u.loc[ytd_start:].pct_change().dropna(how="all") if ytd_start in prices_u.index.astype(str) else returns

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers", len(available))
m2.metric("Sectors", len(selected_sectors))
m3.metric("Trading Days", len(prices_u))

median_ytd = ((1 + ytd_rets).prod() - 1).median()
m4.metric("Median YTD Return", f"{median_ytd:.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Sector Breakdown", "Price Performance", "Correlation Matrix", "Data Table"])

# --- Tab 1: Sector breakdown ---
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        sector_counts = pd.Series({s: len([t for t in available if TICKER_SECTOR.get(t) == s]) for s in selected_sectors})
        sector_counts = sector_counts[sector_counts > 0]
        fig_pie = go.Figure(go.Pie(
            labels=sector_counts.index,
            values=sector_counts.values,
            hole=0.4,
            marker_colors=[SECTOR_COLORS.get(s, "#7f8c8d") for s in sector_counts.index],
            textinfo="label+percent",
            hovertemplate="%{label}<br>Tickers: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        apply_dark(fig_pie, title="Universe Sector Composition", height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Annualized vol by sector
        ann_vol = returns.std() * (252 ** 0.5)
        vol_df = pd.DataFrame({
            "ticker": ann_vol.index,
            "vol": ann_vol.values,
            "sector": [TICKER_SECTOR.get(t, "Unknown") for t in ann_vol.index],
        })
        fig_vol = px.box(
            vol_df, x="sector", y="vol", color="sector",
            color_discrete_map={s: SECTOR_COLORS.get(s, "#7f8c8d") for s in vol_df["sector"].unique()},
            labels={"vol": "Ann. Volatility", "sector": ""},
            title="Volatility by Sector",
        )
        apply_dark(fig_vol, title="Volatility Distribution by Sector", height=380)
        fig_vol.update_layout(showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)

# --- Tab 2: Price Performance ---
with tab2:
    # Normalize to 100
    norm = prices_u / prices_u.iloc[0] * 100
    if spy_prices is not None:
        spy_norm = spy_prices / spy_prices.iloc[0] * 100

    view_mode = st.radio("View", ["All Tickers", "Sector Medians"], horizontal=True)

    if view_mode == "Sector Medians":
        fig_perf = go.Figure()
        if spy_prices is not None:
            fig_perf.add_trace(go.Scatter(
                x=spy_norm.index, y=spy_norm.values,
                mode="lines", name="SPY", line=dict(color="#aaaaaa", width=2, dash="dash"),
            ))
        for sector in selected_sectors:
            s_tickers = [t for t in available if TICKER_SECTOR.get(t) == sector]
            if not s_tickers:
                continue
            med = norm[s_tickers].median(axis=1)
            fig_perf.add_trace(go.Scatter(
                x=med.index, y=med.values,
                mode="lines", name=sector,
                line=dict(color=SECTOR_COLORS.get(sector, "#7f8c8d"), width=2),
            ))
    else:
        fig_perf = go.Figure()
        if spy_prices is not None:
            fig_perf.add_trace(go.Scatter(
                x=spy_norm.index, y=spy_norm.values,
                mode="lines", name="SPY", line=dict(color="#aaaaaa", width=2.5, dash="dash"),
            ))
        for ticker in available:
            sector = TICKER_SECTOR.get(ticker, "Unknown")
            color = SECTOR_COLORS.get(sector, "#7f8c8d")
            fig_perf.add_trace(go.Scatter(
                x=norm.index, y=norm[ticker].values,
                mode="lines", name=ticker,
                line=dict(color=color, width=1),
                opacity=0.6,
            ))

    apply_dark(fig_perf, title="Indexed Price Performance (Base = 100)", height=450)
    fig_perf.update_yaxes(title_text="Index (Base 100)")
    st.plotly_chart(fig_perf, use_container_width=True)

# --- Tab 3: Correlation matrix ---
with tab3:
    st.markdown("Return correlation sorted by sector — blocks reveal intra-sector clustering.")
    corr = returns.corr()
    fig_corr = plot_universe_correlation(corr, TICKER_SECTOR)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Tab 4: Data table ---
with tab4:
    ytd = ((1 + ytd_rets).prod() - 1) * 100
    ann_ret = ((prices_u.iloc[-1] / prices_u.iloc[0]) ** (252 / len(prices_u)) - 1) * 100
    ann_vol_pct = returns.std() * (252 ** 0.5) * 100
    summary = pd.DataFrame({
        "Sector": [TICKER_SECTOR.get(t, "Unknown") for t in available],
        "YTD (%)": ytd.round(1),
        "Ann. Return (%)": ann_ret.round(1),
        "Ann. Vol (%)": ann_vol_pct.round(1),
        "Latest Price": prices_u.iloc[-1].round(2),
    }, index=available)
    summary.index.name = "Ticker"
    st.dataframe(
        summary.sort_values("Sector"),
        use_container_width=True,
        column_config={
            "YTD (%)": st.column_config.NumberColumn(format="%.1f"),
            "Ann. Return (%)": st.column_config.NumberColumn(format="%.1f"),
            "Ann. Vol (%)": st.column_config.NumberColumn(format="%.1f"),
            "Latest Price": st.column_config.NumberColumn(format="$%.2f"),
        }
    )
