"""Home page — workflow overview, factor registry, and cache utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import src.factors  # noqa: register all factors

from src.data.cache import list_cache, clear_cache
from src.factors.base import get_registry

st.title("Factor Models Research Platform")
st.caption("A modular, interactive environment for quant factor research")

# ---------------------------------------------------------------------------
# Workflow diagram
# ---------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

_card = (
    "<div style='background:#1a2744;border-radius:8px;padding:16px;"
    "text-align:center;height:110px'>"
    "<div style='font-size:24px'>{icon}</div>"
    "<div style='color:#4c9be8;font-weight:600;margin-top:4px'>{title}</div>"
    "<div style='color:#aaa;font-size:12px;margin-top:4px'>{body}</div>"
    "</div>"
)

with col1:
    st.markdown(_card.format(icon="🌐", title="Universe",
        body="~75 US large-caps across 11 sectors"), unsafe_allow_html=True)
with col2:
    st.markdown(_card.format(icon="💾", title="Data",
        body="yfinance prices + fundamentals, pickle cached"), unsafe_allow_html=True)
with col3:
    st.markdown(_card.format(icon="🔬", title="Factors",
        body="Momentum · Risk · Value · Quality"), unsafe_allow_html=True)
with col4:
    st.markdown(_card.format(icon="📈", title="Analysis",
        body="IC · ICIR · Quantile portfolios"), unsafe_allow_html=True)
with col5:
    st.markdown(_card.format(icon="💰", title="Backtest",
        body="Long-short · Sharpe · Drawdown"), unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Factor registry + research methodology
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Registered Factors")
    registry = get_registry()
    rows = []
    for name, factor in registry.items():
        rows.append({
            "Factor":    factor.label,
            "Category":  factor.category,
            "Direction": "Higher is better" if factor.direction == 1 else "Lower is better",
            "Panel":     "No (snapshot)" if factor.requires_fundamentals else "Yes",
            "Default":   "✓" if factor.enabled_by_default else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Research Methodology")
    st.markdown(
        """
        **Factor Definition**
        A factor is a measurable characteristic of a stock that has historically
        predicted cross-sectional returns. Factors are scored, ranked, and evaluated
        for their ability to separate future winners from losers.

        **Information Coefficient (IC)**
        IC = Spearman rank correlation between factor scores at time *t* and
        forward returns from *t → t+h*. |IC| > 0.05 is considered meaningful;
        ICIR = IC / std(IC) > 0.5 indicates persistent predictive power.

        **Quantile Analysis**
        Stocks are sorted by factor score and divided into quintiles (Q1–Q5).
        Equal-weight portfolios are rebalanced monthly. Spread = Q5 − Q1.

        **Long-Short Backtest**
        Long top quintile, short bottom quintile. Zero net investment.
        Transaction costs applied at each rebalance.

        **Data Sources**
        - Prices: `yfinance` (adjusted close, daily)
        - Fundamentals: `yfinance.Ticker.info` (point-in-time snapshot)
        - Cache: pickle files in `.cache/` with 24h TTL
        """
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Custom Universe Management
# ---------------------------------------------------------------------------
from src.data.custom_universe import load_custom, add_ticker, remove_ticker, ALL_SECTORS

st.subheader("Custom Universe")
st.caption(
    "Add any ticker to include it in all pages. "
    "Price cache is cleared on changes so new data downloads on next page visit."
)

custom = load_custom()

add_col, _ = st.columns([2, 3])
with add_col:
    with st.form("add_ticker_form", clear_on_submit=True):
        t_col, s_col, btn_col = st.columns([2, 2, 1])
        new_ticker = t_col.text_input("Ticker", placeholder="e.g. PLTR").strip().upper()
        new_sector = s_col.selectbox("Sector", ALL_SECTORS, index=ALL_SECTORS.index("Custom"))
        submitted  = btn_col.form_submit_button("Add", use_container_width=True)
        if submitted and new_ticker:
            add_ticker(new_ticker, new_sector)
            clear_cache(prefix="get_prices_custom")
            st.success(f"Added {new_ticker} ({new_sector})")
            st.rerun()

if custom:
    rows = [{"Ticker": t, "Sector": s} for t, s in sorted(custom.items())]
    st.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)

    remove_choice = st.selectbox("Remove ticker", ["—"] + sorted(custom.keys()))
    if st.button("Remove", disabled=(remove_choice == "—")):
        remove_ticker(remove_choice)
        clear_cache(prefix="get_prices_custom")
        st.rerun()
else:
    st.info("No custom tickers yet.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Cache status (sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Cache Status")
    cache_files = list_cache()
    if cache_files:
        total_kb = sum(f["size_kb"] for f in cache_files)
        st.metric("Cached Files", len(cache_files), help="Pickle files in .cache/")
        st.metric("Cache Size", f"{total_kb:.0f} KB")
        if st.button("Clear All Cache", type="secondary"):
            n = clear_cache()
            st.success(f"Cleared {n} cache files")
            st.rerun()
    else:
        st.info("No cached data yet.")
