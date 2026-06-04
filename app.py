"""Factor Models Research App — Entry Point."""
import sys
from pathlib import Path

# Ensure src/ is importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Factor Models Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import all factors so they self-register
# ---------------------------------------------------------------------------
import src.factors  # noqa: F401  triggers all @register_factor decorators

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
from src.data.cache import list_cache, clear_cache
from src.factors.base import get_registry

st.title("Factor Models Research Platform")
st.caption("A modular, interactive environment for quant factor research")

# Workflow diagram
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div style='background:#1a2744;border-radius:8px;padding:16px;text-align:center;height:110px'>
        <div style='font-size:24px'>🌐</div>
        <div style='color:#4c9be8;font-weight:600;margin-top:4px'>Universe</div>
        <div style='color:#aaa;font-size:12px;margin-top:4px'>~75 US large-caps across 11 sectors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div style='background:#1a2744;border-radius:8px;padding:16px;text-align:center;height:110px'>
        <div style='font-size:24px'>💾</div>
        <div style='color:#4c9be8;font-weight:600;margin-top:4px'>Data</div>
        <div style='color:#aaa;font-size:12px;margin-top:4px'>yfinance prices + fundamentals, pickle cached</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div style='background:#1a2744;border-radius:8px;padding:16px;text-align:center;height:110px'>
        <div style='font-size:24px'>🔬</div>
        <div style='color:#4c9be8;font-weight:600;margin-top:4px'>Factors</div>
        <div style='color:#aaa;font-size:12px;margin-top:4px'>Momentum · Risk · Value · Quality</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div style='background:#1a2744;border-radius:8px;padding:16px;text-align:center;height:110px'>
        <div style='font-size:24px'>📈</div>
        <div style='color:#4c9be8;font-weight:600;margin-top:4px'>Analysis</div>
        <div style='color:#aaa;font-size:12px;margin-top:4px'>IC · ICIR · Quantile portfolios</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        """
        <div style='background:#1a2744;border-radius:8px;padding:16px;text-align:center;height:110px'>
        <div style='font-size:24px'>💰</div>
        <div style='color:#4c9be8;font-weight:600;margin-top:4px'>Backtest</div>
        <div style='color:#aaa;font-size:12px;margin-top:4px'>Long-short · Sharpe · Drawdown</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Two-column: factor registry + methodology
left, right = st.columns([1, 1])

with left:
    st.subheader("Registered Factors")
    registry = get_registry()
    rows = []
    for name, factor in registry.items():
        rows.append({
            "Factor": factor.label,
            "Category": factor.category,
            "Direction": "Higher is better" if factor.direction == 1 else "Lower is better",
            "Panel": "No (snapshot)" if factor.requires_fundamentals else "Yes",
        })
    import pandas as pd
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

# Cache status in sidebar
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
        st.info("No cached data yet. Visit a page to load data.")

    st.markdown("---")
    st.markdown(
        """
        **Navigation**
        Use the pages in the left sidebar:
        1. Universe Explorer
        2. Factor Lab
        3. IC Analysis
        4. Backtest
        5. Factor Correlation
        """
    )
