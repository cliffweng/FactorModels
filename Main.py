"""Factor Models Research App — entry point.

Run with:  streamlit run Main.py

All shared sidebar content (branding, version) lives here and appears on
every page.  Individual pages must NOT call st.set_page_config().
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import src.factors  # noqa: register all @register_factor decorators

st.set_page_config(
    page_title="Factor Models Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Persistent sidebar branding — visible on every page
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("`v1.01` made w/love  by the Wengs 2026", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
pg = st.navigation([
    st.Page("pages/0_Home.py",              title="Home",              icon="🏠"),
    st.Page("pages/1_Universe_Explorer.py", title="Universe Explorer", icon="🌐"),
    st.Page("pages/2_Factor_Library.py",    title="Factor Library",    icon="📚"),
    st.Page("pages/3_Factor_Lab.py",        title="Factor Lab",        icon="🔬"),
    st.Page("pages/4_IC_Analysis.py",       title="IC Analysis",       icon="📈"),
    st.Page("pages/5_Backtest.py",          title="Backtest",          icon="💰"),
    st.Page("pages/6_Factor_Correlation.py", title="Factor Correlation", icon="🔗"),
    st.Page("pages/7_Multi_Factor_Model.py", title="Multi-Factor Model", icon="🧬"),
    st.Page("pages/8_Signal_Lab.py",        title="Signal Lab",        icon="🔭"),
])

pg.run()
