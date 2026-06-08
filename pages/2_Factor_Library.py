"""Factor Library — enable or disable factors globally across all analysis pages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import src.factors  # noqa: register factors
from src.factors.base import get_registry

st.title("Factor Library")
st.caption("Enable or disable factors. Active factors appear in IC Analysis, Backtest, Factor Correlation, Multi-Factor Model, and Signal Lab.")

registry = get_registry()

# ---------------------------------------------------------------------------
# Snapshot-based sync
#
# This page manages _tab_active_{name} toggles → active_factors.
# The header toggle on Factor Lab manages the same active_factors set.
# We use a snapshot of active_factors from our last render to tell apart:
#   • "user just clicked a toggle on this page"  → toggle differs from snapshot
#   • "Factor Lab changed active_factors externally" → active_factors differs from snapshot
# ---------------------------------------------------------------------------

# EDGAR factors are disabled by default to prevent accidental downloads on first visit.
# Users can enable them explicitly via the toggles below.
_DEFAULT_ACTIVE = frozenset(n for n, f in registry.items() if not f.requires_edgar and f.enabled_by_default)
_af = set(st.session_state.get("active_factors", _DEFAULT_ACTIVE))
_snap_raw = st.session_state.get("_flib_snap")

if _snap_raw is None:
    # First visit: initialise every toggle key from active_factors
    for _n in registry:
        st.session_state[f"_tab_active_{_n}"] = _n in _af
else:
    _snap = set(_snap_raw)

    # Detect user clicks: toggles that differ from our last-rendered snapshot
    _clicked = {
        _n: st.session_state[f"_tab_active_{_n}"]
        for _n in registry
        if f"_tab_active_{_n}" in st.session_state
        and st.session_state[f"_tab_active_{_n}"] != (_n in _snap)
    }

    if _clicked:
        # Apply user clicks to active_factors
        for _n, _v in _clicked.items():
            _af.add(_n) if _v else _af.discard(_n)
        st.session_state["active_factors"] = _af
    else:
        # No click — force any keys that drifted due to external changes (Factor Lab toggle)
        for _n in registry:
            st.session_state[f"_tab_active_{_n}"] = _n in _af

# Store snapshot for next rerun
st.session_state["_flib_snap"] = frozenset(_af)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_n_active = sum(1 for n in registry if n in _af)
_c_stat, _c_all, _c_none = st.columns([4, 1, 1])
_c_stat.markdown(f"**{_n_active} / {len(registry)} factors active**")
if _c_all.button("Enable all", use_container_width=True):
    _af = set(registry.keys())
    st.session_state["active_factors"] = _af
    st.session_state["_flib_snap"] = None   # force re-init on next rerun
    st.rerun()
if _c_none.button("Disable all", use_container_width=True):
    _af = set()
    st.session_state["active_factors"] = _af
    st.session_state["_flib_snap"] = None
    st.rerun()

st.markdown("---")

_by_cat: dict[str, list] = {}
for _n, _f in registry.items():
    _by_cat.setdefault(_f.category, []).append((_n, _f))

for _cat in sorted(_by_cat.keys()):
    st.markdown(f"#### {_cat}")
    for _n, _f in _by_cat[_cat]:
        _c_tog, _c_info = st.columns([1, 11])
        with _c_tog:
            st.toggle("", key=f"_tab_active_{_n}", label_visibility="collapsed")
        with _c_info:
            _dir = "Higher is better (+)" if _f.direction == 1 else "Lower is better (−)"
            st.markdown(
                f"**{_f.label}**  \n"
                f"*{_f.category}* · {_dir}  \n"
                f"{_f.description}"
            )
    st.markdown("")
