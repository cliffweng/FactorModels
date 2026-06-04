"""Multi-Factor Model — build, analyse, optimise, and backtest composite factor models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import src.factors  # noqa: register all factors

from src.data.loader import get_prices, get_fundamentals
from src.data.universe import UNIVERSE, BENCHMARK

from src.factors.base import get_registry
from src.factors.composite import CompositeModel

from src.analysis.ic import compute_ic_series, compute_rolling_ic, compute_icir, compute_ic_decay
from src.analysis.backtest import run_backtest
from src.analysis.stats import summary_stats
from src.analysis import optimizer as opt
from src.analysis.strategy_store import list_strategies, save_strategy, delete_strategy

from src.viz.multi_factor_charts import (
    plot_weight_comparison, plot_weights_radar,
    plot_factor_contributions,
    plot_ic_comparison, plot_ic_summary_bars,
    plot_backtest_overlay,
    plot_efficient_frontier, plot_sensitivity,
)
from src.viz.ic_charts import plot_ic_decay, plot_ic_bar
from src.viz.theme import apply_dark

st.set_page_config(page_title="Multi-Factor Model", page_icon="🧬", layout="wide")
st.title("Multi-Factor Model")
st.caption("Combine multiple factors into a composite signal, evaluate IC, and optimise weights")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
registry = get_registry()
# All registered factors (for multiselect)
all_factor_labels   = {f.label: name for name, f in registry.items()}
# Price-based only (used for defaults)
price_factor_labels = {f.label: name for name, f in registry.items() if not f.requires_fundamentals}

# ---------------------------------------------------------------------------
# Apply pending factor selection from Strategies tab BEFORE sidebar renders
# ---------------------------------------------------------------------------
if "pending_sidebar_factors" in st.session_state:
    st.session_state["sel_factors"] = st.session_state.pop("pending_sidebar_factors")

# ---------------------------------------------------------------------------
# Sidebar — factor selection and global settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Settings")

    selected_labels = st.multiselect(
        "Factors",
        options=sorted(all_factor_labels.keys()),
        default=list(price_factor_labels.keys())[:3],
        help="All 10 factors available. Fundamental factors (P/B, ROE, etc.) appear in cross-section scores but are excluded from IC / Backtest / Optimizer.",
        key="sel_factors",
    )

    rebal_freq = st.selectbox(
        "Rebalance Frequency",
        ["ME", "W-FRI", "QE"],
        format_func=lambda x: {"ME": "Monthly", "W-FRI": "Weekly", "QE": "Quarterly"}[x],
    )
    ic_horizon = st.select_slider(
        "IC Horizon",
        options=[1, 5, 10, 21, 42, 63],
        value=21,
        format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m", 42: "2m", 63: "3m"}.get(x, f"{x}d"),
    )
    lookback_years = st.slider("Price History (years)", 2, 5, 3)
    force_refresh = st.button("Refresh Data")

    st.markdown("---")
    st.caption(
        "Fundamental factors (P/B, ROE, etc.) contribute to cross-section scores "
        "but are excluded from IC / Backtest / Optimizer (no historical panel data)."
    )

if len(selected_labels) < 2:
    st.warning("Select at least 2 factors to build a composite model.")
    st.stop()

selected_names = [all_factor_labels[lbl] for lbl in selected_labels]
selected_factors = [registry[n] for n in selected_names]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
tickers = tuple(sorted(set(UNIVERSE + [BENCHMARK])))

@st.cache_data(ttl=86_400, show_spinner="Loading prices...")
def load_prices_cached(tickers, start, end, _force=False):
    return get_prices(tickers, start, end, force_refresh=_force)

with st.spinner("Loading price data..."):
    try:
        prices = load_prices_cached(tickers, start_date, end_date, _force=force_refresh)
    except Exception as e:
        st.error(f"Price data error: {e}")
        st.stop()

stock_cols = [t for t in UNIVERSE if t in prices.columns]
prices_stocks = prices[stock_cols]
daily_returns = prices.pct_change().dropna(how="all")
daily_returns_stocks = prices_stocks.pct_change().dropna(how="all")

# ---------------------------------------------------------------------------
# Pre-compute individual factor panels + IC series (cached per factor)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def compute_factor_panel(factor_name, stock_cols, start, end, freq, _prices):
    f = registry[factor_name]
    return f.compute_panel(_prices[list(stock_cols)], freq=freq)

@st.cache_data(ttl=3600, show_spinner=False)
def compute_factor_ic_series(factor_name, _panel, _returns, horizon):
    return compute_ic_series(_panel, _returns, horizon_days=horizon)

with st.spinner("Computing factor panels..."):
    factor_panels: dict[str, pd.DataFrame] = {}
    for name in selected_names:
        if registry[name].requires_fundamentals:
            continue   # snapshot-only; contributes to scores but not panels
        try:
            panel = compute_factor_panel(
                name, tuple(stock_cols), start_date, end_date,
                rebal_freq, prices_stocks,
            )
            if not panel.empty:
                factor_panels[name] = panel
        except Exception as e:
            st.warning(f"Could not compute panel for {registry[name].label}: {e}")

if not factor_panels:
    st.error("No factor panels could be computed. Try a different factor selection or longer history.")
    st.stop()

# Price-based factors with successful panel computation
active_names   = list(factor_panels.keys())
active_factors = [registry[n] for n in active_names]
active_labels  = {n: registry[n].label for n in active_names}

# Fundamental (snapshot-only) factors among the user's selection
fundamental_names = [n for n in selected_names if registry[n].requires_fundamentals]

# Combined ordered list: price-based first, then fundamental
# (slider keys follow this order so pending_weights applies correctly)
all_selected_names   = active_names + fundamental_names
all_selected_factors = [registry[n] for n in all_selected_names]
all_selected_labels  = {n: registry[n].label for n in all_selected_names}

with st.spinner("Computing individual IC series..."):
    ic_series_map: dict[str, pd.Series] = {}
    for name in active_names:
        panel = factor_panels[name]
        ic_s = compute_factor_ic_series(name, panel, daily_returns_stocks, ic_horizon)
        if len(ic_s) >= 3:
            ic_series_map[name] = ic_s

# Load fundamentals if the user selected any fundamental factors
@st.cache_data(ttl=86_400, show_spinner="Loading fundamental data...")
def _load_fundamentals(tickers, _force=False):
    return get_fundamentals(tickers, force_refresh=_force)

fundamentals = None
if fundamental_names:
    try:
        with st.spinner("Loading fundamental data..."):
            fundamentals = _load_fundamentals(tuple(sorted(UNIVERSE)), _force=force_refresh)
    except Exception:
        fundamentals = None

# ---------------------------------------------------------------------------
# Apply pending weights from Optimizer tab BEFORE sliders are instantiated
# ---------------------------------------------------------------------------
if "pending_weights" in st.session_state:
    for _name, _scaled_w in st.session_state.pop("pending_weights").items():
        st.session_state[f"w_{_name}"] = _scaled_w

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_builder, tab_ic, tab_backtest, tab_optimizer, tab_strategies = st.tabs([
    "Model Builder", "IC Analysis", "Backtest", "Optimizer", "Strategies",
])

# ===========================================================================
# TAB 1 — MODEL BUILDER
# ===========================================================================
with tab_builder:
    st.subheader("Weight Configuration")
    st.caption("Set raw weights for each factor. Weights are normalised to sum to 1.")

    # Weight sliders — one per selected factor (price-based + fundamental)
    weight_cols = st.columns(min(len(all_selected_names), 4))
    raw_weights: list[float] = []
    for i, name in enumerate(all_selected_names):
        col = weight_cols[i % len(weight_cols)]
        with col:
            f_obj = registry[name]
            label = all_selected_labels[name]
            direction_note = "(−)" if f_obj.direction == -1 else "(+)"
            snap_note = " ★" if f_obj.requires_fundamentals else ""
            w = st.slider(
                f"{label}{snap_note} {direction_note}",
                min_value=0.0, max_value=2.0, value=1.0, step=0.05,
                key=f"w_{name}",
                help=f"{f_obj.description}  Direction automatically applied."
                     + (" ★ Snapshot-only — excluded from IC/Backtest/Optimizer."
                        if f_obj.requires_fundamentals else ""),
            )
            raw_weights.append(w)

    if fundamental_names:
        st.caption("★ Snapshot-only factors contribute to the cross-section scores below but are excluded from IC Analysis, Backtest, and Optimizer.")

    # Build composite — all selected factors
    model = CompositeModel(all_selected_factors, raw_weights, name="Composite")

    # Show normalised weight summary
    nw = model.weights_dict()
    st.markdown(
        " · ".join(f"**{all_selected_labels[n]}**: {nw[n]:.1%}" for n in all_selected_names
                   if abs(nw.get(n, 0)) > 1e-9)
    )
    st.markdown("---")

    # Composite cross-section scores (fundamental factors included when loaded)
    _score_kwargs = {"fundamentals": fundamentals} if fundamentals is not None else {}
    composite_scores = model.compute_scores(prices_stocks, **_score_kwargs)
    contrib_df = model.factor_contributions(prices_stocks, **_score_kwargs)

    c_left, c_right = st.columns(2)

    with c_left:
        if not composite_scores.empty:
            from src.viz.factor_charts import plot_factor_bar
            from src.data.universe import TICKER_SECTOR
            fig_scores = plot_factor_bar(
                composite_scores,
                ticker_sector=TICKER_SECTOR,
                title="Composite Scores (current cross-section)",
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("No composite scores available.")

    with c_right:
        if not contrib_df.empty:
            fig_contrib = plot_factor_contributions(
                contrib_df,
                title="Factor Contributions per Ticker",
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.info("No contribution data available.")

    # Radar chart of current weights
    st.markdown("---")
    if len(nw) >= 3:
        radar_col, spacer = st.columns([1, 1])
        with radar_col:
            fig_radar = plot_weights_radar(
                {all_selected_labels.get(n, n): v for n, v in nw.items()},
                title="Weight Allocation",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ------------------------------------------------------------------
    # Save as Strategy
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Save as Strategy")
    st.caption(
        "Snapshot the current factor selection and weights as a named strategy. "
        "Saved strategies can be compared side-by-side in the **Strategies** tab."
    )
    save_col, name_col = st.columns([2, 3])
    with name_col:
        strategy_name = st.text_input(
            "Strategy name",
            value="My Strategy",
            key="strategy_name_input",
            label_visibility="collapsed",
            placeholder="Strategy name…",
        )
    with save_col:
        if st.button("Save Strategy", type="primary"):
            if strategy_name.strip():
                save_strategy(
                    name=strategy_name.strip(),
                    factors={n: float(raw_weights[i]) for i, n in enumerate(all_selected_names)},
                    rebal_freq=rebal_freq,
                )
                st.success(f"Strategy **{strategy_name.strip()}** saved. View it in the **Strategies** tab.")
            else:
                st.warning("Enter a name before saving.")

# ===========================================================================
# TAB 2 — IC ANALYSIS
# ===========================================================================
with tab_ic:
    if len(ic_series_map) < 1:
        st.warning("Not enough IC data. Try a longer price history.")
    else:
        # Compute composite IC using current weights
        @st.cache_data(ttl=3600, show_spinner="Computing composite panel & IC...")
        def get_composite_panel_and_ic(
            factor_names, weights_tuple, start, end, freq, horizon,
            _prices,
        ):
            m = CompositeModel(
                [registry[n] for n in factor_names],
                list(weights_tuple),
            )
            try:
                panel = m.compute_panel(_prices, freq=freq)
            except Exception:
                return pd.DataFrame(), pd.Series(dtype=float)
            if panel.empty:
                return panel, pd.Series(dtype=float)
            dr = _prices.pct_change().dropna(how="all")
            ic_s = compute_ic_series(panel, dr, horizon_days=horizon)
            return panel, ic_s

        # Use all selected names — CompositeModel.compute_panel() skips fundamental factors internally
        w_tuple = tuple(raw_weights)
        composite_panel, composite_ic = get_composite_panel_and_ic(
            tuple(all_selected_names), w_tuple, start_date, end_date,
            rebal_freq, ic_horizon, prices_stocks,
        )

        # Multi-line IC comparison chart
        ic_label_map = {registry[n].label: ic_series_map[n] for n in ic_series_map}
        fig_ic_lines = plot_ic_comparison(
            ic_label_map,
            composite_ic=composite_ic if len(composite_ic) > 0 else None,
            title=f"IC Time Series (3-period rolling, {ic_horizon}d horizon)",
        )
        st.plotly_chart(fig_ic_lines, use_container_width=True)
        st.markdown("---")

        # IC summary bars (Mean IC + ICIR per factor + composite)
        summary_rows = []
        for name in active_names:
            if name in ic_series_map:
                s = ic_series_map[name]
                summary_rows.append({
                    "name": active_labels[name],
                    "IC_mean": s.mean(),
                    "ICIR": compute_icir(s),
                })
        if len(composite_ic) >= 3:
            summary_rows.append({
                "name": "Composite",
                "IC_mean": composite_ic.mean(),
                "ICIR": compute_icir(composite_ic),
            })

        if summary_rows:
            fig_summary = plot_ic_summary_bars(
                summary_rows,
                title="Mean IC and ICIR — Individual Factors vs Composite",
            )
            st.plotly_chart(fig_summary, use_container_width=True)

        # Composite IC decay
        if not composite_panel.empty:
            st.markdown("---")
            st.subheader("Composite IC Decay")
            with st.spinner("Computing IC decay..."):
                @st.cache_data(ttl=3600)
                def get_composite_decay(_panel, _returns, key):
                    return compute_ic_decay(_panel, _returns)

                decay_df = get_composite_decay(
                    composite_panel, daily_returns_stocks,
                    f"{'_'.join(active_names)}_{rebal_freq}_{ic_horizon}",
                )
            if not decay_df.empty:
                fig_decay = plot_ic_decay(decay_df, title="Composite Factor IC Decay")
                st.plotly_chart(fig_decay, use_container_width=True)

        # Individual IC series detail
        with st.expander("Individual Factor IC Series"):
            for name in active_names:
                if name not in ic_series_map:
                    continue
                ic_s = ic_series_map[name]
                rolling = compute_rolling_ic(ic_s, window=12)
                fig_individual = plot_ic_bar(
                    ic_s, rolling,
                    title=f"{active_labels[name]} IC",
                )
                st.plotly_chart(fig_individual, use_container_width=True)

# ===========================================================================
# TAB 3 — BACKTEST
# ===========================================================================
with tab_backtest:
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_factor_backtest(factor_name, _panel, _daily_returns, direction):
        return run_backtest(_panel, _daily_returns, direction=direction)

    @st.cache_data(ttl=3600, show_spinner=False)
    def run_composite_backtest(factor_names, weights_tuple, _prices, _daily_returns, freq):
        m = CompositeModel(
            [registry[n] for n in factor_names],
            list(weights_tuple),
        )
        try:
            panel = m.compute_panel(_prices, freq=freq)
        except Exception:
            return None
        if panel.empty:
            return None
        return run_backtest(panel, _daily_returns, direction=1)

    with st.spinner("Running backtests..."):
        cumulative_dict: dict[str, pd.Series] = {}
        stats_rows: list[dict] = []

        # Individual factor backtests
        for name in active_names:
            if name not in factor_panels:
                continue
            try:
                bt = run_factor_backtest(
                    name, factor_panels[name], daily_returns,
                    registry[name].direction,
                )
                cum = bt.cumulative_ls
                if len(cum) > 0:
                    cumulative_dict[active_labels[name]] = cum
                    row = summary_stats(bt.ls_returns, active_labels[name])
                    stats_rows.append(row)
            except Exception:
                pass

        # Composite backtest — fundamental factors skipped internally by CompositeModel
        bt_composite = run_composite_backtest(
            tuple(all_selected_names), w_tuple, prices_stocks, daily_returns, rebal_freq,
        )
        if bt_composite is not None:
            cum_comp = bt_composite.cumulative_ls
            if len(cum_comp) > 0:
                cumulative_dict["Composite"] = cum_comp
                row = summary_stats(bt_composite.ls_returns, "Composite")
                stats_rows.append(row)

    if not cumulative_dict:
        st.warning("No backtest results available.")
    else:
        # Benchmark
        spy_cum = None
        if bt_composite is not None and len(bt_composite.benchmark_returns) > 0:
            spy_cum = bt_composite.cumulative_benchmark

        fig_overlay = plot_backtest_overlay(
            cumulative_dict,
            benchmark=spy_cum,
            highlight="Composite",
            title="Long-Short Cumulative Returns — Individual Factors vs Composite",
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

        st.markdown("---")
        st.subheader("Performance Statistics")
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows).set_index("Name")
            st.dataframe(stats_df, use_container_width=True)

        # Drawdown for composite
        if bt_composite is not None:
            from src.viz.portfolio_charts import plot_drawdown
            fig_dd = plot_drawdown(bt_composite.drawdown, title="Composite L/S Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)

# ===========================================================================
# TAB 4 — OPTIMIZER
# ===========================================================================
with tab_optimizer:
    if len(ic_series_map) < 2:
        st.warning("Need IC series for at least 2 factors. Increase price history or select more factors.")
    else:
        ic_dict_for_opt = {n: ic_series_map[n] for n in ic_series_map}

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_opt = st.button("Run All Optimisers", type="primary")
        with col_info:
            st.caption(
                "Optimisers work in **IC space** — they find factor weights that maximise "
                "the composite ICIR (predictive power per unit of IC variance), "
                "analogous to Sharpe-ratio optimisation in return space."
            )

        if run_opt or st.session_state.get("opt_ran"):
            st.session_state["opt_ran"] = True

            @st.cache_data(ttl=3600, show_spinner="Running optimisers...")
            def run_optimizers_cached(ic_dict_key, _ic_dict):
                results = opt.run_all(_ic_dict)
                frontier = opt.efficient_frontier(_ic_dict)
                return results, frontier

            ic_key = "_".join(sorted(ic_dict_for_opt.keys())) + f"_{ic_horizon}_{rebal_freq}"
            opt_results, frontier_df = run_optimizers_cached(ic_key, ic_dict_for_opt)

            if not opt_results:
                st.error("Optimisation failed — insufficient IC history.")
            else:
                # ---------------------
                # Summary metrics row
                # ---------------------
                best_icir = max(r.icir for r in opt_results if not np.isnan(r.icir))
                eq_result = next((r for r in opt_results if r.method == "Equal Weight"), opt_results[0])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best ICIR achieved", f"{best_icir:.3f}")
                m2.metric("Equal-Weight ICIR", f"{eq_result.icir:.3f}")
                lift = best_icir - eq_result.icir
                m3.metric("ICIR Lift vs Equal", f"{lift:+.3f}")
                m4.metric("Factors in model", len(ic_dict_for_opt))

                st.markdown("---")

                # ---------------------
                # Weight comparison chart
                # ---------------------
                # Re-key by label for display
                label_results = []
                for r in opt_results:
                    relabelled = opt.OptimizeResult(
                        method=r.method,
                        weights={active_labels.get(n, n): w for n, w in r.weights.items()},
                        ic_mean=r.ic_mean,
                        ic_std=r.ic_std,
                        icir=r.icir,
                    )
                    label_results.append(relabelled)

                fig_weights = plot_weight_comparison(
                    label_results,
                    title="Factor Weights by Optimisation Method",
                )
                st.plotly_chart(fig_weights, use_container_width=True)

                # Optimizer results table
                rows_table = [
                    {
                        "Method": r.method,
                        "Mean IC": f"{r.ic_mean:.4f}",
                        "IC Std": f"{r.ic_std:.4f}",
                        "ICIR": f"{r.icir:.3f}",
                        **{active_labels.get(n, n): f"{w:.1%}" for n, w in r.weights.items()},
                    }
                    for r in opt_results
                ]
                st.dataframe(pd.DataFrame(rows_table).set_index("Method"), use_container_width=True)

                # ---------------------
                # Efficient frontier
                # ---------------------
                st.markdown("---")
                st.subheader("IC Efficient Frontier")
                st.caption(
                    "Each point on the curve is the minimum-IC-variance portfolio for a given "
                    "target Mean IC. Analogous to the Markowitz frontier in return/variance space. "
                    "Named markers show where each optimisation method lands."
                )

                named_pts = [
                    {"method": r.method, "IC_mean": r.ic_mean, "IC_std": r.ic_std, "ICIR": r.icir}
                    for r in opt_results
                ]
                fig_frontier = plot_efficient_frontier(frontier_df, named_pts)
                st.plotly_chart(fig_frontier, use_container_width=True)

                # ---------------------
                # Sensitivity / tornado
                # ---------------------
                st.markdown("---")
                st.subheader("Weight Sensitivity")
                st.caption("How much does composite ICIR change when each factor's weight is increased by +10%?")

                max_icir_result = next(
                    (r for r in opt_results if r.method == "Max ICIR"), opt_results[0]
                )

                @st.cache_data(ttl=3600)
                def compute_sensitivity(_ic_dict, base_weights_dict, nudge=0.10):
                    from src.analysis.optimizer import _align
                    mu, Sigma, names = _align(_ic_dict)
                    base_w = np.array([base_weights_dict.get(n, 0.0) for n in names])
                    base_w = base_w / base_w.sum()

                    base_icir_val = float(mu @ base_w) / max(np.sqrt(base_w @ Sigma @ base_w), 1e-12)

                    deltas = {}
                    for i, name in enumerate(names):
                        w_nudge = base_w.copy()
                        w_nudge[i] += nudge
                        w_nudge /= w_nudge.sum()
                        new_icir = float(mu @ w_nudge) / max(np.sqrt(w_nudge @ Sigma @ w_nudge), 1e-12)
                        deltas[name] = new_icir - base_icir_val

                    return base_icir_val, deltas

                base_icir_val, sensitivity = compute_sensitivity(
                    ic_dict_for_opt, max_icir_result.weights
                )
                sensitivity_labelled = {active_labels.get(n, n): v for n, v in sensitivity.items()}
                fig_sens = plot_sensitivity(base_icir_val, sensitivity_labelled)
                st.plotly_chart(fig_sens, use_container_width=True)

                # ---------------------
                # Apply best weights button
                # ---------------------
                st.markdown("---")
                best_method = max(opt_results, key=lambda r: r.icir if not np.isnan(r.icir) else -999)
                if st.button(f"Apply '{best_method.method}' weights to Model Builder"):
                    # Store as pending — sliders are already instantiated this run,
                    # so we cannot set their keys directly. On the next run the
                    # pending_weights block at the top of this script applies them
                    # before any slider is created.
                    pending = {}
                    for name, w in best_method.weights.items():
                        pending[name] = min(w * len(active_names), 2.0)
                    st.session_state["pending_weights"] = pending
                    st.rerun()
        else:
            st.info("Click **Run All Optimisers** to compute optimal weight allocations and the efficient frontier.")

# ===========================================================================
# TAB 5 — STRATEGIES
# ===========================================================================
with tab_strategies:
    all_strategies = list_strategies()

    if not all_strategies:
        st.info(
            "No strategies saved yet. Configure a model in the **Model Builder** tab "
            "and click **Save Strategy**."
        )
    else:
        # ----------------------------------------------------------------
        # Strategy cards
        # ----------------------------------------------------------------
        st.subheader("Saved Strategies")
        st.caption("Select strategies to compare, load them into the Model Builder, or delete them.")

        selected_strategy_names: list[str] = []

        for strat in all_strategies:
            with st.container(border=True):
                h_col, check_col = st.columns([5, 1])
                with h_col:
                    st.markdown(f"#### {strat['name']}")
                    st.caption(
                        f"Rebalance: **{strat['rebal_freq']}** · "
                        f"Saved: {strat['created_at'][:10]}"
                    )
                    # Factor weight pills
                    factor_total = sum(strat["factors"].values())
                    pills = []
                    for fn, fw in strat["factors"].items():
                        pct = fw / factor_total if factor_total > 1e-12 else 0
                        label = active_labels.get(fn, fn)
                        pills.append(f"**{label}** {pct:.0%}")
                    st.markdown("  ·  ".join(pills))
                with check_col:
                    selected = st.checkbox(
                        "Compare",
                        key=f"cmp_{strat['name']}",
                        value=False,
                    )
                    if selected:
                        selected_strategy_names.append(strat["name"])

                btn_load, btn_delete, _ = st.columns([1, 1, 4])
                with btn_load:
                    if st.button("Load into Builder", key=f"load_{strat['name']}"):
                        # Convert strategy factors to labels the sidebar multiselect understands
                        strat_factor_names = list(strat["factors"].keys())
                        strat_labels = [
                            registry[n].label
                            for n in strat_factor_names
                            if n in registry and n in price_factor_labels.values()
                        ]
                        if strat_labels:
                            st.session_state["pending_sidebar_factors"] = strat_labels
                        # Set weights — will be consumed by pending_weights block on next run
                        pending = {}
                        total_w = sum(strat["factors"].values())
                        n_factors = len(strat_factor_names)
                        for fn, fw in strat["factors"].items():
                            pending[fn] = min((fw / total_w if total_w > 1e-12 else 1.0 / n_factors) * n_factors, 2.0)
                        st.session_state["pending_weights"] = pending
                        st.rerun()
                with btn_delete:
                    if st.button("Delete", key=f"del_{strat['name']}"):
                        delete_strategy(strat["name"])
                        st.rerun()

        # ----------------------------------------------------------------
        # Comparison section
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("Strategy Comparison")

        compare_strategies = [s for s in all_strategies if s["name"] in selected_strategy_names]

        if len(compare_strategies) < 1:
            st.info("Tick **Compare** on one or more strategies above to visualise them together.")
        else:
            @st.cache_data(ttl=3600, show_spinner=False)
            def _strategy_backtest(strat_name, factor_names_tuple, weights_tuple, freq,
                                   _prices, _daily_returns):
                m = CompositeModel(
                    [registry[n] for n in factor_names_tuple if n in registry],
                    list(weights_tuple),
                )
                try:
                    panel = m.compute_panel(_prices, freq=freq)
                except Exception:
                    return None
                if panel.empty:
                    return None
                return run_backtest(panel, _daily_returns, direction=1)

            @st.cache_data(ttl=3600, show_spinner=False)
            def _strategy_ic(strat_name, factor_names_tuple, weights_tuple, freq,
                             horizon, _prices, _daily_returns):
                m = CompositeModel(
                    [registry[n] for n in factor_names_tuple if n in registry],
                    list(weights_tuple),
                )
                try:
                    panel = m.compute_panel(_prices, freq=freq)
                except Exception:
                    return pd.Series(dtype=float)
                if panel.empty:
                    return pd.Series(dtype=float)
                return compute_ic_series(panel, _daily_returns, horizon_days=horizon)

            with st.spinner("Running strategy comparisons..."):
                cmp_cumulative: dict[str, pd.Series] = {}
                cmp_ic: dict[str, pd.Series] = {}
                cmp_stats: list[dict] = []
                spy_cum_cmp = None

                for strat in compare_strategies:
                    fn_tuple = tuple(n for n in strat["factors"] if n in registry)
                    if not fn_tuple:
                        continue
                    total_w = sum(strat["factors"].get(n, 0) for n in fn_tuple)
                    w_tuple_cmp = tuple(
                        strat["factors"].get(n, 0) / total_w if total_w > 1e-12 else 1.0 / len(fn_tuple)
                        for n in fn_tuple
                    )
                    freq_cmp = strat.get("rebal_freq", "ME")

                    bt = _strategy_backtest(
                        strat["name"], fn_tuple, w_tuple_cmp, freq_cmp,
                        prices_stocks, daily_returns,
                    )
                    if bt is not None and len(bt.cumulative_ls) > 0:
                        cmp_cumulative[strat["name"]] = bt.cumulative_ls
                        row = summary_stats(bt.ls_returns, strat["name"])
                        cmp_stats.append(row)
                        if spy_cum_cmp is None and len(bt.cumulative_benchmark) > 0:
                            spy_cum_cmp = bt.cumulative_benchmark

                    ic_s = _strategy_ic(
                        strat["name"], fn_tuple, w_tuple_cmp, freq_cmp,
                        ic_horizon, prices_stocks, daily_returns_stocks,
                    )
                    if len(ic_s) >= 3:
                        cmp_ic[strat["name"]] = ic_s

            if cmp_cumulative:
                fig_cmp = plot_backtest_overlay(
                    cmp_cumulative,
                    benchmark=spy_cum_cmp,
                    highlight=None,
                    title="Long-Short Cumulative Returns — Strategy Comparison",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            if cmp_stats:
                st.markdown("---")
                st.subheader("Performance Statistics")
                cmp_stats_df = pd.DataFrame(cmp_stats).set_index("Name")
                st.dataframe(cmp_stats_df, use_container_width=True)

            if cmp_ic:
                st.markdown("---")
                st.subheader("IC Comparison")
                fig_ic_cmp = plot_ic_comparison(
                    cmp_ic,
                    composite_ic=None,
                    title=f"3-Period Rolling IC — Strategy Comparison ({ic_horizon}d horizon)",
                )
                st.plotly_chart(fig_ic_cmp, use_container_width=True)

                ic_summary_cmp = [
                    {"name": name, "IC_mean": s.mean(), "ICIR": compute_icir(s)}
                    for name, s in cmp_ic.items()
                ]
                fig_ic_sum_cmp = plot_ic_summary_bars(
                    ic_summary_cmp,
                    title="Mean IC and ICIR — Strategy Comparison",
                )
                st.plotly_chart(fig_ic_sum_cmp, use_container_width=True)
