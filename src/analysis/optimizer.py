"""Multi-factor weight optimization routines operating in IC space.

All methods accept `ic_dict: dict[str, pd.Series]` — a mapping from factor name
to its historical IC time series — and return an `OptimizeResult`.

The IC time series are inner-joined on dates before any computation so all
methods see the same set of observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    method: str
    weights: dict[str, float]   # factor_name → weight, sums to 1.0
    ic_mean: float               # portfolio IC mean = μ'w
    ic_std: float                # portfolio IC std = √(w'Σw)
    icir: float                  # ic_mean / ic_std


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align(ic_dict: dict[str, pd.Series]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align IC series to common dates.

    Returns:
        mu    shape (N,)   — mean IC per factor
        Sigma shape (N,N)  — IC covariance matrix (regularised)
        names list[str]    — factor names in column order
    """
    df = pd.DataFrame({k: v for k, v in ic_dict.items()}).dropna()
    if len(df) < 3:
        raise ValueError(
            f"Need at least 3 overlapping IC observations across factors. "
            f"Got {len(df)} after alignment."
        )
    names = list(df.columns)
    mu = df.mean().values.astype(float)
    Sigma = df.cov().values.astype(float)
    # Regularise to handle near-singular covariance (highly correlated factors)
    Sigma += np.eye(len(names)) * 1e-6
    return mu, Sigma, names


def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> tuple[float, float, float]:
    """Compute (ic_mean, ic_std, icir) for a weight vector."""
    ic_mean = float(mu @ weights)
    var = float(weights @ Sigma @ weights)
    ic_std = float(np.sqrt(max(var, 0)))
    icir = ic_mean / ic_std if ic_std > 1e-12 else float("nan")
    return ic_mean, ic_std, icir


def _make_result(method: str, w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, names: list[str]) -> OptimizeResult:
    ic_mean, ic_std, icir = _portfolio_stats(w, mu, Sigma)
    return OptimizeResult(
        method=method,
        weights={n: float(w[i]) for i, n in enumerate(names)},
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
    )


def _equal_w(names: list[str]) -> np.ndarray:
    n = len(names)
    return np.ones(n) / n


# ---------------------------------------------------------------------------
# Optimisation methods
# ---------------------------------------------------------------------------

def equal_weight(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Uniform 1/N weight across all factors."""
    mu, Sigma, names = _align(ic_dict)
    w = _equal_w(names)
    return _make_result("Equal Weight", w, mu, Sigma, names)


def ic_proportional(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Weights proportional to mean IC (negative mean IC → zero weight)."""
    mu, Sigma, names = _align(ic_dict)
    raw = np.maximum(mu, 0.0)
    total = raw.sum()
    w = raw / total if total > 1e-12 else _equal_w(names)
    return _make_result("IC-Proportional", w, mu, Sigma, names)


def icir_proportional(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Weights proportional to ICIR = mean(IC) / std(IC)."""
    df = pd.DataFrame({k: v for k, v in ic_dict.items()}).dropna()
    if len(df) < 3:
        raise ValueError("Insufficient overlapping IC observations.")
    names = list(df.columns)
    mu_arr = df.mean().values.astype(float)
    Sigma = df.cov().values.astype(float) + np.eye(len(names)) * 1e-6

    icir_vec = np.array([
        mu_arr[i] / df.iloc[:, i].std() if df.iloc[:, i].std() > 1e-12 else 0.0
        for i in range(len(names))
    ])
    raw = np.maximum(icir_vec, 0.0)
    total = raw.sum()
    w = raw / total if total > 1e-12 else _equal_w(names)
    return _make_result("ICIR-Proportional", w, mu_arr, Sigma, names)


def max_icir(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Tangency portfolio in IC space: maximise ICIR = μ'w / √(w'Σw).

    Long-only (w ≥ 0), weights sum to 1.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Max ICIR", np.array([1.0]), mu, Sigma, names)

    def neg_icir(w: np.ndarray) -> float:
        ic_m = mu @ w
        var = w @ Sigma @ w
        if var <= 1e-24:
            return 0.0
        return -ic_m / np.sqrt(var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    best_result, best_val = None, np.inf

    # Multi-start to escape local optima
    for seed_w in [_equal_w(names)] + [
        np.eye(n)[i] for i in range(n)
    ]:
        res = minimize(
            neg_icir, seed_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None or not best_result.success:
        return equal_weight(ic_dict)

    w = np.maximum(best_result.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Max ICIR", w, mu, Sigma, names)


def min_variance(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Minimum IC-variance portfolio."""
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Min Variance", np.array([1.0]), mu, Sigma, names)

    def port_var(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        port_var, _equal_w(names), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not res.success:
        return equal_weight(ic_dict)

    w = np.maximum(res.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Min Variance", w, mu, Sigma, names)


def risk_parity(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Equal IC-risk contribution from each factor.

    Finds weights such that each factor contributes the same share of total
    portfolio IC variance: w_i·(Σw)_i  =  w'Σw / N  for all i.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Risk Parity", np.array([1.0]), mu, Sigma, names)

    def risk_contrib_sse(w: np.ndarray) -> float:
        port_var = float(w @ Sigma @ w)
        if port_var < 1e-16:
            return 0.0
        rc = w * (Sigma @ w)
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(1e-4, 1.0)] * n   # small lower bound keeps every factor in

    best_res, best_val = None, np.inf
    for seed in [_equal_w(names)] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(
            risk_contrib_sse, seed, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    if best_res is None:
        return equal_weight(ic_dict)
    w = np.maximum(best_res.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Risk Parity", w, mu, Sigma, names)


def mean_variance(ic_dict: dict[str, pd.Series], risk_aversion: float = 1.0) -> OptimizeResult:
    """Mean-variance optimal portfolio: maximise  μ'w − λ·w'Σw.

    Args:
        risk_aversion: λ — trade-off coefficient.
            λ → 0:   pure IC maximiser (all weight on highest-IC factor).
            λ → ∞:   pure variance minimiser (approaches min-variance portfolio).
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result(f"Mean-Variance (λ={risk_aversion:.1f})", np.array([1.0]), mu, Sigma, names)

    def neg_utility(w: np.ndarray) -> float:
        return -(float(mu @ w) - risk_aversion * float(w @ Sigma @ w))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    best_res, best_val = None, np.inf
    for seed in [_equal_w(names)] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(
            neg_utility, seed, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val, best_res = res.fun, res

    if best_res is None or not best_res.success:
        return equal_weight(ic_dict)
    w = np.maximum(best_res.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result(f"Mean-Variance (λ={risk_aversion:.1f})", w, mu, Sigma, names)


def max_expected_return(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Maximise mean IC: concentrates all weight on the factor with the highest mean IC."""
    mu, Sigma, names = _align(ic_dict)
    w = np.zeros(len(names))
    w[int(np.argmax(mu))] = 1.0
    return _make_result("Max Expected Return", w, mu, Sigma, names)


def min_tracking_error(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Minimum tracking-error portfolio vs equal-weight benchmark.

    Minimises (w − w_eq)'Σ(w − w_eq) subject to w'μ ≥ w_eq'μ (at least equal-weight IC).
    Produces the tightest-tracking portfolio that does not sacrifice IC vs the equal-weight baseline.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Min Tracking Error", np.array([1.0]), mu, Sigma, names)

    w_bm = _equal_w(names)
    bm_ic = float(mu @ w_bm)

    def te_sq(w: np.ndarray) -> float:
        diff = w - w_bm
        return float(diff @ Sigma @ diff)

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: float(mu @ w) - bm_ic},
    ]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        te_sq, w_bm, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not res.success:
        return equal_weight(ic_dict)

    w = np.maximum(res.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Min Tracking Error", w, mu, Sigma, names)


def max_active_ir(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Maximise active Information Ratio vs equal-weight benchmark.

    Active IR = (w − w_eq)'μ / √((w − w_eq)'Σ(w − w_eq))
    Analogous to maximising the Sharpe of the *active* (benchmark-relative) IC.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Max Active IR", np.array([1.0]), mu, Sigma, names)

    w_bm = _equal_w(names)

    def neg_active_ir(w: np.ndarray) -> float:
        active = w - w_bm
        excess_ic = float(mu @ active)
        te_sq = float(active @ Sigma @ active)
        if te_sq <= 1e-24:
            return 0.0
        return -excess_ic / np.sqrt(te_sq)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    best_result, best_val = None, np.inf

    for seed_w in [_equal_w(names)] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(
            neg_active_ir, seed_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None or not best_result.success:
        return equal_weight(ic_dict)

    w = np.maximum(best_result.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Max Active IR", w, mu, Sigma, names)


def max_diversification(ic_dict: dict[str, pd.Series]) -> OptimizeResult:
    """Maximise the diversification ratio: Σ(w_i · σ_i) / √(w'Σw).

    σ_i = marginal IC volatility of factor i (sqrt of diagonal of Σ).
    Higher ratio means factor IC volatilities are less correlated — rewards combining
    factors whose IC series move independently.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        return _make_result("Max Diversification", np.array([1.0]), mu, Sigma, names)

    sigma_i = np.sqrt(np.diag(Sigma))

    def neg_dr(w: np.ndarray) -> float:
        weighted_vol = float(sigma_i @ w)
        port_var = float(w @ Sigma @ w)
        if port_var <= 1e-24:
            return 0.0
        return -weighted_vol / np.sqrt(port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    best_result, best_val = None, np.inf

    for seed_w in [_equal_w(names)] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(
            neg_dr, seed_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res

    if best_result is None or not best_result.success:
        return equal_weight(ic_dict)

    w = np.maximum(best_result.x, 0.0)
    w /= w.sum() if w.sum() > 1e-12 else 1.0
    return _make_result("Max Diversification", w, mu, Sigma, names)


def run_all(ic_dict: dict[str, pd.Series], risk_aversion: float = 1.0) -> list[OptimizeResult]:
    """Run all optimisers and return their results."""
    results = []
    for fn in [
        equal_weight,
        max_expected_return,
        ic_proportional,
        icir_proportional,
        max_icir,
        max_active_ir,
        min_variance,
        min_tracking_error,
        risk_parity,
        max_diversification,
    ]:
        try:
            results.append(fn(ic_dict))
        except Exception:
            pass
    try:
        results.append(mean_variance(ic_dict, risk_aversion=risk_aversion))
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    ic_dict: dict[str, pd.Series],
    n_points: int = 40,
) -> pd.DataFrame:
    """Trace the mean-variance frontier in IC space.

    For each target IC from μ_min to μ_max, find the minimum-variance portfolio
    that achieves that expected IC.

    Returns:
        DataFrame with columns: IC_mean, IC_std, ICIR, {name}_weight for each factor.
        Sorted by IC_mean ascending. Only feasible solutions included.
    """
    mu, Sigma, names = _align(ic_dict)
    n = len(names)

    if n == 1:
        ic_m, ic_s, icir = _portfolio_stats(np.array([1.0]), mu, Sigma)
        return pd.DataFrame([{
            "IC_mean": ic_m, "IC_std": ic_s, "ICIR": icir,
            f"{names[0]}_weight": 1.0,
        }])

    # Bounds for target IC: from minimum achievable to max_icir portfolio
    mu_lo = float(mu.min())
    mu_hi = float(mu.max())  # achievable since we can put all weight on one factor
    targets = np.linspace(mu_lo, mu_hi, n_points)

    rows = []
    prev_w = _equal_w(names)

    for target in targets:
        def port_var(w: np.ndarray) -> float:
            return float(w @ Sigma @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(mu @ w) - t},
        ]
        bounds = [(0.0, 1.0)] * n

        res = minimize(
            port_var, prev_w, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if not res.success:
            continue

        w = np.maximum(res.x, 0.0)
        total = w.sum()
        if total < 1e-12:
            continue
        w /= total
        prev_w = w.copy()  # warm-start next iteration

        ic_m, ic_s, icir = _portfolio_stats(w, mu, Sigma)
        row = {"IC_mean": ic_m, "IC_std": ic_s, "ICIR": icir}
        for i, name in enumerate(names):
            row[f"{name}_weight"] = float(w[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("IC_mean").reset_index(drop=True)
