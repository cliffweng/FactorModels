"""Information Coefficient (IC) analysis.

IC = Spearman rank correlation between factor scores at time t
     and forward returns from t → t+horizon.

Key convention:
  - factor_panel rows are rebalance dates (e.g. month-end)
  - fwd_returns are pre-shifted so that fwd_returns[t] = return from t to t+1 period
  - We join on the rebalance date index to avoid any look-ahead bias
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core IC computation
# ---------------------------------------------------------------------------

def compute_forward_returns(
    daily_returns: pd.DataFrame,
    horizon_days: int = 21,
    rebal_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Compute horizon-day forward returns aligned to rebalance dates.

    Returns DataFrame with same index as rebal_dates.
    forward_return[t] = cumulative return from t+1 to t+horizon.
    """
    # Compound daily returns over the horizon window
    cum = (1 + daily_returns).cumprod()
    # Forward return at each day = cum[t+h] / cum[t] - 1
    fwd = cum.shift(-horizon_days) / cum - 1

    if rebal_dates is not None:
        # Align to rebalance dates (nearest available date before or on rebal date)
        aligned = fwd.reindex(rebal_dates, method="pad")
        return aligned
    return fwd


def compute_ic(
    factor_scores: pd.Series,
    fwd_returns: pd.Series,
    min_obs: int = 10,
) -> float:
    """Spearman IC for one cross-section.

    Args:
        factor_scores: Cross-section of factor values at time t.
        fwd_returns:   Cross-section of forward returns at time t.
        min_obs:       Minimum number of valid pairs.

    Returns:
        IC value ∈ [-1, 1] or NaN if insufficient data.
    """
    combined = pd.concat([factor_scores, fwd_returns], axis=1).dropna()
    combined.columns = ["factor", "fwd"]
    if len(combined) < min_obs:
        return float("nan")
    rho, _ = stats.spearmanr(combined["factor"], combined["fwd"])
    return float(rho)


def compute_ic_series(
    factor_panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    horizon_days: int = 21,
    min_obs: int = 10,
) -> pd.Series:
    """Compute IC at every rebalance date in factor_panel.

    Args:
        factor_panel:  (rebal_dates × tickers) factor scores.
        daily_returns: (dates × tickers) daily return DataFrame.
        horizon_days:  Forward return horizon in trading days (default 21 ≈ 1 month).
        min_obs:       Min cross-sectional observations per date.

    Returns:
        pd.Series of IC values indexed by rebalance date.
    """
    fwd = compute_forward_returns(
        daily_returns, horizon_days=horizon_days, rebal_dates=factor_panel.index
    )

    ics = {}
    for date in factor_panel.index:
        if date not in fwd.index:
            continue
        scores = factor_panel.loc[date].dropna()
        returns = fwd.loc[date].dropna()
        ics[date] = compute_ic(scores, returns, min_obs=min_obs)

    return pd.Series(ics, name="IC").dropna()


def compute_rolling_ic(ic_series: pd.Series, window: int = 12) -> pd.Series:
    """Rolling mean IC over a window of rebalance periods."""
    return ic_series.rolling(window, min_periods=max(1, window // 2)).mean()


def compute_icir(ic_series: pd.Series) -> float:
    """Information Coefficient Information Ratio = mean(IC) / std(IC)."""
    std = ic_series.std()
    if np.isnan(std) or std < 1e-10:
        return float("nan")
    return float(ic_series.mean() / std)


def compute_ic_decay(
    factor_panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    horizons: list[int] | None = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """IC at multiple forward horizons to show how predictive power decays.

    Args:
        horizons: List of forward horizons in trading days.
                  Default: [1, 5, 10, 21, 42, 63, 126]

    Returns:
        DataFrame with index = horizon (days) and columns = ['IC_mean', 'ICIR', 'IC_std', 't_stat'].
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63, 126]

    rows = []
    for h in horizons:
        ic_s = compute_ic_series(factor_panel, daily_returns, horizon_days=h, min_obs=min_obs)
        if len(ic_s) < 3:
            rows.append({"horizon_days": h, "IC_mean": np.nan, "IC_std": np.nan, "ICIR": np.nan, "t_stat": np.nan})
            continue
        ic_mean = ic_s.mean()
        ic_std = ic_s.std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_s))) if ic_std > 0 else np.nan
        rows.append(
            {
                "horizon_days": h,
                "IC_mean": ic_mean,
                "IC_std": ic_std,
                "ICIR": icir,
                "t_stat": t_stat,
            }
        )

    return pd.DataFrame(rows).set_index("horizon_days")
