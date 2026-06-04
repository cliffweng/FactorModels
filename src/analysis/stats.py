"""Shared statistical helpers for performance analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def annualize_return(daily_returns: pd.Series) -> float:
    total = (1 + daily_returns).prod()
    n = len(daily_returns)
    if n == 0:
        return float("nan")
    return float(total ** (_TRADING_DAYS / n) - 1)


def annualize_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(_TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    mu = annualize_return(daily_returns) - rf
    sigma = annualize_vol(daily_returns)
    if sigma == 0:
        return float("nan")
    return mu / sigma


def max_drawdown(cumulative: pd.Series) -> float:
    roll_max = cumulative.cummax()
    dd = (cumulative - roll_max) / roll_max
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    ann_ret = annualize_return(daily_returns)
    cum = (1 + daily_returns).cumprod()
    mdd = abs(max_drawdown(cum))
    if mdd == 0:
        return float("nan")
    return ann_ret / mdd


def drawdown_series(cumulative: pd.Series) -> pd.Series:
    roll_max = cumulative.cummax()
    return (cumulative - roll_max) / roll_max


def rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    roll_mu = daily_returns.rolling(window).mean() * _TRADING_DAYS
    roll_vol = daily_returns.rolling(window).std() * np.sqrt(_TRADING_DAYS)
    return roll_mu / roll_vol


def summary_stats(daily_returns: pd.Series, name: str = "") -> dict:
    cum = (1 + daily_returns).cumprod()
    return {
        "Name": name,
        "Ann. Return": f"{annualize_return(daily_returns):.1%}",
        "Ann. Vol": f"{annualize_vol(daily_returns):.1%}",
        "Sharpe": f"{sharpe_ratio(daily_returns):.2f}",
        "Max DD": f"{max_drawdown(cum):.1%}",
        "Calmar": f"{calmar_ratio(daily_returns):.2f}",
        "Observations": len(daily_returns),
    }
