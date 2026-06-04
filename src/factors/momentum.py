"""Price momentum and reversal factors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor


@register_factor
class Momentum12_1(BaseFactor):
    name = "MOM_12_1"
    label = "Momentum 12-1"
    description = "12-month price return skipping last month (Jegadeesh & Titman 1993)"
    category = "Momentum"
    direction = 1

    # Trading-day lookbacks
    _LONG = 252   # ~12 months
    _SKIP = 21    # ~1 month

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._LONG + 1:
            return pd.Series(dtype=float)
        end = prices.iloc[-self._SKIP]      # price 1 month ago
        start = prices.iloc[-self._LONG]    # price 12 months ago
        scores = end / start - 1
        return scores.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        # Vectorized: shift prices by skip and compute rolling return
        p_end = prices.shift(self._SKIP)    # price as of 1-month-ago at each date
        p_start = prices.shift(self._LONG)  # price as of 12-months-ago at each date
        panel = p_end / p_start - 1

        # Resample to rebalance dates
        rebal = panel.resample(freq).last()
        # Drop early rows before first valid score
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal


@register_factor
class Momentum6_1(BaseFactor):
    name = "MOM_6_1"
    label = "Momentum 6-1"
    description = "6-month price return skipping last month"
    category = "Momentum"
    direction = 1

    _LONG = 126
    _SKIP = 21

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._LONG + 1:
            return pd.Series(dtype=float)
        end = prices.iloc[-self._SKIP]
        start = prices.iloc[-self._LONG]
        return (end / start - 1).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        p_end = prices.shift(self._SKIP)
        p_start = prices.shift(self._LONG)
        panel = p_end / p_start - 1
        rebal = panel.resample(freq).last()
        return rebal


@register_factor
class ShortTermReversal(BaseFactor):
    name = "STR"
    label = "Short-Term Reversal"
    description = "Negative 1-week return (De Bondt & Thaler reversal)"
    category = "Momentum"
    direction = 1   # higher = more reversal expected (score is -1 * past_return)

    _LOOKBACK = 5   # trading days

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._LOOKBACK + 1:
            return pd.Series(dtype=float)
        ret = prices.iloc[-1] / prices.iloc[-self._LOOKBACK] - 1
        return (-ret).dropna()   # negated: losers get high score

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = -(prices / prices.shift(self._LOOKBACK) - 1)
        rebal = panel.resample(freq).last()
        return rebal
