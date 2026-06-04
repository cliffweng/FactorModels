"""Value factors: price-based proxy and fundamental P/B, P/E."""
from __future__ import annotations

import pandas as pd

from src.factors.base import BaseFactor, register_factor


@register_factor
class FiftyTwoWeekHighRatio(BaseFactor):
    name = "52W_HIGH"
    label = "52-Week High Ratio"
    description = "Price / 52-week high — lower ratio may indicate undervaluation (George & Hwang 2004)"
    category = "Value"
    direction = -1   # farther from high → potential value opportunity

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        rolling_max = prices.tail(self._WINDOW).max()
        current = prices.iloc[-1]
        return (current / rolling_max).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = prices / prices.rolling(self._WINDOW).max()
        rebal = panel.resample(freq).last()
        return rebal


@register_factor
class PriceToBook(BaseFactor):
    name = "P_B"
    label = "Price-to-Book (snapshot)"
    description = "Inverse P/B ratio from yfinance.info — higher B/P = cheaper valuation"
    category = "Value"
    direction = 1   # higher 1/PB → more value
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals: pd.DataFrame | None = kwargs.get("fundamentals")
        if fundamentals is None or "pb" not in fundamentals.columns:
            return pd.Series(dtype=float)
        # Return inverse P/B = B/P; higher → cheaper
        pb = fundamentals["pb"].dropna()
        pb = pb[pb > 0]
        return (1.0 / pb).rename("P_B")


@register_factor
class PriceToEarnings(BaseFactor):
    name = "P_E"
    label = "Price-to-Earnings (snapshot)"
    description = "Inverse trailing P/E from yfinance.info — higher E/P = more earnings yield"
    category = "Value"
    direction = 1   # higher 1/PE → more value
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals: pd.DataFrame | None = kwargs.get("fundamentals")
        if fundamentals is None or "pe" not in fundamentals.columns:
            return pd.Series(dtype=float)
        pe = fundamentals["pe"].dropna()
        pe = pe[pe > 0]
        return (1.0 / pe).rename("P_E")
