"""Risk-based factors: realized volatility and market beta."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor

_ANNUALIZE = np.sqrt(252)


@register_factor
class RealizedVol60(BaseFactor):
    name = "RVOL_60"
    label = "Realized Volatility (60d)"
    description = "Annualized 60-day realized return volatility — lower is better (Low-Vol anomaly)"
    category = "Risk"
    direction = -1   # lower vol → higher expected risk-adjusted return (Low-Vol factor)

    _WINDOW = 60

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        rets = prices.pct_change()
        if len(rets.dropna()) < self._WINDOW:
            return pd.Series(dtype=float)
        vol = rets.tail(self._WINDOW).std() * _ANNUALIZE
        return vol.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        panel = rets.rolling(self._WINDOW).std() * _ANNUALIZE
        rebal = panel.resample(freq).last()
        return rebal


@register_factor
class Beta252(BaseFactor):
    name = "BETA_252"
    label = "Market Beta (252d)"
    description = "252-day rolling beta vs SPY — deviation from 1.0 used as factor score"
    category = "Risk"
    direction = -1   # low beta → lower systematic risk

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if "SPY" not in prices.columns:
            return pd.Series(dtype=float)
        rets = prices.pct_change().tail(self._WINDOW).dropna()
        if len(rets) < self._WINDOW // 2:
            return pd.Series(dtype=float)
        market = rets["SPY"]
        mkt_var = market.var()
        if mkt_var == 0:
            return pd.Series(dtype=float)
        betas = {}
        for col in rets.columns:
            if col == "SPY":
                continue
            cov = rets[col].cov(market)
            betas[col] = cov / mkt_var
        return pd.Series(betas).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        if "SPY" not in prices.columns:
            return pd.DataFrame()
        rets = prices.pct_change()
        market = rets["SPY"]

        results = {}
        rebal_dates = rets.resample(freq).last().index
        for date in rebal_dates:
            window_rets = rets.loc[:date].tail(self._WINDOW).dropna(how="all")
            if len(window_rets) < self._WINDOW // 2:
                continue
            mkt_w = market.loc[window_rets.index]
            mkt_var = mkt_w.var()
            if mkt_var == 0:
                continue
            betas = {}
            for col in window_rets.columns:
                if col == "SPY":
                    continue
                cov = window_rets[col].cov(mkt_w)
                betas[col] = cov / mkt_var
            results[date] = pd.Series(betas)

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).T
