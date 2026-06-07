"""Value factors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor


@register_factor
class FiftyTwoWeekHighRatio(BaseFactor):
    name = "52W_HIGH"
    label = "52-Week High Ratio"
    description = "Price / 52-week high — lower ratio may indicate undervaluation (George & Hwang 2004)"
    category = "Value"
    direction = -1

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        rolling_max = prices.tail(self._WINDOW).max()
        current = prices.iloc[-1]
        return (current / rolling_max).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252, **kwargs) -> pd.DataFrame:
        panel = prices / prices.rolling(self._WINDOW).max()
        rebal = panel.resample(freq).last()
        return rebal


# ---------------------------------------------------------------------------
# EDGAR helpers (shared by value.py and quality.py)
# ---------------------------------------------------------------------------

def _edgar_latest(edgar_panel: dict, field: str) -> pd.Series:
    """Most recent cross-sectional value for a field from the EDGAR panel."""
    raw = edgar_panel.get(field)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    return raw.ffill().iloc[-1].dropna()


def _edgar_compute_ratio_panel(
    edgar_panel: dict,
    field: str,
    prices: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Panel of (fundamental_value / price) at each rebalancing date.

    Used for B/P = BVPS/Price and E/P = EPS_TTM/Price.
    Forward-fills quarterly filing data to daily, then divides by contemporaneous price.
    This computes the ratio at the *rebalancing date price*, avoiding look-ahead bias.
    """
    raw = edgar_panel.get(field)
    if raw is None or raw.empty:
        return pd.DataFrame()

    common = raw.columns.intersection(prices.columns)
    if common.empty:
        return pd.DataFrame()

    all_dates = prices.index.union(raw.index).sort_values()
    aligned = raw[common].reindex(all_dates).ffill().reindex(prices.index)
    ratio = aligned.div(prices[common]).replace([np.inf, -np.inf], np.nan)

    rebal = ratio.resample(freq).last()
    return rebal.dropna(how="all")


def _edgar_compute_panel_raw(
    edgar_panel: dict,
    field: str,
    prices: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Forward-fill fundamental panel data to rebalancing dates (no price division).

    Used for ROE and Gross Margin which are already unit-less ratios.
    """
    raw = edgar_panel.get(field)
    if raw is None or raw.empty:
        return pd.DataFrame()

    all_dates = prices.index.union(raw.index).sort_values()
    aligned = raw.reindex(all_dates).ffill().reindex(prices.index)
    rebal = aligned.resample(freq).last()
    return rebal.dropna(how="all")


# ---------------------------------------------------------------------------
# Fundamental value factors
# ---------------------------------------------------------------------------

@register_factor
class PriceToBook(BaseFactor):
    """Book-to-Price ratio (B/P = BVPS / Price).

    Source: SEC EDGAR XBRL (StockholdersEquity / SharesOutstanding), point-in-time
    safe via filing date. Ratio computed at each rebalancing date using contemporaneous
    price — avoids look-ahead bias present in pre-computed P/B ratios.
    Falls back to yfinance.info snapshot (B/P from pb field) when no EDGAR panel.
    """

    name = "P_B"
    label = "Price-to-Book"
    description = "B/P (book-to-price) — higher = cheaper valuation. EDGAR XBRL quarterly data."
    category = "Value"
    direction = 1       # higher B/P = more value
    requires_edgar = True
    _edgar_field = "bvps"

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is not None:
            bvps = _edgar_latest(edgar_panel, self._edgar_field)
            price = prices.iloc[-1].reindex(bvps.index)
            bp = (bvps / price.replace(0, np.nan)).dropna()
            return bp[bp > 0].rename(self.name)

        fundamentals = kwargs.get("fundamentals")
        if fundamentals is not None and "pb" in fundamentals.columns:
            pb = fundamentals["pb"].dropna()
            pb = pb[pb > 0]
            return (1.0 / pb).rename(self.name)

        return pd.Series(dtype=float)

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252, **kwargs) -> pd.DataFrame:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is None:
            return pd.DataFrame()
        return _edgar_compute_ratio_panel(edgar_panel, self._edgar_field, prices, freq)


@register_factor
class PriceToEarnings(BaseFactor):
    """Earnings-to-Price ratio (E/P = EPS_TTM / Price).

    Source: SEC EDGAR XBRL (EarningsPerShareDiluted TTM), point-in-time safe via
    filing date. E/P computed at each rebalancing date using contemporaneous price.
    Falls back to yfinance.info snapshot when no EDGAR panel.
    """

    name = "P_E"
    label = "Price-to-Earnings"
    description = "E/P earnings yield — higher = cheaper. EDGAR XBRL TTM EPS data."
    category = "Value"
    direction = 1       # higher E/P = more value
    requires_edgar = True
    _edgar_field = "eps_ttm"

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is not None:
            eps = _edgar_latest(edgar_panel, self._edgar_field)
            price = prices.iloc[-1].reindex(eps.index)
            ep = (eps / price.replace(0, np.nan)).dropna()
            return ep[ep > 0].rename(self.name)

        fundamentals = kwargs.get("fundamentals")
        if fundamentals is not None and "pe" in fundamentals.columns:
            pe = fundamentals["pe"].dropna()
            pe = pe[pe > 0]
            return (1.0 / pe).rename(self.name)

        return pd.Series(dtype=float)

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252, **kwargs) -> pd.DataFrame:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is None:
            return pd.DataFrame()
        return _edgar_compute_ratio_panel(edgar_panel, self._edgar_field, prices, freq)
