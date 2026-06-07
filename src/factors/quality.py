"""Quality factors."""
from __future__ import annotations

import pandas as pd

from src.factors.base import BaseFactor, register_factor
from src.factors.value import _edgar_latest, _edgar_compute_panel_raw


@register_factor
class ROEFactor(BaseFactor):
    """Return on Equity.

    Source: SEC EDGAR XBRL (TTM NetIncomeLoss / StockholdersEquity), point-in-time
    safe via filing date. Falls back to yfinance.info snapshot without EDGAR panel.
    """

    name = "ROE"
    label = "Return on Equity"
    description = "ROE — higher = better capital efficiency. EDGAR XBRL TTM data."
    category = "Quality"
    direction = 1
    requires_edgar = True
    _edgar_field = "roe"

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is not None:
            return _edgar_latest(edgar_panel, self._edgar_field).rename(self.name)

        fundamentals = kwargs.get("fundamentals")
        if fundamentals is not None and "roe" in fundamentals.columns:
            return fundamentals["roe"].dropna().rename(self.name)

        return pd.Series(dtype=float)

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252, **kwargs) -> pd.DataFrame:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is None:
            return pd.DataFrame()
        return _edgar_compute_panel_raw(edgar_panel, self._edgar_field, prices, freq)


@register_factor
class GrossMarginFactor(BaseFactor):
    """Gross Profit Margin.

    Source: SEC EDGAR XBRL (TTM GrossProfit / Revenue), point-in-time safe via
    filing date. Falls back to yfinance.info snapshot without EDGAR panel.
    """

    name = "GROSS_MARGIN"
    label = "Gross Margin"
    description = "Gross margin % — higher = stronger competitive moat. EDGAR XBRL TTM data."
    category = "Quality"
    direction = 1
    requires_edgar = True
    _edgar_field = "gross_margin"

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is not None:
            return _edgar_latest(edgar_panel, self._edgar_field).rename(self.name)

        fundamentals = kwargs.get("fundamentals")
        if fundamentals is not None and "gross_margins" in fundamentals.columns:
            return fundamentals["gross_margins"].dropna().rename(self.name)

        return pd.Series(dtype=float)

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252, **kwargs) -> pd.DataFrame:
        edgar_panel = kwargs.get("edgar_panel")
        if edgar_panel is None:
            return pd.DataFrame()
        return _edgar_compute_panel_raw(edgar_panel, self._edgar_field, prices, freq)
