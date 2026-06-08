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
    formula        = "score = Net Income TTM / Stockholders' Equity"
    academic_ref   = "Haugen & Baker (1996); Novy-Marx (2013) — The Other Side of Value"
    interpretation = (
        "**High score** — management is generating a large profit from every dollar of "
        "shareholders' equity; a hallmark of durable competitive advantages (moats). "
        "ROE > 15% is often considered high quality; ROE > 30% suggests exceptional returns "
        "on capital (e.g. asset-light businesses with pricing power). "
        "**Low or negative ROE** — capital is being consumed or destroyed; typically "
        "predicts underperformance. EDGAR TTM data is point-in-time safe."
    )

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
    formula        = "score = (Revenue − COGS) / Revenue  (gross profit margin)"
    academic_ref   = "Novy-Marx (2013) — The Other Side of Value: The Gross Profitability Premium"
    interpretation = (
        "**High score** — business retains a large fraction of revenue after direct production "
        "costs; signals pricing power, brand strength, or technological differentiation. "
        "Novy-Marx (2013) showed gross profitability has predictive power comparable to B/P "
        "and they are nearly uncorrelated — combining them in a composite adds real alpha. "
        "**Low score** — commodity-like business with thin margins, vulnerable to input cost "
        "increases. Gross margin is more stable than net margin (less affected by financing "
        "or accounting choices), making it a cleaner quality signal."
    )

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
