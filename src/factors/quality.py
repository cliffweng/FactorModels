"""Quality factors sourced from yfinance.info (snapshot only)."""
from __future__ import annotations

import pandas as pd

from src.factors.base import BaseFactor, register_factor


@register_factor
class ROEFactor(BaseFactor):
    name = "ROE"
    label = "Return on Equity (snapshot)"
    description = "ROE from yfinance.info — higher ROE indicates better capital efficiency"
    category = "Quality"
    direction = 1
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals: pd.DataFrame | None = kwargs.get("fundamentals")
        if fundamentals is None or "roe" not in fundamentals.columns:
            return pd.Series(dtype=float)
        return fundamentals["roe"].dropna().rename("ROE")


@register_factor
class GrossMarginFactor(BaseFactor):
    name = "GROSS_MARGIN"
    label = "Gross Margin (snapshot)"
    description = "Gross margin % from yfinance.info — higher margin → stronger competitive moat"
    category = "Quality"
    direction = 1
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals: pd.DataFrame | None = kwargs.get("fundamentals")
        if fundamentals is None or "gross_margins" not in fundamentals.columns:
            return pd.Series(dtype=float)
        return fundamentals["gross_margins"].dropna().rename("GROSS_MARGIN")
