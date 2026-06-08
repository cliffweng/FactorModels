"""Short-interest factors — snapshot signals from yfinance short-sale data.

Academic basis: heavily-shorted stocks systematically underperform (Asquith,
Pathak & Ritter 2005; Dechow et al. 2001).  All three factors carry direction=-1
so that *lower* short pressure ranks in the long book.

All three are snapshot-only (requires_fundamentals=True); no historical panel
is available from yfinance, so they appear in Factor Lab and Multi-Factor Model
cross-section scoring but not in IC/Backtest time-series analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor


# ---------------------------------------------------------------------------
# 1. Short Interest Ratio  (Days to Cover)
# ---------------------------------------------------------------------------

@register_factor
class ShortInterestRatio(BaseFactor):
    """Days-to-cover ratio: shares short ÷ average daily volume.

    A high ratio means it would take many days of average volume for all short
    sellers to buy back their positions — a crowded, illiquid short.
    Academic research finds crowded shorts underperform; this factor is
    inverted so that low-crowding stocks rank highest.

    Source: yfinance.info  shortRatio  (snapshot).
    """

    name               = "SHORT_RATIO"
    label              = "Short Interest Ratio"
    description        = "Days-to-cover (shares short ÷ avg daily vol) — lower = less crowded short"
    category           = "Risk"
    direction          = -1    # higher days-to-cover → more crowded → expect underperformance
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None or "short_ratio" not in fundamentals.columns:
            return pd.Series(dtype=float)
        s = fundamentals["short_ratio"].dropna()
        return s[s > 0].rename(self.name)

    # compute_panel intentionally not overridden — snapshot only.


# ---------------------------------------------------------------------------
# 2. Short Percent of Float
# ---------------------------------------------------------------------------

@register_factor
class ShortPercentFloat(BaseFactor):
    """Short interest expressed as a percentage of the free float.

    Captures how much of the tradeable supply has been sold short.
    Stocks with a high short-float % are crowded and face elevated squeeze
    risk *and* persistent selling pressure.  Lower values indicate less
    speculative bearish positioning.

    Source: yfinance.info  shortPercentOfFloat  (snapshot).
    """

    name               = "SHORT_PCT_FLOAT"
    label              = "Short % of Float"
    description        = "Shares short as % of float — lower = less short-seller pressure"
    category           = "Risk"
    direction          = -1    # high short % → crowded → underperforms
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None or "short_pct_float" not in fundamentals.columns:
            return pd.Series(dtype=float)
        s = fundamentals["short_pct_float"].dropna()
        return s[s >= 0].rename(self.name)


# ---------------------------------------------------------------------------
# 3. Short Interest Change (Month-over-Month)
# ---------------------------------------------------------------------------

@register_factor
class ShortInterestChange(BaseFactor):
    """Month-over-month change in short interest as a fraction of prior short interest.

    Rising short interest signals growing bearish conviction from informed
    sellers and predicts further underperformance.  Falling short interest
    (short covering) can be bullish, but this factor treats it conservatively
    as a relief-of-pressure signal rather than a momentum squeeze.

    Score = (shares_short − shares_short_prior) / |shares_short_prior|

    Source: yfinance.info  sharesShort / sharesShortPriorMonth  (snapshot).
    """

    name               = "SHORT_CHNG"
    label              = "Short Interest Change (MoM)"
    description        = "Month-over-month % change in shares short — lower = shorts being covered"
    category           = "Risk"
    direction          = -1    # increasing short interest → bearish → underperforms
    requires_fundamentals = True

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None:
            return pd.Series(dtype=float)
        needed = {"shares_short", "shares_short_prior"}
        if not needed.issubset(fundamentals.columns):
            return pd.Series(dtype=float)

        cur   = fundamentals["shares_short"].dropna()
        prior = fundamentals["shares_short_prior"].reindex(cur.index).dropna()
        common = cur.index.intersection(prior.index)
        if common.empty:
            return pd.Series(dtype=float)

        cur, prior = cur.loc[common], prior.loc[common]
        prior_abs = prior.abs().replace(0, np.nan)
        change = (cur - prior) / prior_abs
        return change.dropna().rename(self.name)
