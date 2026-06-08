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
    formula        = "score = shares_short / avg_daily_volume  (days to cover)"
    academic_ref   = "Asquith, Pathak & Ritter (2005) — Short Interest, Institutional Ownership, and Stock Returns"
    interpretation = (
        "**High score (many days to cover)** — it would take the entire short-seller "
        "community many days of average trading volume to exit their positions: a crowded, "
        "illiquid short that tends to underperform as it is hard to exit quickly. "
        "**Low score** — short sellers can exit easily; less short-seller-driven price pressure. "
        "A days-to-cover ratio above 10 is often considered 'heavily shorted'. "
        "Short sellers are typically informed: high short interest signals fundamental concerns "
        "identified by sophisticated investors through detailed company analysis."
    )

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
    formula        = "score = shares_short / float_shares"
    academic_ref   = "Dechow, Hutton, Meulbroek & Sloan (2001) — Short-Sellers, Fundamental Analysis and Stock Returns"
    interpretation = (
        "**>10% short of float** — heavily shorted; short sellers have collectively bet "
        "a significant fraction of the tradeable supply will fall. "
        "Dechow et al. showed short sellers are skilled fundamental analysts: stocks they target "
        "are genuinely overvalued and subsequently underperform. "
        "**<2% short of float** — minimal short interest; either little institutional bearish "
        "coverage, or the stock is difficult to borrow. "
        "**Squeeze risk**: very high short float can occasionally reverse dramatically when "
        "positive news forces short covering — a separate event risk not captured by this factor."
    )

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
    formula        = "score = (shares_short − shares_short_prior_month) / |shares_short_prior_month|"
    academic_ref   = "Desai, Ramesh, Thiagarajan & Balachandran (2002) — An Investigation of the Informational Role of Short Interest in the Nasdaq Market"
    interpretation = (
        "**Positive score (rising short interest)** — more shares are now sold short versus "
        "last month; growing bearish conviction from informed traders. "
        "Desai et al. found that *increases* in short interest predict negative returns "
        "even more reliably than the level of short interest. "
        "**Negative score (short covering)** — short sellers are buying back shares to exit "
        "positions; can be a near-term bullish catalyst as covering demand supports the price. "
        "A score of +0.20 means short interest grew 20% month-over-month — a significant escalation."
    )

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
