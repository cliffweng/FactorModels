"""Price momentum and reversal factors.

Academic foundation
-------------------
Jegadeesh & Titman (1993) documented that stocks with high returns over the
past 3–12 months continue to outperform over the next 3–12 months — one of
the most robust anomalies in empirical asset pricing.  The standard
interpretation is that information diffuses slowly into prices, so past
winners continue to be under-priced relative to their fundamentals.

All three factors in this file operate on adjusted close prices only and
implement `compute_panel()` with vectorised pandas operations (shift + resample)
so that the full cross-section can be scored at every rebalance date in a single
pass — no Python loops over dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor


@register_factor
class Momentum12_1(BaseFactor):
    """12-month price momentum skipping the most recent month.

    Construction (Jegadeesh & Titman 1993)
    ----------------------------------------
    Rank stocks by their return from t−252 to t−21 (i.e. approximately
    12 months ago to 1 month ago).  The most recent month is *skipped*
    for two reasons:
        1. Short-term reversal: the 1-month return is negatively auto-correlated
           (see ShortTermReversal), so including it would partially cancel the
           momentum signal.
        2. Bid-ask bounce: micro-structure noise in the most recent price
           inflates measured returns for illiquid names.

    Lookback constants
    ------------------
    _LONG = 252  ≈ 12 calendar months of trading days  (the formation window start)
    _SKIP = 21   ≈  1 calendar month  of trading days  (the skip period)

    Formula
    -------
        score = price[t − _SKIP] / price[t − _LONG] − 1

    direction = +1  (higher past return → expect continuation → long book)
    """

    name      = "MOM_12_1"
    label     = "Momentum 12-1"
    description = "12-month price return skipping last month (Jegadeesh & Titman 1993)"
    category  = "Momentum"
    direction = 1
    formula        = "return = price(t−21) / price(t−252) − 1"
    academic_ref   = "Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers"
    interpretation = (
        "**High score** — stock has delivered a strong 11-month return with a 1-month skip; "
        "momentum is expected to continue over the next 3–12 months. "
        "**Low score** — recent loser; academic evidence suggests further underperformance. "
        "One of the most replicated anomalies in asset pricing, documented across 40+ markets."
    )

    # Trading-day lookbacks
    _LONG = 252   # ~12 months: start of the return measurement window
    _SKIP = 21    # ~1 month:  end of the window (we stop here, not at today)

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional momentum score at the most recent available date.

        We need at least _LONG + 1 rows so that both the start price (row
        −_LONG) and the end price (row −_SKIP) exist.

        iloc[-_SKIP] is the close price approximately 1 month ago.
        iloc[-_LONG] is the close price approximately 12 months ago.
        Their ratio minus 1 is the 11-month return with a 1-month skip.
        """
        if len(prices) < self._LONG + 1:
            return pd.Series(dtype=float)

        # Price as of 1 month ago — the "end" of the formation period
        end   = prices.iloc[-self._SKIP]
        # Price as of 12 months ago — the "start" of the formation period
        start = prices.iloc[-self._LONG]

        # Simple return over the formation window; NaN if either price is missing
        scores = end / start - 1
        return scores.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        """Rolling panel of momentum scores sampled at rebalance dates.

        Instead of looping over dates, we shift the entire price DataFrame:
            • p_end   = prices.shift(_SKIP)   — "price 1 month ago" at every date
            • p_start = prices.shift(_LONG)   — "price 12 months ago" at every date

        At each row t, p_end[t] == prices[t − _SKIP] and
        p_start[t] == prices[t − _LONG], so the ratio p_end/p_start − 1
        equals the score that compute() would return if called on prices[:t].

        resample(freq).last() down-samples to month-end (or whichever
        rebalance frequency is requested), keeping only the last available
        value in each period.
        """
        # Align the numerator and denominator by shifting
        p_end   = prices.shift(self._SKIP)    # price 1 month ago at each row
        p_start = prices.shift(self._LONG)    # price 12 months ago at each row
        panel   = p_end / p_start - 1

        # Resample to rebalance dates; drop all-NaN rows in the warm-up period
        rebal = panel.resample(freq).last()
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal


@register_factor
class Momentum6_1(BaseFactor):
    """6-month price momentum skipping the most recent month.

    Identical construction to Momentum12_1 but uses a 6-month formation
    window (≈ 126 trading days) instead of 12 months.  The 1-month skip
    is retained for the same micro-structure reasons.

    Shorter formation window
    -------------------------
    The 6-1 variant responds more quickly to changes in trend direction.
    It tends to be more correlated with recent price momentum signals (RSI,
    MACD) than the 12-1 factor, but it also turns over faster and may incur
    higher transaction costs in live portfolios.

    Empirically, both 6-1 and 12-1 momentum have positive IC (Fama & French
    1996), and combining them in a composite model provides modest
    diversification because the formation windows only partially overlap.

    direction = +1  (higher 6-month return → expect continuation → long book)
    """

    name      = "MOM_6_1"
    label     = "Momentum 6-1"
    description = "6-month price return skipping last month"
    category  = "Momentum"
    direction = 1
    formula        = "return = price(t−21) / price(t−126) − 1"
    academic_ref   = "Jegadeesh & Titman (1993); Fama & French (1996)"
    interpretation = (
        "**High score** — strong 6-month return with 1-month skip; shorter formation window "
        "makes this more reactive than the 12-1 factor but also increases portfolio turnover. "
        "**Combining 6-1 and 12-1** in a composite provides modest diversification because "
        "the formation windows only partially overlap (~5 months in common)."
    )

    _LONG = 126   # ~6 calendar months
    _SKIP = 21    # ~1 calendar month (same skip as 12-1)

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional 6-1 momentum at the most recent date.

        Logic mirrors Momentum12_1.compute(); only _LONG differs (126 vs 252).
        Requires _LONG + 1 rows so that both anchor prices are available.
        """
        if len(prices) < self._LONG + 1:
            return pd.Series(dtype=float)

        # Price at the end of the formation window (1 month ago)
        end   = prices.iloc[-self._SKIP]
        # Price at the start of the formation window (6 months ago)
        start = prices.iloc[-self._LONG]
        return (end / start - 1).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        """Rolling 6-1 momentum panel, vectorised via shift().

        Same approach as Momentum12_1.compute_panel() with a 126-day shift
        for the start price instead of 252 days.  Because the window is
        shorter, valid scores appear roughly 6 months into the price history
        rather than 12 months — the panel will have fewer NaN rows at the top.
        """
        p_end   = prices.shift(self._SKIP)
        p_start = prices.shift(self._LONG)
        panel   = p_end / p_start - 1

        rebal = panel.resample(freq).last()
        return rebal


@register_factor
class ShortTermReversal(BaseFactor):
    """Negative 1-week price return — the short-term reversal factor.

    Academic basis (De Bondt & Thaler 1985; Jegadeesh 1990)
    ---------------------------------------------------------
    At very short horizons (days to a few weeks), stock returns exhibit
    *negative* autocorrelation: last week's losers tend to outperform next
    week and vice versa.  The mechanism is primarily micro-structural:
        • Market makers widen bid-ask spreads after large order flow, and
          prices revert as liquidity is restored.
        • Institutional investors may temporarily push prices beyond fair
          value when executing large block trades; the subsequent reversion
          is the return to fundamental value.

    Construction
    ------------
    We compute the 5-day return and *negate* it, so that last week's losers
    receive a *high* score and last week's winners receive a *low* score.
    With direction = +1 the long book therefore holds recent losers and the
    short book holds recent winners — a classic mean-reversion strategy.

    _LOOKBACK = 5  ≈ 1 trading week

    Formula
    -------
        score = −(price[t] / price[t−5] − 1)

    direction = +1  (high score = recent loser = expect reversal upward)

    Note: short-term reversal and intermediate momentum (12-1, 6-1) are
    negatively correlated.  Including both in a composite model is a common
    way to hedge the momentum factor's exposure to sudden market reversals.
    """

    name      = "STR"
    label     = "Short-Term Reversal"
    description = "Negative 1-week return (De Bondt & Thaler reversal)"
    category  = "Momentum"
    direction = 1   # score is −return: high score = recent loser = expected winner
    formula        = "score = −(price(t) / price(t−5) − 1)"
    academic_ref   = "Jegadeesh (1990); Lo & MacKinlay (1990) — When Are Contrarian Profits Due to Stock Market Overreaction?"
    interpretation = (
        "**High score** — stock fell sharply last week; bid-ask bounce and liquidity restoration "
        "typically push prices back up over the next few days. "
        "**Low score** — recent winner; micro-structure noise may have inflated the price, "
        "and a small reversion is likely. "
        "Negatively correlated with intermediate momentum (12-1, 6-1) — "
        "useful as a hedge against momentum crash risk."
    )

    _LOOKBACK = 5   # 1 trading week

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional reversal score at the most recent date.

        Compute the 5-day return and negate it.
        A stock that fell −3 % last week gets a score of +0.03 (positive,
        hence ranked high in the long book).
        A stock that rose +3 % last week gets a score of −0.03 (negative,
        ranked in the short book).
        """
        if len(prices) < self._LOOKBACK + 1:
            return pd.Series(dtype=float)

        # 5-day simple return: (price today) / (price 5 days ago) − 1
        ret = prices.iloc[-1] / prices.iloc[-self._LOOKBACK] - 1

        # Negate so recent losers score high (expected to revert upward)
        return (-ret).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        """Rolling reversal panel, vectorised via shift().

        prices.shift(_LOOKBACK) gives, at each row t, the price from 5 days
        earlier.  The element-wise division then gives the 5-day return at
        every date for every ticker simultaneously.  Negating converts it to a
        reversal score.

        Note: resampling a reversal factor to month-end captures the 5-day
        return ending at the last trading day of each month — not the average
        5-day return during the month.  This is the standard approach for
        consistency with other monthly panel factors.
        """
        # Element-wise 5-day return across the full price matrix
        five_day_ret = prices / prices.shift(self._LOOKBACK) - 1

        # Negate: reversal score = −recent_return
        panel = -five_day_ret
        rebal = panel.resample(freq).last()
        return rebal
