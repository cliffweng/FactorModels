"""
Tutorial Factors
================
Five simple, heavily commented price-based factors — designed to teach
programmers how the factor system works.

Reading order (simplest → more involved):
    1. SMACross          — ratio of two rolling means
    2. UpDayRatio        — counting-based (boolean rolling mean)
    3. RollingSharpe     — combining mean & std (risk-adjusted momentum)
    4. ReturnSkewness    — higher-moment signal using rolling .skew()
    5. ShortTermZScore   — deviation-from-mean mean-reversion signal

Every factor follows the same two-method contract:

    compute(prices)        → pd.Series   one score per ticker, right now
    compute_panel(prices)  → pd.DataFrame  scores at every rebalance date

Both methods use the *same* mathematical formula; compute_panel() just
applies it to the entire price history in a single vectorised pass using
pandas .shift() and .rolling() — no Python loop over dates.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.factors.base import BaseFactor, register_factor


# ---------------------------------------------------------------------------
# 1. SMA Cross (50-day / 200-day)
# ---------------------------------------------------------------------------

@register_factor
class SMACross(BaseFactor):
    """
    SMA Cross: 50-day SMA ÷ 200-day SMA − 1.

    INTUITION
    ---------
    The classic "golden cross" signal made continuous and rankable.
    When the 50-day average price is above the 200-day average the stock
    is in an uptrend; when below, it's in a downtrend.  Expressing it as
    a ratio lets us rank all stocks against each other — not just a binary
    signal — so Q5 captures the strongest uptrends in the universe.

    WHY IT (MIGHT) WORK
    -------------------
    Trend-following exploits momentum and price stickiness.  Institutional
    rebalancing and stop-loss rules create feedback loops that sustain
    trends longer than the efficient-market hypothesis predicts.

    DIRECTION = +1
    → Higher ratio  (fast SMA >> slow SMA)  → stronger uptrend → long.
    → Lower  ratio  (fast SMA << slow SMA)  → downtrend        → short.
    """

    name      = "SMA_CROSS"
    label     = "SMA Cross (50/200)"
    description = "50-day SMA divided by 200-day SMA minus 1 — trend-following crossover"
    category  = "Momentum"
    direction = 1          # positive ratio = uptrend = expected outperformance
    formula        = "score = SMA(50) / SMA(200) − 1"
    academic_ref   = "Brock, Lakonishok & LeBaron (1992) — Simple Technical Trading Rules and the Stochastic Properties of Stock Returns"
    interpretation = (
        "**Positive score (golden cross)** — 50-day average above 200-day: the stock is in a "
        "sustained uptrend. The *magnitude* tells you how far the trend has extended: "
        "+0.10 means the fast average is 10% above the slow average — a strong trend. "
        "**Negative score (death cross)** — 50-day below 200-day: downtrend in place. "
        "Brock et al. (1992) provided early statistical evidence that simple moving-average "
        "rules generated significant excess returns on the DJIA from 1897–1986."
    )

    _FAST = 50             # days for the fast (responsive) moving average
    _SLOW = 200            # days for the slow (baseline) moving average

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        # We need at least _SLOW + 1 rows to compute a valid slow SMA.
        if len(prices) < self._SLOW + 1:
            return pd.Series(dtype=float)

        # rolling(N).mean() returns a DataFrame of the same shape as prices.
        # .iloc[-1] takes the most recent row — one SMA value per ticker.
        sma_fast = prices.rolling(self._FAST).mean().iloc[-1]   # shape: (n_tickers,)
        sma_slow = prices.rolling(self._SLOW).mean().iloc[-1]

        # Ratio minus 1:  0 means the two averages are equal (no trend).
        #                 +0.05 means fast SMA is 5 % above slow SMA (uptrend).
        #                 -0.05 means fast SMA is 5 % below slow SMA (downtrend).
        return (sma_fast / sma_slow - 1).dropna()

    def compute_panel(
        self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252
    ) -> pd.DataFrame:
        # Compute rolling means across the ENTIRE price history in one call.
        # Each row of the result is what compute() would return on that date.
        sma_fast = prices.rolling(self._FAST, min_periods=self._FAST).mean()
        sma_slow = prices.rolling(self._SLOW, min_periods=self._SLOW).mean()

        # Element-wise division across the (dates × tickers) DataFrame.
        panel = sma_fast / sma_slow - 1

        # Sample at rebalance frequency (e.g. month-end = "ME").
        # .resample().last() picks the last available value in each period.
        rebal = panel.resample(freq).last()

        # Drop leading all-NaN rows that exist before the SLOW window fills up.
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal.dropna(how="all")


# ---------------------------------------------------------------------------
# 2. Up-Day Ratio
# ---------------------------------------------------------------------------

@register_factor
class UpDayRatio(BaseFactor):
    """
    Up-Day Ratio: fraction of trading days with a positive return, trailing year.

    INTUITION
    ---------
    Two stocks can both be up 30 % over the past year.  One achieved it in
    a single gap-up and then went sideways; the other went up almost every
    day.  The up-day ratio rewards the consistent climber.

    This factor measures *trend quality* rather than raw return.  It is
    cheap to compute and easy to explain — a good sanity-check factor.

    CONSTRUCTION TRICK
    ------------------
    (daily_returns > 0) produces a DataFrame of True/False.
    pandas treats True as 1 and False as 0, so .rolling().mean() on a
    boolean DataFrame directly gives the rolling fraction of up-days.
    No explicit count() or sum() call is needed.

    DIRECTION = +1
    → More up-days → more consistent trend → expect continuation.
    """

    name      = "UP_DAY_252"
    label     = "Up-Day Ratio (252d)"
    description = "Fraction of positive-return days over the trailing 252 trading days"
    category  = "Momentum"
    direction = 1
    formula        = "score = #{days with daily_return > 0} / 252  over trailing year"
    academic_ref   = "Related to Grinblatt & Moskowitz (2004) — directional consistency measure"
    interpretation = (
        "**Score of 0.55** — 55% of trading days in the past year were positive: "
        "persistently more buyers than sellers. "
        "**Score near 0.50** — no sustained directional bias. "
        "**Score of 0.60+** — exceptionally consistent uptrend, typically seen in quality "
        "compounders with steady earnings growth. "
        "Unlike raw momentum (which can come from a single large day), a high up-day ratio "
        "indicates broad-based, recurring buying activity across the full year."
    )

    _WINDOW = 252   # trailing year

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)

        daily_returns = prices.pct_change()        # convert price → daily return

        # .tail() slices the last N rows across every column simultaneously.
        window = daily_returns.tail(self._WINDOW)

        # (window > 0) → boolean DataFrame.
        # .sum() counts True per column; divide by non-NaN count for the ratio.
        up_ratio = (window > 0).sum() / window.notna().sum()
        return up_ratio.dropna()

    def compute_panel(
        self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252
    ) -> pd.DataFrame:
        daily_returns = prices.pct_change()

        # (daily_returns > 0) is a boolean DataFrame (True=1, False=0).
        # rolling().mean() on a 0/1 series = rolling fraction of up-days.
        # This is equivalent to computing the ratio at each calendar date
        # without any Python loop.
        panel = (
            (daily_returns > 0)
            .rolling(self._WINDOW, min_periods=self._WINDOW)
            .mean()
        )

        rebal = panel.resample(freq).last()
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal.dropna(how="all")


# ---------------------------------------------------------------------------
# 3. Rolling Sharpe Ratio (252-day)
# ---------------------------------------------------------------------------

@register_factor
class RollingSharpe(BaseFactor):
    """
    Rolling Sharpe Ratio — trailing 252-day mean return ÷ volatility.

    INTUITION
    ---------
    Standard momentum (12-1) measures raw past return.  Two stocks with
    the same momentum score may have very different risk profiles: one
    achieved it with smooth, steady gains; the other with wild swings.
    The Sharpe ratio rewards the *smoother* outperformer.

    Formula:  Sharpe = mean(daily_ret) / std(daily_ret)

    Note: we omit the √252 annualisation multiplier because it is a
    constant that cancels out when ranking stocks cross-sectionally.

    GOTCHA — zero volatility
    ------------------------
    If a stock hasn't moved in 252 days (std ≈ 0) we get a division-by-zero.
    We handle this by replacing zero std with NaN so the ticker is dropped
    from the cross-section for that period.

    DIRECTION = +1
    → Higher Sharpe → smoother outperformance → expect continuation.
    """

    name      = "SHARPE_252"
    label     = "Rolling Sharpe (252d)"
    description = "252-day mean daily return divided by daily return volatility"
    category  = "Momentum"
    direction = 1
    formula        = "score = mean(daily_ret, 252d) / std(daily_ret, 252d)  (unnormalised Sharpe)"
    academic_ref   = "Sharpe (1966) — Mutual Fund Performance; applied cross-sectionally as a quality-momentum signal"
    interpretation = (
        "**High score** — the stock has delivered above-average returns *smoothly*: "
        "high reward relative to its own variability. "
        "Think of it as momentum quality: two stocks up 20% over the year score differently "
        "if one did it with 10% annualised vol (score ≈ 1.26) vs 30% vol (score ≈ 0.42). "
        "**Zero or negative score** — flat or declining stock. "
        "Note: the √252 annualisation factor is omitted because it cancels in cross-sectional ranking. "
        "The denominator is replaced with NaN if vol = 0 (stock never moved)."
    )

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)

        # pct_change(): price series → daily return series
        daily_returns = prices.pct_change()

        # Slice the last _WINDOW rows for each ticker.
        window = daily_returns.tail(self._WINDOW)

        mean_ret = window.mean()                          # avg daily return
        std_ret  = window.std().replace(0, float("nan")) # daily vol; NaN if flat

        return (mean_ret / std_ret).dropna()

    def compute_panel(
        self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252
    ) -> pd.DataFrame:
        daily_returns = prices.pct_change()

        # One rolling call per statistic — fully vectorised.
        roll_mean = daily_returns.rolling(self._WINDOW, min_periods=self._WINDOW).mean()
        roll_std  = daily_returns.rolling(self._WINDOW, min_periods=self._WINDOW).std()

        # Replace 0 std with NaN to avoid division-by-zero.
        roll_std  = roll_std.replace(0, float("nan"))

        panel = roll_mean / roll_std

        rebal = panel.resample(freq).last()
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal.dropna(how="all")


# ---------------------------------------------------------------------------
# 4. Return Skewness (60-day)
# ---------------------------------------------------------------------------

@register_factor
class ReturnSkewness(BaseFactor):
    """
    60-Day Return Skewness — the 'lottery stock' factor.

    INTUITION
    ---------
    Skewness measures asymmetry in the return distribution.
    A stock with *high positive skew* has many small losses and
    occasional giant gains — a "lottery ticket."  Investors irrationally
    overpay for this lottery-like exposure, driving the price up and
    future returns *down*.

    The academic result (Bali, Cakici & Whitelaw 2011):
        High positive skewness → overpriced → low future return.
        Low  / negative skewness → underpriced → higher future return.

    DIRECTION = -1
    → High skewness (lottery stock) → overpriced → goes in Q1 (short).
    → Low  skewness                 → cheaper    → goes in Q5 (long).
    The direction flag tells the backtest engine to flip Q1/Q5 so that
    Q5 is *low* skewness (our "long" side).

    PANDAS TIP
    ----------
    pandas has a built-in rolling().skew() method that computes the
    Fisher-Pearson standardised third moment efficiently — no need for
    rolling().apply(scipy.stats.skew), which would be much slower.
    """

    name      = "SKEW_60"
    label     = "Return Skewness (60d)"
    description = "60-day rolling skewness of daily returns — lottery-stock signal"
    category  = "Risk"
    direction = -1   # high skew = lottery = overpriced → direction inverted
    formula        = "score = E[(r − μ)³] / σ³  over trailing 60 trading days"
    academic_ref   = "Bali, Cakici & Whitelaw (2011) — Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns"
    interpretation = (
        "**High positive skew** — many small losses but occasional huge gains: a 'lottery ticket' stock. "
        "Investors overpay for the chance at a big payoff, pushing prices above fair value. "
        "High-skew stocks subsequently *underperform* (direction = −1: low-skew ranks in long book). "
        "**Symmetric or negative skew** — return distribution closer to normal or left-tailed; "
        "not over-priced for lottery-ticket appeal. "
        "Bali et al. found that the MAX statistic (highest single-day return over the month) "
        "captures similar information and is highly correlated with rolling skewness."
    )

    _WINDOW = 60

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)

        daily_returns = prices.pct_change()

        # pandas .skew() on a DataFrame computes column-wise skewness.
        # Fisher definition: 0 = symmetric, >0 = right-tailed, <0 = left-tailed.
        return daily_returns.tail(self._WINDOW).skew().dropna()

    def compute_panel(
        self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252
    ) -> pd.DataFrame:
        daily_returns = prices.pct_change()

        # rolling().skew() is natively vectorised in pandas — no .apply() needed.
        # min_periods=30 allows a score to appear before the full 60-day window fills,
        # but we use _WINDOW to be conservative.
        panel = daily_returns.rolling(self._WINDOW, min_periods=self._WINDOW).skew()

        rebal = panel.resample(freq).last()
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal.dropna(how="all")


# ---------------------------------------------------------------------------
# 5. Short-Term Price Z-Score (20-day)
# ---------------------------------------------------------------------------

@register_factor
class ShortTermZScore(BaseFactor):
    """
    Short-Term Price Z-Score — how far is today's price from its 20-day mean?

    Formula:  z = (price − SMA_20) / rolling_std_20

    INTUITION
    ---------
    A z-score of +2 means today's price is two standard deviations above
    its recent average — the stock is "overbought."  A z-score of -2
    means it is "oversold."  Mean-reversion theory says both extremes
    are temporary and prices will drift back toward the average.

    This is different from ShortTermReversal (which uses raw 1-week return):
      • The z-score normalises by volatility — a 5 % move means something
        different for a 15 % annualised vol stock vs. a 60 % vol stock.
      • Because we divide by std, high-vol and low-vol stocks are put on
        the same scale, making the cross-sectional ranking fairer.

    DIRECTION = -1
    → High  z-score → overbought → expect reversion downward → short.
    → Low   z-score → oversold   → expect recovery upward    → long.
    Setting direction = -1 ensures Q5 = most oversold = our long book.

    KEY DIFFERENCE vs SMACross (factor 1 above)
    --------------------------------------------
    SMACross uses a 200-day slow window and is a *trend-following* signal.
    ShortTermZScore uses a 20-day window and is a *mean-reversion* signal.
    They are often negatively correlated — combining them in a composite
    model provides meaningful diversification.
    """

    name      = "STZ_20"
    label     = "Short-Term Z-Score (20d)"
    description = "Price deviation from 20-day SMA normalized by 20-day price std — mean-reversion signal"
    category  = "Momentum"   # reversal lives in the Momentum category by convention
    direction = -1            # high z-score = overbought = expect fall
    formula        = "score = (price − SMA20) / rolling_std(price, 20d)"
    academic_ref   = "Poterba & Summers (1988) — Mean Reversion in Stock Prices: Evidence and Implications"
    interpretation = (
        "**High z-score (+2 or above)** — price is 2 standard deviations above its 20-day mean: "
        "overbought; mean-reversion theory predicts a pullback. direction=−1 → this ranks in the *short* book. "
        "**Low z-score (−2 or below)** — oversold: price has moved far below its recent average; "
        "bounce expected → ranks in the *long* book. "
        "Unlike Short-Term Reversal (which uses raw 1-week return), the z-score adjusts for "
        "volatility: a 5% move means more for a low-vol stock than a high-vol one."
    )

    _WINDOW = 20

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)

        # Rolling statistics computed over the full history, then .iloc[-1] for today.
        sma = prices.rolling(self._WINDOW).mean().iloc[-1]    # 20-day average price
        std = prices.rolling(self._WINDOW).std().iloc[-1]     # 20-day price std dev

        today = prices.iloc[-1]

        # Z-score formula.  std.replace(0, NaN) drops stocks that haven't moved.
        z = (today - sma) / std.replace(0, float("nan"))
        return z.dropna()

    def compute_panel(
        self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252
    ) -> pd.DataFrame:
        # Vectorised: compute rolling mean and std for every date simultaneously.
        sma = prices.rolling(self._WINDOW, min_periods=self._WINDOW).mean()
        std = prices.rolling(self._WINDOW, min_periods=self._WINDOW).std()

        # Replace zero std with NaN to prevent division-by-zero.
        std = std.replace(0, float("nan"))

        # Element-wise z-score across the full (dates × tickers) price DataFrame.
        # Each row of `panel` is what compute() would return if called on that date.
        panel = (prices - sma) / std

        # Resample to rebalance dates and strip the NaN-only warm-up period.
        rebal = panel.resample(freq).last()
        rebal = rebal[rebal.notna().any(axis=1).cumsum().gt(0)]
        return rebal.dropna(how="all")
