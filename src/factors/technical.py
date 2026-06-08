"""Technical indicator factors — RSI, MACD, Bollinger, CCI, and more.

All computation is done with vectorised pandas/numpy so that a full
cross-section of hundreds of tickers is calculated in one pass.

The `ta` library (pip install ta) provides identical indicators for
single-ticker Series.  Each private helper below includes a comment
showing the `ta` one-liner equivalent so you can cross-check values
or swap implementations for a single ticker.

ta docs: https://technical-analysis-library-in-python.readthedocs.io/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor

# Annualisation multiplier: daily std → annual std (√252 trading days/year)
_ANNUALIZE = np.sqrt(252)


# ===========================================================================
# Private helpers
#
# Each helper accepts a *prices* DataFrame (dates × tickers) and returns a
# DataFrame of the same shape containing the indicator value at every date.
# This is more efficient than calling a per-ticker function in a loop because
# all EWM/rolling operations are vectorised across columns simultaneously.
#
# `ta` equivalent shown for single-column (single-ticker) usage.
# To apply `ta` to a full DataFrame, iterate:
#   df.apply(lambda col: ta_func(col).result(), axis=0)
# ===========================================================================

def _rsi_panel(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute the Relative Strength Index for every ticker at every date.

    Formula (Wilder 1978):
        delta  = price.diff()                          # daily change
        gain   = EWM(max(delta, 0), com=window-1)      # smoothed avg gain
        loss   = EWM(max(-delta, 0), com=window-1)     # smoothed avg loss
        RS     = gain / loss
        RSI    = 100 − 100 / (1 + RS)

    The EWM with com=window−1 is the "Wilder smoothing" (equivalent to
    alpha = 1/window), which is the original RSI smoothing method.
    Using adjust=False ensures the recursive formula matches charting
    platforms like TradingView.

    Values range 0–100.  RSI > 70 is traditionally "overbought";
    RSI < 30 is "oversold".  As a cross-sectional factor, higher RSI
    stocks have exhibited stronger recent momentum.

    ta equivalent (single ticker):
        import ta
        rsi_series = ta.momentum.RSIIndicator(close=prices["AAPL"], window=14).rsi()
    """
    # Signed daily price change
    delta = prices.diff()

    # Separate gains (positive changes) and losses (negative changes, sign-flipped)
    up   = delta.clip(lower=0)   # zero out negative deltas → keep only gains
    down = -delta.clip(upper=0)  # zero out positive deltas, negate → positive losses

    # Wilder smoothing: exponential weighted mean with com = window − 1
    # (equivalent to a decay factor α = 1/window, matching the original formula)
    gain = up.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    loss = down.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    # Replace zero loss with NaN to avoid division-by-zero when every day is a gain
    rs = gain / loss.replace(0, np.nan)

    return 100 - 100 / (1 + rs)


def _macd_histogram(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the MACD histogram (MACD line minus signal line) for all tickers.

    Formula (Appel 1979):
        MACD line   = EMA(12) − EMA(26)    # fast minus slow EMA
        Signal line = EMA(9)  of MACD line  # smoothed MACD
        Histogram   = MACD line − Signal line

    The histogram is positive when the MACD line is above its signal line,
    indicating that short-term momentum is accelerating relative to longer-term
    momentum — a bullish signal.  A negative histogram indicates decelerating
    or reversing momentum.

    Parameters are the industry standard: 12/26/9 (fast/slow/signal).

    ta equivalent (single ticker):
        import ta
        macd_ind  = ta.trend.MACD(close=prices["AAPL"],
                                   window_fast=12, window_slow=26, window_sign=9)
        histogram = macd_ind.macd_diff()   # MACD line minus signal line
        # Also available: macd_ind.macd()  → MACD line only
        #                  macd_ind.macd_signal() → signal line only
    """
    # Fast and slow exponential moving averages
    ema12 = prices.ewm(span=12, adjust=False).mean()   # responds quickly to price changes
    ema26 = prices.ewm(span=26, adjust=False).mean()   # slower baseline trend

    # MACD line: how far the fast EMA is above (or below) the slow EMA
    macd = ema12 - ema26

    # Signal line: a 9-period EMA of the MACD line itself (smooths out noise)
    signal = macd.ewm(span=9, adjust=False).mean()

    # Histogram: divergence between MACD and its own signal — measures momentum change
    return macd - signal


def _bollinger_pct_b(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute Bollinger Band %B for all tickers.

    Formula (Bollinger 2001):
        SMA   = rolling mean over `window` days
        σ     = rolling std  over `window` days
        Upper = SMA + 2σ
        Lower = SMA − 2σ
        %B    = (price − lower) / (upper − lower)

    Interpretation:
        %B = 1.0  → price is at the upper band  (2 std above mean)
        %B = 0.5  → price is at the middle band (at the mean)
        %B = 0.0  → price is at the lower band  (2 std below mean)
        %B > 1.0  → price above upper band — extreme strength / breakout
        %B < 0.0  → price below lower band — extreme weakness

    Used here as a trend-following signal: stocks riding or above the upper
    band tend to continue their momentum in cross-sectional studies.

    ta equivalent (single ticker):
        import ta
        bb = ta.volatility.BollingerBands(close=prices["AAPL"], window=20, window_dev=2)
        pct_b = bb.bollinger_pband()   # %B ∈ [0,1] when inside bands
        # Also available: bb.bollinger_hband(), .bollinger_lband(), .bollinger_mavg()

    Note on ta vs our implementation:
        ta.BollingerBands internally calls check_fillna(value=0), replacing any
        NaN %B (e.g. during the warm-up period or when bands have zero width) with 0.
        Our version propagates NaN instead, which is safer for cross-sectional
        ranking: a 0 score looks like a stock sitting at its lower band, whereas
        NaN correctly excludes the ticker from the ranking.  Values differ only
        in these edge cases; inside a normal price window the results are identical.
    """
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()

    upper = sma + 2 * std
    lower = sma - 2 * std

    # Band width; replace 0 with NaN to skip flat-price periods
    band = (upper - lower).replace(0, np.nan)

    return (prices - lower) / band


def _cci_panel(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute the Commodity Channel Index for all tickers.

    Formula (Lambert 1980):
        MA  = rolling mean of close over `window` days
        MAD = mean absolute deviation of close from its MA
        CCI = (close − MA) / (0.015 × MAD)

    The constant 0.015 was chosen by Lambert so that ~70–80 % of CCI
    values fall in the −100 to +100 range for typical price series.

    Interpretation:
        CCI > +100 → price significantly above average → momentum / overbought
        CCI < −100 → price significantly below average → weakness / oversold
        As a cross-sectional factor, higher CCI = stronger relative trend.

    Note on OHLC vs close-only:
        The canonical CCI uses a "typical price" = (high + low + close) / 3.
        Since this factor library only has close prices, we use close as a
        proxy.  The signal direction is preserved; absolute values will differ
        from charting platforms that have full OHLC data.

    ta equivalent (single ticker, with close-only proxy for high/low):
        import ta
        cci = ta.trend.CCIIndicator(
            high=prices["AAPL"],    # using close as proxy for high
            low=prices["AAPL"],     # using close as proxy for low
            close=prices["AAPL"],
            window=20,
        ).cci()
        # With real OHLC data, replace high/low with actual high/low series.
    """
    ma = prices.rolling(window).mean()

    # Mean absolute deviation: average of |x_i − x̄| over the rolling window.
    # pandas has no built-in rolling MAD so we use apply(); raw=True uses a
    # numpy array for speed.  For large universes this is the bottleneck —
    # an approximation via rolling().std() * sqrt(2/pi) would be 10× faster
    # but introduces a normality assumption.
    mad = prices.rolling(window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    return (prices - ma) / (0.015 * mad.replace(0, np.nan))


def _rolling_idio_vol(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling idiosyncratic volatility for all tickers via vectorised OLS.

    Idiosyncratic (stock-specific) volatility is the portion of a stock's
    return variance that is *not* explained by the market factor:

        r_i = α + β_i × r_m + ε_i
        idio_vol = annualised std(ε_i) over `window` days

    We estimate β_i at each date using a rolling covariance formula:

        β_i = Cov(r_i, r_m) / Var(r_m)

    where Cov and Var are computed over the trailing `window` trading days.
    This avoids a per-date Python loop by expressing covariance as:

        Cov(X, Y) = E[XY] − E[X]·E[Y]

    All three rolling means are computed in a single vectorised pass across
    all tickers.

    `ta` has no equivalent for idiosyncratic volatility — it is a
    multi-asset portfolio metric, not a single-ticker indicator.
    The closest single-ticker volatility measures in `ta` are:
        ta.volatility.AverageTrueRange(...)  — range-based vol
        ta.volatility.UlcerIndex(close, ...)  — downside vol
    Neither removes market exposure.

    Requires 'SPY' to be present as the market proxy column in prices.
    """
    if "SPY" not in prices.columns:
        return pd.DataFrame()

    rets = prices.pct_change()
    mkt  = rets["SPY"]

    # Rolling means needed for the covariance identity
    mkt_mean  = mkt.rolling(window).mean()
    rets_mean = rets.rolling(window).mean()

    # E[r_i × r_m] across all tickers simultaneously
    cross_mean = rets.multiply(mkt, axis=0).rolling(window).mean()

    # Cov(r_i, r_m) = E[r_i × r_m] − E[r_i] × E[r_m]
    rolling_cov = cross_mean - rets_mean.multiply(mkt_mean, axis=0)

    # Var(r_m); replace 0 to avoid dividing by zero in flat-market periods
    rolling_var = mkt.rolling(window).var().replace(0, np.nan)

    # β_i at each date: shape (dates × tickers)
    rolling_beta = rolling_cov.divide(rolling_var, axis=0)

    # Residual = actual return − market-explained return
    residuals = rets.subtract(rolling_beta.multiply(mkt, axis=0))

    # Annualised std of residuals
    panel = residuals.rolling(window).std() * _ANNUALIZE

    return panel.drop(columns=["SPY"], errors="ignore")


# ===========================================================================
# Factors
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. RSI (14d)
# ---------------------------------------------------------------------------

@register_factor
class RSI14(BaseFactor):
    """14-day Relative Strength Index.

    Cross-sectional interpretation: stocks with higher RSI have experienced
    more consistent up-days relative to down-days over the past two weeks and
    therefore rank higher in a momentum-sorted portfolio.

    direction = +1  (higher RSI → stronger momentum → expect outperformance)

    Typical range: 0–100.  Values are bounded, so z-scoring or winsorising
    before combining in a composite model is recommended.
    """

    name      = "RSI_14"
    label     = "RSI (14d)"
    description = "14-day Relative Strength Index — high RSI signals strong upward momentum"
    category  = "Momentum"
    direction = 1

    _WINDOW = 14   # Wilder's original lookback period

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        # Need at least window+1 rows so diff() produces window non-NaN values
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        # Compute full panel then take today's cross-section (last row)
        return _rsi_panel(prices, self._WINDOW).iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        # Compute RSI for every date, then down-sample to rebalance frequency.
        # .resample().last() picks the last value within each period (e.g. month-end).
        panel = _rsi_panel(prices, self._WINDOW)
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 2. MACD Histogram (12/26/9)
# ---------------------------------------------------------------------------

@register_factor
class MACDHistogram(BaseFactor):
    """MACD histogram: MACD line minus its 9-period signal line.

    A positive histogram means the fast EMA is pulling away from the slow EMA
    *and* that divergence is still growing — a double confirmation of
    accelerating bullish momentum.  A negative histogram signals the opposite.

    Unlike the raw MACD line, the histogram responds more quickly to changes
    in momentum direction, making it useful for ranking stocks by the *rate
    of change* of their trend strength.

    direction = +1  (positive histogram → momentum accelerating → outperform)

    The histogram is unbounded (unlike RSI), so winsorising before compositing
    is important to prevent outliers from dominating cross-sectional rankings.
    """

    name      = "MACD_HIST"
    label     = "MACD Histogram"
    description = "MACD(12,26,9) histogram — positive values signal bullish momentum"
    category  = "Momentum"
    direction = 1

    # Minimum history needed: EMA-26 needs ≥ 26 warm-up bars, plus 9 for the
    # signal EMA.  35 is a safe lower bound before values stabilise.
    _MIN_BARS = 35

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._MIN_BARS:
            return pd.Series(dtype=float)
        hist = _macd_histogram(prices)
        # Take the most recent row as the cross-sectional snapshot
        return hist.iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        # Full panel: every date has a histogram value for every ticker;
        # resample to rebalance dates
        return _macd_histogram(prices).resample(freq).last()


# ---------------------------------------------------------------------------
# 3. Bollinger Band %B (20d)
# ---------------------------------------------------------------------------

@register_factor
class BollingerPctB(BaseFactor):
    """Price position within 20-day Bollinger Bands (%B).

    %B is a normalised measure of where the current price sits relative to
    its recent trading range.  Unlike raw price or raw deviation from the mean,
    %B adjusts for the current volatility regime: a 3 % move in a calm period
    (narrow bands) registers differently from the same move in a volatile period.

    Cross-sectional use: stocks persistently above their upper band (high %B)
    are exhibiting breakout behaviour and tend to continue outperforming in
    short-to-medium horizons — consistent with price momentum literature.

    direction = +1  (high %B → trending above recent range → expect continuation)

    Note: %B can exceed 1.0 or go below 0.0 when price breaks outside the bands.
    """

    name      = "BB_PCTB"
    label     = "Bollinger %B (20d)"
    description = "Price position within 20-day Bollinger Bands — >0.5 signals upward trend"
    category  = "Momentum"
    direction = 1

    _WINDOW = 20   # standard Bollinger Band length

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        return _bollinger_pct_b(prices, self._WINDOW).iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        return _bollinger_pct_b(prices, self._WINDOW).resample(freq).last()


# ---------------------------------------------------------------------------
# 4. CCI (20d)
# ---------------------------------------------------------------------------

@register_factor
class CCI20(BaseFactor):
    """20-day Commodity Channel Index.

    CCI measures how far price has deviated from its recent average, normalised
    by the typical variability of that deviation (mean absolute deviation × 0.015).

    The 0.015 scaling constant ensures ~70–80 % of observations fall within ±100
    under typical conditions, giving the ±100 thresholds practical meaning.

    As a cross-sectional factor, CCI ranks stocks by their relative departure
    from recent mean prices.  High CCI stocks are trending strongly above their
    average — a momentum signal similar to %B but with a different normalisation.

    direction = +1  (high CCI → price above recent average → momentum signal)

    Important: this implementation uses close-only prices.  The standard CCI
    formula uses the "typical price" = (H + L + C) / 3; without OHLC data,
    close is used for all three, which understates the typical price range.
    Values are directionally correct for cross-sectional ranking purposes.
    """

    name      = "CCI_20"
    label     = "CCI (20d)"
    description = "20-day Commodity Channel Index — measures price deviation from its moving average"
    category  = "Momentum"
    direction = 1

    _WINDOW = 20

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        return _cci_panel(prices, self._WINDOW).iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        return _cci_panel(prices, self._WINDOW).resample(freq).last()


# ---------------------------------------------------------------------------
# 5. Price Acceleration
# ---------------------------------------------------------------------------

@register_factor
class PriceAcceleration(BaseFactor):
    """Rate of change of momentum: recent 1-month return minus the prior 1-month return.

    Formula:
        mom_recent = price[t]       / price[t-21] − 1   (last 21 trading days)
        mom_prior  = price[t-21]    / price[t-42] − 1   (21 days before that)
        acceleration = mom_recent − mom_prior

    A positive acceleration means the stock is gaining momentum (the last
    month was stronger than the month before).  A negative acceleration means
    momentum is fading — even if the stock is still up, it is up less than before.

    This factor captures the *second derivative* of price, whereas standard
    momentum factors capture the first derivative.  It is less correlated with
    Momentum 12-1 than most other momentum factors and can add diversification
    to a composite model.

    No direct `ta` equivalent.  The closest is the Rate-of-Change indicator:
        import ta
        roc = ta.momentum.ROCIndicator(close=prices["AAPL"], window=21).roc()
        # roc[t] = mom_recent in our formula.  Subtract roc[t-21] to get acceleration.

    direction = +1  (positive acceleration → momentum picking up → outperform)
    """

    name      = "PRICE_ACCEL"
    label     = "Price Acceleration"
    description = "Recent 1-month return minus prior 1-month return — catches accelerating momentum"
    category  = "Momentum"
    direction = 1

    _WINDOW = 21   # ≈ 1 calendar month of trading days

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        # Need 2 full windows of history for both recent and prior returns
        if len(prices) < w * 2 + 1:
            return pd.Series(dtype=float)
        # Return over the most recent window (t-21 → t)
        mom_recent = prices.iloc[-1] / prices.iloc[-w] - 1
        # Return over the window immediately before (t-42 → t-21)
        mom_prior  = prices.iloc[-w] / prices.iloc[-w * 2] - 1
        return (mom_recent - mom_prior).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        # Vectorised: shift the price series by w and 2w to get lagged levels
        mom_recent = prices / prices.shift(w) - 1          # return[t-w, t]
        mom_prior  = prices.shift(w) / prices.shift(w * 2) - 1  # return[t-2w, t-w]
        panel = mom_recent - mom_prior
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 6. Trend Consistency (126d)
# ---------------------------------------------------------------------------

@register_factor
class TrendConsistency(BaseFactor):
    """Fraction of positive-return days over the trailing 6-month window.

    Unlike raw 6-month momentum, which rewards a single large gap-up followed
    by a flat period, this factor rewards stocks that go up *consistently* day
    after day.  A stock that rises smoothly every day scores close to 1.0; a
    stock with one huge up-day and then a long sideways drift scores lower.

    Intuition: persistent, low-variance uptrends are driven by repeated
    institutional buying — a more durable signal than a single event pop.

    Formula:
        score = #{days with pct_change > 0} / 126

    This is equivalent to the mean of a 0/1 indicator series, computed via
    a rolling boolean mean — efficient in pandas.

    Difference from UpDayRatio (252d, tutorial factor): this factor uses a
    6-month window (126 days) rather than a full year, making it more
    responsive to recent trend changes.

    No direct `ta` equivalent.  Closest: `ta.trend.AroonIndicator`, which
    measures how recently the highest/lowest price occurred:
        import ta
        aroon = ta.trend.AroonIndicator(high=prices["AAPL"], low=prices["AAPL"], window=25)
        aroon_up = aroon.aroon_up()  # % of periods since the highest high in window

    direction = +1  (high up-day fraction → consistent uptrend → outperform)
    """

    name      = "TREND_CONS"
    label     = "Trend Consistency (126d)"
    description = "Fraction of up-days over 6 months — rewards smooth, sustained directional moves"
    category  = "Momentum"
    direction = 1

    _WINDOW = 126   # ≈ 6 calendar months

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change()
        # Boolean mean: (rets > 0) creates a True/False DataFrame;
        # pandas treats True=1, False=0 so .mean() gives the up-day fraction
        return (rets.tail(self._WINDOW) > 0).mean().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        # Rolling boolean mean across the full history in one vectorised call
        panel = rets.gt(0).rolling(self._WINDOW, min_periods=self._WINDOW).mean()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 7. Calmar Ratio (252d)
# ---------------------------------------------------------------------------

@register_factor
class CalmarRatio(BaseFactor):
    """Return-to-drawdown ratio over the trailing year.

    Formula:
        annual_return = price[t] / price[t−252] − 1
        max_drawdown  = max peak-to-trough decline over the same window
        Calmar        = annual_return / |max_drawdown|

    Originally a hedge-fund performance metric (Young 1991), the Calmar ratio
    measures how much return a stock generates per unit of its worst loss.
    In a cross-sectional context, high-Calmar stocks have earned strong returns
    without suffering large drawdowns — they combine trend momentum with
    controlled downside risk.

    Edge case: if max_drawdown is exactly 0 (price never fell), the denominator
    is replaced with NaN and the ticker is dropped from the cross-section rather
    than producing an infinite score.

    No direct `ta` equivalent.  `ta.volatility.UlcerIndex` is related — it
    measures the depth and duration of drawdowns:
        import ta
        ulcer = ta.volatility.UlcerIndex(close=prices["AAPL"], window=14).ulcer_index()
    Calmar and Ulcer Index both penalise drawdown but use different formulas.

    direction = +1  (higher Calmar → better risk-adjusted return → outperform)
    """

    name      = "CALMAR_252"
    label     = "Calmar Ratio (252d)"
    description = "Annualized 1-year return divided by maximum drawdown — reward-to-risk quality"
    category  = "Momentum"
    direction = 1

    _WINDOW = 252   # 1 trading year

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        if len(prices) < w:
            return pd.Series(dtype=float)
        window_p  = prices.tail(w)
        annual_ret = prices.iloc[-1] / prices.iloc[-w] - 1
        # cummax() at each row is the running peak up to that date
        rolling_max = window_p.cummax()
        # Drawdown at each date: how far below the peak is the current price?
        # min() across dates gives the worst drawdown in the window
        max_dd = ((window_p - rolling_max) / rolling_max).min().abs()
        calmar = annual_ret / max_dd.replace(0, np.nan)
        return calmar.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        annual_ret  = prices / prices.shift(w) - 1
        # Rolling max: highest price in the past w days (the "peak" for drawdown calc)
        rolling_max = prices.rolling(w, min_periods=w // 4).max()
        # Drawdown at each date; rolling min gives the worst single-day drawdown
        # in the trailing window
        max_dd = (
            (prices - rolling_max) / rolling_max.replace(0, np.nan)
        ).rolling(w, min_periods=w // 4).min().abs()
        panel = annual_ret / max_dd.replace(0, np.nan)
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 8. Normalized ATR (14d)
# ---------------------------------------------------------------------------

@register_factor
class NormalizedATR(BaseFactor):
    """Mean absolute daily return over 14 days — a close-to-close volatility proxy.

    True ATR (Wilder 1978) uses the maximum of:
        |high − low|
        |high − prev_close|
        |low  − prev_close|
    averaged over the window.  This requires OHLC data.

    Since this factor library only has adjusted close prices, we approximate
    ATR using the mean of |daily_return| over the window.  This is equivalent
    to the close-to-close component of ATR and preserves the same directional
    signal: stocks with higher intraday ranges on average also have higher
    close-to-close moves.

    Stocks with a high ATR are more volatile — they carry more risk per unit
    of expected return in most factor frameworks (low-volatility anomaly:
    Ang et al. 2006, Baker et al. 2011).

    ta equivalent (single ticker, requires OHLC — uses close as proxy here):
        import ta
        atr = ta.volatility.AverageTrueRange(
            high=prices["AAPL"],    # close as proxy for high
            low=prices["AAPL"],     # close as proxy for low
            close=prices["AAPL"],
            window=14,
        ).average_true_range()
        # With real OHLC data: replace the first two args with actual high/low.
        # The ta version will differ slightly because it uses Wilder smoothing;
        # our version uses a simple arithmetic mean.

    direction = −1  (higher ATR → higher volatility → low-vol anomaly → underperform)
    """

    name      = "ATR_NORM"
    label     = "Normalized ATR (14d)"
    description = "14-day mean absolute daily return — close-to-close volatility range measure"
    category  = "Risk"
    direction = -1   # low-volatility anomaly: less volatile stocks outperform

    _WINDOW = 14   # Wilder's original ATR lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change()
        # Mean of absolute returns over the last window days
        return rets.abs().tail(self._WINDOW).mean().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        # Rolling mean of absolute returns across full history
        panel = rets.abs().rolling(self._WINDOW).mean()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 9. Maximum Drawdown (252d)
# ---------------------------------------------------------------------------

@register_factor
class MaxDrawdown252(BaseFactor):
    """Worst peak-to-trough decline over the trailing year (absolute magnitude).

    Formula:
        rolling_peak  = rolling max of price over the window
        drawdown[t]   = (price[t] − rolling_peak[t]) / rolling_peak[t]  ≤ 0
        max_drawdown  = |min(drawdown over window)|   ≥ 0

    A drawdown of 0.30 means the stock fell 30 % from its peak at some point
    during the year.  We store the absolute value so that a larger number
    always means a worse drawdown — consistent with direction = −1.

    This factor directly penalises tail risk rather than variance.  A stock
    with two large bad months scores worse than one with the same average
    daily vol but smaller individual moves.

    No direct `ta` equivalent.  `ta.volatility.UlcerIndex` is the closest:
        import ta
        ulcer = ta.volatility.UlcerIndex(close=prices["AAPL"], window=252).ulcer_index()
        # Ulcer Index = RMS of drawdowns; max drawdown is a simpler, harder measure.

    direction = −1  (larger max drawdown → bigger tail risk → underperform)
    """

    name      = "MAX_DD_252"
    label     = "Max Drawdown (252d)"
    description = "Worst peak-to-trough decline over the trailing year — absolute magnitude"
    category  = "Risk"
    direction = -1

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        if len(prices) < w // 2:   # require at least half a year of data
            return pd.Series(dtype=float)
        window_p    = prices.tail(w)
        rolling_max = window_p.cummax()
        # Drawdown series: negative values represent declines from peak
        drawdown = (window_p - rolling_max) / rolling_max
        # .min() finds the worst (most negative) drawdown; .abs() flips sign
        return drawdown.min().abs().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        # Rolling peak: the highest price in the trailing w-day window
        rolling_max = prices.rolling(w, min_periods=w // 4).max()
        dd = (prices - rolling_max) / rolling_max.replace(0, np.nan)
        # Rolling minimum drawdown over the same window = worst single drawdown
        panel = dd.rolling(w, min_periods=w // 4).min().abs()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 10. Idiosyncratic Volatility (60d)
# ---------------------------------------------------------------------------

@register_factor
class IdiosyncraticVol(BaseFactor):
    """60-day annualised volatility of market-adjusted (residual) returns.

    We run a rolling single-factor OLS regression for each stock against SPY:

        r_i = α_i + β_i × r_SPY + ε_i

    and compute the annualised standard deviation of the residuals ε_i.

    Why remove market beta first?
        Raw realised volatility (see RVOL_60) conflates systematic risk with
        stock-specific risk.  A high-beta stock will always show high raw vol
        even if all the volatility is simply the market moving.  Idiosyncratic
        vol isolates the portion of risk that is *unique* to the stock — news
        events, earnings surprises, management changes, etc.

        Research (Ang et al. 2006) finds that high idiosyncratic volatility
        predicts *lower* future returns — an anomaly sometimes called the
        "idiosyncratic volatility puzzle" (high specific risk is not rewarded).

    Computation efficiency:
        The panel version uses the identity Cov(X,Y) = E[XY] − E[X]·E[Y]
        to compute rolling betas in one vectorised pass across all tickers,
        avoiding an expensive Python loop over rebalance dates.

    `ta` has no equivalent for idiosyncratic volatility.  See _rolling_idio_vol
    for the full implementation.  Requires 'SPY' in prices.

    direction = −1  (higher idio vol → higher unexplained risk → underperform)
    """

    name      = "IDIO_VOL"
    label     = "Idiosyncratic Volatility (60d)"
    description = "60-day annualized residual return vol after removing market beta exposure"
    category  = "Risk"
    direction = -1

    _WINDOW = 60

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if "SPY" not in prices.columns:
            return pd.Series(dtype=float)
        w    = self._WINDOW
        rets = prices.pct_change().tail(w).dropna()
        if len(rets) < w // 2:   # require at least 30 days of data
            return pd.Series(dtype=float)
        mkt     = rets["SPY"]
        mkt_var = mkt.var()
        if mkt_var == 0:
            return pd.Series(dtype=float)
        result = {}
        for col in rets.columns:
            if col == "SPY":
                continue
            # OLS beta: Cov(r_i, r_m) / Var(r_m)
            beta  = rets[col].cov(mkt) / mkt_var
            # Residual = actual return minus market-explained return
            resid = rets[col] - beta * mkt
            result[col] = resid.std() * _ANNUALIZE
        return pd.Series(result).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        # Delegate to the vectorised helper which avoids a per-date Python loop
        panel = _rolling_idio_vol(prices, self._WINDOW)
        if panel.empty:
            return pd.DataFrame()
        return panel.resample(freq).last()
