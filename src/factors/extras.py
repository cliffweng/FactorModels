"""Extra opt-in factors — disabled by default (enabled_by_default = False).

Enable any of these in the Factor Library page.  They are separated here
because they are either niche, overlap with existing factors, computationally
heavier than the defaults, or require snapshot fundamental data.

Price-based (7)  — compute() and compute_panel() both available:
    ROC_20        Rate of Change (20d)
    WILLIAMS_R    Williams %R (14d)
    STOCH_K       Stochastic %K (14d)
    AROON_OSC     Aroon Oscillator (25d)
    KURT_60       Return Kurtosis (60d)
    ULCER_14      Ulcer Index (14d)
    HIST_VAR      Historical VaR 5% (252d)

Snapshot fundamental (3) — requires_fundamentals = True, snapshot only:
    DIVIDEND_YIELD   Dividend yield from yfinance.info
    SALES_TO_PRICE   Revenue / market cap (S/P ratio) from yfinance.info
    ROA              Return on assets from yfinance.info
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor

_ANNUALIZE = np.sqrt(252)


# ===========================================================================
# Private helpers
# ===========================================================================

def _rolling_range(prices: pd.DataFrame, window: int):
    """Return (rolling_high, rolling_low) over `window` days for all tickers."""
    high = prices.rolling(window).max()
    low  = prices.rolling(window).min()
    return high, low


def _aroon_panel(prices: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Compute Aroon Oscillator for all tickers at all dates.

    Aroon Up   = ((window - days since window-period high) / window) × 100
    Aroon Down = ((window - days since window-period low)  / window) × 100
    Oscillator = Aroon Up − Aroon Down  ∈ [−100, +100]

    Implementation: rolling().apply(argmax/argmin) locates the position of the
    most recent high/low within each window.  Position 0 = oldest bar,
    position window−1 = today.  So:
        days_since_high = (window − 1) − argmax_position
        aroon_up        = (window − days_since_high) / window × 100
                        = (1 + argmax_position) / window × 100

    Using raw=True passes a numpy array to the lambda, which is much faster
    than a pandas Series per window.

    Note: rolling().apply() is still O(n × window) — the dominant cost for
    large universes.  For the snapshot compute() the calculation is immediate.

    ta equivalent (single ticker — uses actual high/low columns):
        import ta
        aroon = ta.trend.AroonIndicator(high=df["high"], low=df["low"], window=25)
        osc   = aroon.aroon_indicator()   # Up minus Down
    Our version uses close as a proxy for both high and low (close-only data).
    """
    high_pos = prices.rolling(window).apply(np.argmax, raw=True)  # 0=oldest, w-1=today
    low_pos  = prices.rolling(window).apply(np.argmin, raw=True)
    aroon_up   = (1 + high_pos) / window * 100
    aroon_down = (1 + low_pos)  / window * 100
    return aroon_up - aroon_down


def _ulcer_panel(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Rolling Ulcer Index for all tickers.

    UI = sqrt( mean( ((close − rolling_max) / rolling_max)² ) )

    Each squared drawdown term is always ≥ 0 (drawdowns are negative ratios,
    so squaring them removes the sign but preserves magnitude).  Averaging
    before the square root captures both depth and duration: a shallow but
    long drawdown scores similarly to a brief deep one.

    ta equivalent (single ticker):
        import ta
        ui = ta.volatility.UlcerIndex(close=prices["AAPL"], window=14).ulcer_index()
    """
    rolling_max = prices.rolling(window).max()
    pct_dd      = (prices - rolling_max) / rolling_max.replace(0, np.nan)  # ≤ 0
    return np.sqrt((pct_dd ** 2).rolling(window).mean())


def _hist_var_panel(prices: pd.DataFrame, window: int = 252, pct: float = 0.05) -> pd.DataFrame:
    """Rolling historical Value-at-Risk (left-tail percentile of returns).

    Fully vectorised via pandas rolling().quantile().  Each cell contains
    the `pct` quantile of the trailing `window` daily returns — a negative
    number representing the typical worst-day loss at the given confidence.

    No distributional assumptions are made (unlike parametric VaR).

    ta has no VaR equivalent.
    """
    rets = prices.pct_change()
    return rets.rolling(window).quantile(pct)


# ===========================================================================
# Price-based technical factors
# ===========================================================================

@register_factor
class RateOfChange20(BaseFactor):
    """20-day Rate of Change — the simplest price momentum signal.

    Formula: ROC = (price[t] / price[t−20] − 1) × 100

    Unlike Momentum 12-1, there is no skip period — the most recent 20 days
    are fully included.  This makes ROC more reactive and more correlated with
    very-short-term noise, but it also captures fast-moving trends that the
    skip-adjusted momentum factors miss.

    ROC is the most basic form of the broader Price-ROC family used in
    classical technical analysis; it equals the 20-day simple return expressed
    as a percentage rather than a decimal.

    direction = +1  (positive ROC → recent price strength → expect continuation)
    """

    name              = "ROC_20"
    label             = "Rate of Change (20d)"
    description       = "20-day price rate of change — simplest momentum signal, no skip"
    category          = "Momentum"
    direction         = 1
    enabled_by_default = False
    formula           = "score = (price(t) / price(t−20) − 1) × 100"
    academic_ref      = "Related to Jegadeesh & Titman (1993); ROC is the raw building block of all momentum factors"
    interpretation    = (
        "**Positive score** — price is higher than 20 trading days ago: short-term uptrend. "
        "**Negative score** — price has fallen over the last month. "
        "Unlike 12-1 momentum, no 1-month skip is applied, so very recent price moves are fully "
        "captured — making this more reactive but also noisier. "
        "Best used in combination with longer-horizon momentum to confirm trend direction."
    )

    _WINDOW = 20

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        # Simple return over the window, scaled to percentage for readability
        return ((prices.iloc[-1] / prices.iloc[-self._WINDOW] - 1) * 100).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = (prices / prices.shift(self._WINDOW) - 1) * 100
        return panel.resample(freq).last()


@register_factor
class WilliamsR14(BaseFactor):
    """14-day Williams %R — close position relative to the recent high-low range.

    Formula: %R = (H14 − close) / (H14 − L14) × (−100)
        Range: [−100, 0]
        %R = 0    → close is at the 14-day high   (bullish)
        %R = −100 → close is at the 14-day low    (bearish)

    Note on close-only approximation:
        Standard Williams %R uses the intraday high and low of each bar.
        Since only adjusted close prices are available here, the rolling
        14-day max and min of close prices are used as proxies.  The signal
        direction is preserved; absolute values will differ from charting
        platforms that use OHLC data.

    Relationship to Stochastic %K:
        %R = −(100 − %K), so the two indicators are mirror images.  A stock
        with %R = −10 has Stochastic %K = 90.  Both are included for
        educational completeness; using both in the same composite adds no
        diversification.

    ta equivalent (single ticker — uses close as high/low proxy):
        import ta
        wr = ta.momentum.WilliamsRIndicator(
            high=prices["AAPL"], low=prices["AAPL"], close=prices["AAPL"], lbp=14
        ).williams_r()

    direction = +1  (higher %R, i.e. closer to 0, → close near recent high → bullish)
    """

    name              = "WILLIAMS_R"
    label             = "Williams %R (14d)"
    description       = "Close position within 14-day range — higher = near recent high = bullish"
    category          = "Momentum"
    direction         = 1
    enabled_by_default = False
    formula           = "%R = (H14 − close) / (H14 − L14) × (−100)  [range −100 to 0]"
    academic_ref      = "Williams (1979) — How I Made One Million Dollars Last Year Trading Commodities"
    interpretation    = (
        "**Score near 0** — the close is at or above the 14-day high: strong bullish momentum. "
        "**Score near −100** — the close is at or below the 14-day low: bearish. "
        "Traditional overbought threshold: %R > −20 (close in top 20% of range). "
        "Traditional oversold threshold: %R < −80 (close in bottom 20%). "
        "As a cross-sectional factor, stocks closest to their recent high outperform. "
        "Mirror image of Stochastic %K — using both adds zero diversification."
    )

    _WINDOW = 14

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        high, low = _rolling_range(prices, self._WINDOW)
        rng = (high - low).iloc[-1].replace(0, np.nan)
        wr  = (high.iloc[-1] - prices.iloc[-1]) / rng * -100
        return wr.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        high, low = _rolling_range(prices, self._WINDOW)
        rng   = (high - low).replace(0, np.nan)
        panel = (high - prices) / rng * -100
        return panel.resample(freq).last()


@register_factor
class StochasticK14(BaseFactor):
    """14-day Stochastic %K — close position within the recent trading range.

    Formula: %K = (close − L14) / (H14 − L14) × 100
        Range: [0, 100]
        %K = 100 → close at 14-day high   (bullish)
        %K = 0   → close at 14-day low    (bearish)

    Developed by George Lane in the 1950s as one of the first momentum
    oscillators.  The indicator was motivated by the observation that in
    uptrending markets, closing prices tend to cluster near the daily high;
    in downtrending markets, they cluster near the low.

    This implementation computes the raw %K line.  The standard practice is
    to smooth %K with a 3-period SMA to get the %D signal line, but for
    cross-sectional ranking the raw %K is used directly.

    Note on close-only approximation: see WilliamsR14 for the same caveat.

    ta equivalent (single ticker — close as high/low proxy):
        import ta
        stoch = ta.momentum.StochasticOscillator(
            high=prices["AAPL"], low=prices["AAPL"], close=prices["AAPL"],
            window=14, smooth_window=3
        )
        k = stoch.stoch()       # raw %K
        d = stoch.stoch_signal() # smoothed %D

    direction = +1  (higher %K → close near recent high → bullish momentum)
    """

    name              = "STOCH_K"
    label             = "Stochastic %K (14d)"
    description       = "14-day Stochastic oscillator — close relative to recent range"
    category          = "Momentum"
    direction         = 1
    enabled_by_default = False
    formula           = "%K = (close − L14) / (H14 − L14) × 100  [range 0 to 100]"
    academic_ref      = "Lane (1984) — Stochastics: A Unique New Technical Indicator"
    interpretation    = (
        "**Score near 100** — the close is at or above the 14-day high: stock is in a "
        "strong local uptrend. Traditional overbought zone: %K > 80. "
        "**Score near 0** — the close is at the 14-day low: bearish pressure. "
        "Traditional oversold zone: %K < 20. "
        "As a cross-sectional factor: high %K stocks are exhibiting sustained strength "
        "within their recent range — a momentum continuation signal. "
        "Mathematically: %K = 100 + Williams %R."
    )

    _WINDOW = 14

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        high, low = _rolling_range(prices, self._WINDOW)
        rng = (high - low).iloc[-1].replace(0, np.nan)
        k   = (prices.iloc[-1] - low.iloc[-1]) / rng * 100
        return k.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        high, low = _rolling_range(prices, self._WINDOW)
        rng   = (high - low).replace(0, np.nan)
        panel = (prices - low) / rng * 100
        return panel.resample(freq).last()


@register_factor
class AroonOscillator25(BaseFactor):
    """25-period Aroon Oscillator — recency of price extremes.

    Aroon Up   = ((n − days since n-period high) / n) × 100
    Aroon Down = ((n − days since n-period low)  / n) × 100
    Oscillator = Aroon Up − Aroon Down  ∈ [−100, +100]

    The key insight: a stock making new highs recently (Aroon Up = 100) is
    in a strong uptrend, regardless of the *magnitude* of the high.  This
    distinguishes Aroon from momentum — Aroon measures *recency* of extremes,
    not *size* of moves.

    Example:
        Oscillator = +100 → a new 25-day high was just printed today AND no
                             new 25-day low in the entire window: ideal uptrend.
        Oscillator =    0 → high and low occurred equally recently: range-bound.
        Oscillator = −100 → new 25-day low today, no new high: downtrend.

    Implementation note: rolling argmax/argmin are used to find the position
    of the most recent extreme within each window.  raw=True is required for
    performance (passes numpy array instead of pandas Series to lambda).

    ta equivalent (single ticker):
        import ta
        aroon = ta.trend.AroonIndicator(high=prices["AAPL"], low=prices["AAPL"], window=25)
        osc   = aroon.aroon_indicator()   # Aroon Up − Aroon Down

    direction = +1  (positive oscillator → recent highs → uptrend → outperform)
    """

    name              = "AROON_OSC"
    label             = "Aroon Oscillator (25d)"
    description       = "Recency of 25-day high vs low — positive = high more recent than low"
    category          = "Momentum"
    direction         = 1
    enabled_by_default = False
    formula           = "Osc = AroonUp − AroonDown;  AroonUp = (25 − days_since_25d_high) / 25 × 100"
    academic_ref      = "Chande (1994) — A New Momentum Oscillator"
    interpretation    = (
        "**Score +100** — the 25-day high occurred today AND no 25-day low occurred "
        "in the entire window: a textbook uptrend. "
        "**Score 0** — high and low occurred with equal recency: no trend. "
        "**Score −100** — pure downtrend: new 25-day low today. "
        "Unlike raw momentum (which measures how much prices moved), Aroon measures "
        "how *recently* price extremes occurred — useful for identifying trend starts "
        "even before significant price movement has accumulated."
    )

    _WINDOW = 25

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        window_p = prices.tail(self._WINDOW)
        # argmax/argmin: position of extreme; 0=oldest bar, window-1=today
        high_pos = window_p.apply(np.argmax)
        low_pos  = window_p.apply(np.argmin)
        aroon_up   = (1 + high_pos) / self._WINDOW * 100
        aroon_down = (1 + low_pos)  / self._WINDOW * 100
        return (aroon_up - aroon_down).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = _aroon_panel(prices, self._WINDOW)
        return panel.resample(freq).last()


@register_factor
class ReturnKurtosis60(BaseFactor):
    """60-day excess kurtosis of daily returns — fat-tail risk measure.

    Formula: excess kurtosis = E[(r − μ)⁴] / σ⁴ − 3

    A normal distribution has excess kurtosis = 0.  Positive excess kurtosis
    (leptokurtic) means the distribution has fatter tails than normal — more
    frequent extreme events in either direction.  Negative (platykurtic) means
    thinner tails.

    Relationship to skewness (SKEW_60):
        Skewness captures asymmetry (right vs left tail).
        Kurtosis captures total tail heaviness regardless of direction.
        A stock can have high kurtosis with low skewness if it has both
        extreme up AND down days.  Including both in a composite provides
        better tail-risk coverage than either alone.

    Cross-sectional evidence: Bali & Cakici (2004) showed that high-kurtosis
    stocks carry higher downside risk and underperform on a risk-adjusted basis.
    Fat tails indicate uncertain, surprise-driven return profiles that
    investors cannot hedge easily.

    pandas provides a built-in rolling().kurt() method — no .apply() needed.

    direction = −1  (high kurtosis → fat tails → elevated tail risk → underperform)
    """

    name              = "KURT_60"
    label             = "Return Kurtosis (60d)"
    description       = "60-day excess kurtosis of daily returns — fat-tail risk indicator"
    category          = "Risk"
    direction         = -1
    enabled_by_default = False
    formula           = "score = E[(r − μ)⁴] / σ⁴ − 3  over trailing 60 trading days"
    academic_ref      = "Bali & Cakici (2004) — Value at Risk and Expected Stock Returns"
    interpretation    = (
        "**Excess kurtosis = 0** — return distribution looks like a normal bell curve. "
        "**Positive kurtosis** — fat tails: the stock experiences extreme moves (large gains "
        "AND large losses) more often than a normal distribution predicts. "
        "**Negative kurtosis** — thin tails: returns are more tightly clustered around the mean. "
        "High-kurtosis stocks are hard to risk-manage (unexpected jumps dominate), "
        "carry hidden downside tail risk, and tend to underperform. "
        "Complement to Return Skewness (SKEW_60): skewness measures asymmetry, kurtosis measures total fat-tailness."
    )

    _WINDOW = 60

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change()
        # pandas .kurt() computes Fisher excess kurtosis (normal = 0)
        return rets.tail(self._WINDOW).kurt().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        # rolling().kurt() is natively vectorised in pandas
        panel = rets.rolling(self._WINDOW, min_periods=self._WINDOW).kurt()
        return panel.resample(freq).last()


@register_factor
class UlcerIndex14(BaseFactor):
    """14-day Ulcer Index — depth-and-duration drawdown risk.

    Formula: UI = sqrt( mean( ((close − rolling_max) / rolling_max)² ) )

    Developed by Peter Martin (1989) as a more intuitive alternative to
    standard deviation for measuring investment risk.  The key properties:

    1. Only penalises downward moves (drawdowns), not upward moves —
       volatility on the upside is not a risk.
    2. Weights deeper drawdowns more than shallow ones (squaring).
    3. Penalises both depth AND duration: a shallow drawdown that lasts many
       periods scores similarly to a brief deep one of the same squared area.

    Unlike Max Drawdown (which captures only the single worst episode),
    Ulcer Index reflects the *typical* drawdown experience over the window.

    Example: UI = 0.05 means the average squared drawdown from peak, over
    the trailing 14 days, corresponds to a 5% typical drawdown — moderate risk.

    ta equivalent (single ticker):
        import ta
        ui = ta.volatility.UlcerIndex(close=prices["AAPL"], window=14).ulcer_index()

    direction = −1  (higher UI → deeper/longer drawdowns → higher downside risk → underperform)
    """

    name              = "ULCER_14"
    label             = "Ulcer Index (14d)"
    description       = "14-day drawdown depth-and-duration risk — lower is calmer"
    category          = "Risk"
    direction         = -1
    enabled_by_default = False
    formula           = "UI = sqrt(mean(((close − rolling_max(14d)) / rolling_max)²))"
    academic_ref      = "Martin & McCann (1989) — The Investor's Guide to Fidelity Funds"
    interpretation    = (
        "**Low UI** — the stock has barely pulled back from its rolling peak over 14 days: "
        "smooth, upward-drifting price action with minimal intraday stress. "
        "**High UI** — the stock has been in a persistent drawdown from its recent high: "
        "either trending down or oscillating with wide swings below peak. "
        "Unlike standard deviation (which penalises upside volatility equally), "
        "Ulcer Index only penalises downward moves — a more investor-friendly risk measure."
    )

    _WINDOW = 14

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        return _ulcer_panel(prices, self._WINDOW).iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        return _ulcer_panel(prices, self._WINDOW).resample(freq).last()


@register_factor
class HistoricalVaR(BaseFactor):
    """Historical Value-at-Risk (5th percentile of daily returns over 252 days).

    Formula: VaR = 5th percentile of { daily_return[t−251], …, daily_return[t] }

    Non-parametric: no distributional assumption.  The score is a negative
    number representing a daily loss that was exceeded on only 5% of trading
    days in the past year.  A score of −0.03 means: on your 5 worst-per-100
    days, you lost at least 3% per day.

    Interpretation for cross-sectional ranking:
        Less negative (closer to 0) → milder worst-day losses → lower tail risk.
        More negative (e.g. −0.06) → severe tail events → higher risk.
        direction = −1: more negative VaR → long-book penalty.

    Relationship to RealizedVol:
        RealizedVol captures average volatility; VaR captures tail severity.
        A stock with modest average vol but occasional extreme drops scores
        differently on VaR vs RVol — VaR is specifically a downside tail measure.

    ta has no VaR equivalent.

    direction = −1  (more negative VaR → worse tail → underperform)
    """

    name              = "HIST_VAR"
    label             = "Historical VaR 5% (252d)"
    description       = "5th percentile of daily returns over 252 days — downside tail risk"
    category          = "Risk"
    direction         = -1
    enabled_by_default = False
    formula           = "score = 5th percentile of daily_returns over trailing 252 trading days"
    academic_ref      = "Jorion (2001) — Value at Risk: The New Benchmark for Managing Financial Risk (3rd ed.)"
    interpretation    = (
        "**Score of −0.02** — on the worst 5% of days in the past year, the stock fell ≥2%: "
        "relatively calm tail risk. "
        "**Score of −0.05** — on bad days, losses were at least 5%: significant tail exposure. "
        "Unlike parametric VaR (which assumes normality), historical VaR makes no "
        "distributional assumptions — it uses the actual observed worst days. "
        "Useful for detecting stocks prone to sudden large drops due to earnings misses, "
        "regulatory surprises, or credit events."
    )

    _WINDOW = 252
    _PCT    = 0.05

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW:
            return pd.Series(dtype=float)
        rets = prices.pct_change().tail(self._WINDOW).dropna()
        return rets.quantile(self._PCT).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = _hist_var_panel(prices, self._WINDOW, self._PCT)
        return panel.resample(freq).last()


# ===========================================================================
# Snapshot fundamental factors  (requires_fundamentals = True, snapshot only)
# ===========================================================================

@register_factor
class DividendYield(BaseFactor):
    """Dividend yield — annual dividends as a fraction of the current price.

    Source: yfinance.info  dividendYield  (snapshot, current values only).

    Dividend yield acts simultaneously as:
        1. An income signal: directly measures the cash return to shareholders.
        2. A value signal: rising yield (falling price) indicates cheapness.
        3. A quality filter: sustained dividends require stable free cash flow.

    Zero-yield stocks (non-payers) receive NaN and are excluded from the
    cross-section — this factor effectively ranks only dividend-paying stocks.
    For a universe with many growth stocks, coverage will be partial.

    direction = +1  (higher yield → more income / cheaper price → outperform)
    """

    name                 = "DIV_YIELD"
    label                = "Dividend Yield"
    description          = "Annual dividend / price — income signal and valuation anchor"
    category             = "Value"
    direction            = 1
    requires_fundamentals = True
    enabled_by_default   = False
    formula              = "score = annual_dividends / current_price  (from yfinance.info)"
    academic_ref         = "Fama & French (1988) — Dividend Yields and Expected Stock Returns; Litzenberger & Ramaswamy (1982)"
    interpretation       = (
        "**High yield** — the stock pays a large dividend relative to its price: "
        "classic income and value signal. Yield > 4% may indicate cheapness or "
        "a payout that is unsustainable (a 'yield trap') — context matters. "
        "**Zero yield** — non-dividend-paying stock; excluded from the cross-section "
        "for this factor (NaN). "
        "Fama & French (1988) showed that high dividend-yield portfolios outperform "
        "on multi-year horizons, particularly in low-interest-rate regimes."
    )

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None or "dividend_yield" not in fundamentals.columns:
            return pd.Series(dtype=float)
        dy = fundamentals["dividend_yield"].dropna()
        return dy[dy > 0].rename(self.name)


@register_factor
class SalesToPrice(BaseFactor):
    """Sales-to-Price ratio (S/P = 1 / Price-to-Sales) — revenue-based value signal.

    Source: yfinance.info  priceToSalesTrailing12Months  → inverted to S/P.

    S/P avoids two weaknesses of E/P (price-to-earnings):
        • Sales are always positive (no negative-earnings distortion).
        • Revenue is harder to manipulate than net income (fewer accounting
          adjustments stand between revenue and the income statement).

    High-S/P (low-P/S) stocks are cheap relative to their revenue base.
    This is especially useful for capital-light businesses (software, services)
    where earnings are reinvested heavily and P/E is distorted.

    direction = +1  (higher S/P = lower P/S = cheaper revenue = value premium)
    """

    name                 = "SALES_PRICE"
    label                = "Sales-to-Price (S/P)"
    description          = "Revenue / market cap — revenue-based value signal, avoids earnings manipulation"
    category             = "Value"
    direction            = 1
    requires_fundamentals = True
    enabled_by_default   = False
    formula              = "score = 1 / Price-to-Sales_TTM  (= Revenue_TTM / Market_Cap)"
    academic_ref         = "O'Shaughnessy (1997) — What Works on Wall Street; Barbee, Mukherji & Raines (1996)"
    interpretation       = (
        "**High score (low P/S)** — you are paying little per dollar of annual revenue: "
        "strong value signal, especially for mature businesses with stable margins. "
        "**Low score (high P/S)** — expensive relative to revenue; typical of hyper-growth "
        "SaaS companies where investors pay for future revenue, not current sales. "
        "O'Shaughnessy (1997) found low P/S one of the strongest single-factor predictors "
        "across 50+ years of US data, outperforming P/E, P/B, and dividend yield. "
        "Denominator is TTM revenue, same data source as fundamentals pipeline."
    )

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None or "ps_ratio" not in fundamentals.columns:
            return pd.Series(dtype=float)
        ps = fundamentals["ps_ratio"].dropna()
        ps = ps[ps > 0]
        return (1.0 / ps).rename(self.name)


@register_factor
class ReturnOnAssets(BaseFactor):
    """Return on Assets — profitability relative to total asset base.

    Source: yfinance.info  returnOnAssets  (snapshot, TTM or latest annual).

    ROA = Net Income / Total Assets

    Advantages over ROE:
        • Not distorted by financial leverage (a highly leveraged company can
          show high ROE with mediocre underlying profitability).
        • Captures efficiency of the *entire* asset base, not just equity.
        • Less susceptible to share buyback inflation of equity returns.

    Fama & French (2015) included asset profitability in their 5-factor model,
    finding that profitable-asset firms earn higher returns after controlling
    for size and value.

    direction = +1  (higher ROA → more efficient asset utilisation → outperform)
    """

    name                 = "ROA"
    label                = "Return on Assets"
    description          = "Net income / total assets — unlevered profitability, quality signal"
    category             = "Quality"
    direction            = 1
    requires_fundamentals = True
    enabled_by_default   = False
    formula              = "score = Net_Income_TTM / Total_Assets"
    academic_ref         = "Fama & French (2015) — A Five-Factor Asset Pricing Model; Chen, Novy-Marx & Zhang (2011)"
    interpretation       = (
        "**High ROA** — every dollar of assets generates strong profit: an asset-efficient, "
        "capital-light business. Tech, pharma, and consumer brands typically have high ROA. "
        "**Low or negative ROA** — assets are poorly utilised or the firm is loss-making; "
        "typical of heavy-industry companies with large PP&E or firms in distress. "
        "ROA is less affected by leverage than ROE: a company can boost ROE by borrowing "
        "more, but ROA rises only if assets generate real profits. "
        "Fama & French (2015) included operating profitability (closely related) as "
        "one of five systematic factors explaining the cross-section of returns."
    )

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None or "roa" not in fundamentals.columns:
            return pd.Series(dtype=float)
        return fundamentals["roa"].dropna().rename(self.name)
