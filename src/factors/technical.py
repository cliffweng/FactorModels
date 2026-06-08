"""Technical indicator factors — RSI, MACD, Bollinger, CCI, and more."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor

_ANNUALIZE = np.sqrt(252)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi_panel(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    loss = down.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_histogram(prices: pd.DataFrame) -> pd.DataFrame:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _bollinger_pct_b(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band = (upper - lower).replace(0, np.nan)
    return (prices - lower) / band


def _cci_panel(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ma = prices.rolling(window).mean()
    # Mean absolute deviation per column
    mad = prices.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (prices - ma) / (0.015 * mad.replace(0, np.nan))


def _rolling_idio_vol(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling idiosyncratic vol using vectorised beta via rolling cov/var."""
    if "SPY" not in prices.columns:
        return pd.DataFrame()
    rets = prices.pct_change()
    mkt = rets["SPY"]
    # cov(r_i, r_m) = E[r_i * r_m] - E[r_i]*E[r_m]
    mkt_mean = mkt.rolling(window).mean()
    rets_mean = rets.rolling(window).mean()
    cross_mean = rets.multiply(mkt, axis=0).rolling(window).mean()
    rolling_cov = cross_mean - rets_mean.multiply(mkt_mean, axis=0)
    rolling_var = mkt.rolling(window).var().replace(0, np.nan)
    rolling_beta = rolling_cov.divide(rolling_var, axis=0)
    residuals = rets.subtract(rolling_beta.multiply(mkt, axis=0))
    panel = residuals.rolling(window).std() * _ANNUALIZE
    return panel.drop(columns=["SPY"], errors="ignore")


# ---------------------------------------------------------------------------
# 1. RSI (14d)
# ---------------------------------------------------------------------------

@register_factor
class RSI14(BaseFactor):
    name = "RSI_14"
    label = "RSI (14d)"
    description = "14-day Relative Strength Index — high RSI signals strong upward momentum"
    category = "Momentum"
    direction = 1

    _WINDOW = 14

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        return _rsi_panel(prices, self._WINDOW).iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = _rsi_panel(prices, self._WINDOW)
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 2. MACD Histogram (12/26/9)
# ---------------------------------------------------------------------------

@register_factor
class MACDHistogram(BaseFactor):
    name = "MACD_HIST"
    label = "MACD Histogram"
    description = "MACD(12,26,9) histogram — positive values signal bullish momentum"
    category = "Momentum"
    direction = 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < 35:
            return pd.Series(dtype=float)
        hist = _macd_histogram(prices)
        return hist.iloc[-1].dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        return _macd_histogram(prices).resample(freq).last()


# ---------------------------------------------------------------------------
# 3. Bollinger Band %B (20d)
# ---------------------------------------------------------------------------

@register_factor
class BollingerPctB(BaseFactor):
    name = "BB_PCTB"
    label = "Bollinger %B (20d)"
    description = "Price position within 20-day Bollinger Bands — >0.5 signals upward trend"
    category = "Momentum"
    direction = 1

    _WINDOW = 20

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
    name = "CCI_20"
    label = "CCI (20d)"
    description = "20-day Commodity Channel Index — measures price deviation from its moving average"
    category = "Momentum"
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
    name = "PRICE_ACCEL"
    label = "Price Acceleration"
    description = "Recent 1-month return minus prior 1-month return — catches accelerating momentum"
    category = "Momentum"
    direction = 1

    _WINDOW = 21

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        if len(prices) < w * 2 + 1:
            return pd.Series(dtype=float)
        mom_recent = prices.iloc[-1] / prices.iloc[-w] - 1
        mom_prior = prices.iloc[-w] / prices.iloc[-w * 2] - 1
        return (mom_recent - mom_prior).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        mom_recent = prices / prices.shift(w) - 1
        mom_prior = prices.shift(w) / prices.shift(w * 2) - 1
        panel = mom_recent - mom_prior
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 6. Trend Consistency (126d)
# ---------------------------------------------------------------------------

@register_factor
class TrendConsistency(BaseFactor):
    name = "TREND_CONS"
    label = "Trend Consistency (126d)"
    description = "Fraction of up-days over 6 months — rewards smooth, sustained directional moves"
    category = "Momentum"
    direction = 1

    _WINDOW = 126

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change()
        return (rets.tail(self._WINDOW) > 0).mean().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        panel = rets.gt(0).rolling(self._WINDOW, min_periods=self._WINDOW).mean()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 7. Calmar Ratio (252d)
# ---------------------------------------------------------------------------

@register_factor
class CalmarRatio(BaseFactor):
    name = "CALMAR_252"
    label = "Calmar Ratio (252d)"
    description = "Annualized 1-year return divided by maximum drawdown — reward-to-risk quality"
    category = "Momentum"
    direction = 1

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        if len(prices) < w:
            return pd.Series(dtype=float)
        window_p = prices.tail(w)
        annual_ret = prices.iloc[-1] / prices.iloc[-w] - 1
        rolling_max = window_p.cummax()
        max_dd = ((window_p - rolling_max) / rolling_max).min().abs()
        calmar = annual_ret / max_dd.replace(0, np.nan)
        return calmar.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        annual_ret = prices / prices.shift(w) - 1
        rolling_max = prices.rolling(w, min_periods=w // 4).max()
        max_dd = ((prices - rolling_max) / rolling_max.replace(0, np.nan)).rolling(w, min_periods=w // 4).min().abs()
        panel = annual_ret / max_dd.replace(0, np.nan)
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 8. Normalized ATR (14d)
# ---------------------------------------------------------------------------

@register_factor
class NormalizedATR(BaseFactor):
    name = "ATR_NORM"
    label = "Normalized ATR (14d)"
    description = "14-day mean absolute daily return — close-to-close volatility range measure"
    category = "Risk"
    direction = -1

    _WINDOW = 14

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if len(prices) < self._WINDOW + 1:
            return pd.Series(dtype=float)
        rets = prices.pct_change()
        return rets.abs().tail(self._WINDOW).mean().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        rets = prices.pct_change()
        panel = rets.abs().rolling(self._WINDOW).mean()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 9. Maximum Drawdown (252d)
# ---------------------------------------------------------------------------

@register_factor
class MaxDrawdown252(BaseFactor):
    name = "MAX_DD_252"
    label = "Max Drawdown (252d)"
    description = "Worst peak-to-trough decline over the trailing year — absolute magnitude"
    category = "Risk"
    direction = -1

    _WINDOW = 252

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        w = self._WINDOW
        if len(prices) < w // 2:
            return pd.Series(dtype=float)
        window_p = prices.tail(w)
        rolling_max = window_p.cummax()
        drawdown = (window_p - rolling_max) / rolling_max
        return drawdown.min().abs().dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        w = self._WINDOW
        rolling_max = prices.rolling(w, min_periods=w // 4).max()
        dd = (prices - rolling_max) / rolling_max.replace(0, np.nan)
        panel = dd.rolling(w, min_periods=w // 4).min().abs()
        return panel.resample(freq).last()


# ---------------------------------------------------------------------------
# 10. Idiosyncratic Volatility (60d)
# ---------------------------------------------------------------------------

@register_factor
class IdiosyncraticVol(BaseFactor):
    name = "IDIO_VOL"
    label = "Idiosyncratic Volatility (60d)"
    description = "60-day annualized residual return vol after removing market beta exposure"
    category = "Risk"
    direction = -1

    _WINDOW = 60

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        if "SPY" not in prices.columns:
            return pd.Series(dtype=float)
        w = self._WINDOW
        rets = prices.pct_change().tail(w).dropna()
        if len(rets) < w // 2:
            return pd.Series(dtype=float)
        mkt = rets["SPY"]
        mkt_var = mkt.var()
        if mkt_var == 0:
            return pd.Series(dtype=float)
        result = {}
        for col in rets.columns:
            if col == "SPY":
                continue
            beta = rets[col].cov(mkt) / mkt_var
            resid = rets[col] - beta * mkt
            result[col] = resid.std() * _ANNUALIZE
        return pd.Series(result).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        panel = _rolling_idio_vol(prices, self._WINDOW)
        if panel.empty:
            return pd.DataFrame()
        return panel.resample(freq).last()
