"""Risk-based factors: realised volatility and market beta.

Low-volatility / low-beta anomaly
-----------------------------------
Classical finance theory (CAPM) predicts that higher risk should be
rewarded with higher return.  Empirically the opposite often holds:
low-volatility and low-beta stocks have delivered *higher* risk-adjusted
returns than high-volatility and high-beta stocks — the "low-vol anomaly"
(Ang et al. 2006; Baker, Bradley & Wurgler 2011).

Proposed explanations include:
    • Leverage-constrained investors bid up high-beta stocks to gain
      market exposure without borrowing.
    • Benchmarked fund managers prefer high-volatility stocks for their
      option-like payoffs and career-risk properties.
    • Lottery-ticket demand inflates prices of the most volatile stocks.

Both factors below carry direction = −1, meaning *lower* values rank in
the long book (less volatile / less market-sensitive stocks are preferred).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor, register_factor

# Annualisation multiplier: converts daily standard deviation to annual.
# Assumes 252 trading days per year; daily vol × √252 = annual vol.
_ANNUALIZE = np.sqrt(252)


@register_factor
class RealizedVol60(BaseFactor):
    """Annualised realised return volatility over a 60-trading-day window.

    Construction
    ------------
    Compute daily log returns (pct_change), take the trailing 60-day
    standard deviation, then annualise by multiplying by √252.

    Formula
    -------
        daily_ret[t] = price[t] / price[t−1] − 1
        RVol[t]      = std(daily_ret[t−59 : t]) × √252

    Window choice
    -------------
    _WINDOW = 60 ≈ 3 calendar months.  This is long enough to be
    statistically stable (≈ 60 observations) but short enough to adapt to
    regime changes in a stock's risk profile.  Common alternatives are 21d
    (1 month, more reactive) and 252d (1 year, more stable).

    Relationship to ATR
    -------------------
    Realised vol and Normalised ATR (technical.py) both measure daily price
    variation but differ in scaling:
        • RVol is annualised (×√252) and uses the sample std (ddof=1).
        • ATR_NORM uses the mean of |returns| (not std) with no annualisation.
    They are highly correlated (~0.95) and will be nearly redundant in a
    composite model — consider using only one.

    direction = −1  (lower vol → low-vol anomaly → expected outperformance)
    """

    name      = "RVOL_60"
    label     = "Realized Volatility (60d)"
    description = "Annualized 60-day realized return volatility — lower is better (Low-Vol anomaly)"
    category  = "Risk"
    direction = -1   # low volatility → higher risk-adjusted return (low-vol anomaly)
    formula        = "RVol = std(daily_returns, 60d) × √252"
    academic_ref   = "Ang, Hodrick, Xing & Zhang (2006) — The Cross-Section of Volatility and Expected Returns"
    interpretation = (
        "**Counter-intuitive result**: lower volatility stocks historically *outperform* "
        "higher volatility stocks (low-vol anomaly). "
        "**High score** — volatile stock; leverage-constrained investors overpay for high-vol "
        "names seeking market exposure without borrowing, inflating their prices. "
        "**Low score** — calm, predictable price path; tends to outperform on a risk-adjusted basis. "
        "Highly correlated with ATR_NORM — consider using only one in a composite."
    )

    _WINDOW = 60   # 60 trading days ≈ 3 calendar months

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional realised vol at the most recent date.

        We need at least _WINDOW rows of *returns* (i.e. _WINDOW + 1 prices)
        for a full-window estimate.  Fewer rows would give a std over an
        incomplete window, which tends to understate true volatility.

        dropna() on the returns removes the first NaN row produced by
        pct_change(), ensuring the std is computed over exactly _WINDOW
        observations.
        """
        rets = prices.pct_change()

        # Require a full window of non-NaN returns
        if len(rets.dropna()) < self._WINDOW:
            return pd.Series(dtype=float)

        # Trailing 60-day std then annualise; dropna() removes tickers with
        # insufficient history (e.g. recent IPOs)
        vol = rets.tail(self._WINDOW).std() * _ANNUALIZE
        return vol.dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        """Rolling realised-vol panel, vectorised via rolling().std().

        pandas rolling().std() uses ddof=1 (sample std) by default, which
        matches the compute() snapshot.  All tickers are computed in a single
        call — no Python loop over dates.

        The warm-up period (first _WINDOW rows where rolling std is NaN) is
        retained in the full panel but rows with all NaN are dropped after
        resampling via resample().last().
        """
        rets = prices.pct_change()

        # Rolling standard deviation across the full history (shape: dates × tickers)
        panel = rets.rolling(self._WINDOW).std() * _ANNUALIZE

        # Down-sample to rebalance dates
        rebal = panel.resample(freq).last()
        return rebal


@register_factor
class Beta252(BaseFactor):
    """252-day rolling market beta estimated against SPY.

    What beta measures
    ------------------
    Beta is the slope coefficient from the single-factor market model:

        r_i = α + β × r_SPY + ε

    A beta of 1.0 means the stock moves one-for-one with the market.
    A beta > 1.0 amplifies market moves (aggressive); beta < 1.0 dampens
    them (defensive).  Beta < 0 is counter-cyclical (e.g. some gold miners
    or volatility products).

    Estimation
    ----------
    We use the OLS formula:

        β = Cov(r_i, r_SPY) / Var(r_SPY)

    computed over the trailing _WINDOW trading days.  This rolling estimate
    adapts over time as a stock's market sensitivity changes (e.g. after
    a leverage event or a strategic pivot).

    _WINDOW = 252  ≈ 1 trading year.  A full year gives a statistically
    reliable estimate (~252 observations) while capturing changes in the
    stock's sector exposures over time.

    Why beta > 1 predicts underperformance
    ----------------------------------------
    The low-beta anomaly (Black 1972; Frazzini & Pedersen 2014) finds that
    the security market line is *flatter* than CAPM predicts: high-beta stocks
    are over-priced relative to their risk, so they underperform on a
    risk-adjusted basis.  The mechanism mirrors the low-vol anomaly — see
    the module docstring above.

    Market proxy
    ------------
    SPY (SPDR S&P 500 ETF) is used as the market portfolio.  The factor
    requires SPY to be present as a column in the prices DataFrame.  Pages
    that call this factor include SPY via the BENCHMARK constant in
    src/data/universe.py.

    direction = −1  (lower beta → less systematic risk → expected outperformance)
    """

    name      = "BETA_252"
    label     = "Market Beta (252d)"
    description = "252-day rolling beta vs SPY — deviation from 1.0 used as factor score"
    category  = "Risk"
    direction = -1   # low beta → lower systematic risk → outperformance (low-beta anomaly)
    formula        = "β = Cov(r_i, r_SPY) / Var(r_SPY)  over 252 trading days"
    academic_ref   = "Black (1972); Frazzini & Pedersen (2014) — Betting Against Beta"
    interpretation = (
        "**High score** — amplifies market moves; CAPM predicts it should earn higher returns, "
        "but empirically the security market line is *flatter* than theory says: high-beta "
        "stocks are over-priced and underperform on a risk-adjusted basis. "
        "**Low score (β < 1)** — defensive stock; dampens market drawdowns. "
        "The Frazzini & Pedersen (2014) BAB factor earns ~7% per year across global markets "
        "by being long low-beta and short high-beta stocks."
    )

    _WINDOW = 252   # 1 trading year for a reliable OLS estimate

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional beta snapshot at the most recent date.

        We use the trailing _WINDOW returns for every stock and compute the
        OLS beta against SPY in a tight loop over tickers.  A Python loop is
        acceptable here because compute() is a one-shot snapshot (called once
        per page load), unlike compute_panel() which processes many dates.

        Guard rails:
            • Return early if SPY is not in prices (no market return available).
            • Require at least _WINDOW // 2 clean rows (126 days) so early in
              a stock's history we don't produce wildly unreliable estimates.
            • Replace Var(r_SPY) == 0 with NaN to skip flat-market windows.
        """
        if "SPY" not in prices.columns:
            return pd.Series(dtype=float)

        # Take the trailing window; drop leading NaN from pct_change
        rets = prices.pct_change().tail(self._WINDOW).dropna()

        # Require at least half the window to be present
        if len(rets) < self._WINDOW // 2:
            return pd.Series(dtype=float)

        market  = rets["SPY"]
        mkt_var = market.var()
        if mkt_var == 0:
            return pd.Series(dtype=float)

        # OLS beta for each ticker: Cov(r_i, r_m) / Var(r_m)
        betas = {}
        for col in rets.columns:
            if col == "SPY":
                continue   # skip the benchmark itself
            cov        = rets[col].cov(market)
            betas[col] = cov / mkt_var

        return pd.Series(betas).dropna()

    def compute_panel(self, prices: pd.DataFrame, freq: str = "ME", min_periods: int = 252) -> pd.DataFrame:
        """Rolling beta panel computed at each rebalance date.

        Because rolling().cov() is not available as a DataFrame-vs-Series
        operation in all pandas versions, we iterate over rebalance dates.
        At each date we slice the trailing _WINDOW rows and run the same
        OLS beta computation as compute().

        Performance note: iterating over rebalance dates (12 per year × years)
        is far cheaper than iterating over daily rows, so this remains
        practical even for large universes.  A fully vectorised alternative
        using the identity Cov(X,Y) = E[XY] − E[X]·E[Y] is used in
        _rolling_idio_vol (technical.py) if you need a daily panel.

        Results are accumulated in a dict {date → pd.Series} and assembled
        into the final DataFrame at the end.
        """
        if "SPY" not in prices.columns:
            return pd.DataFrame()

        rets = prices.pct_change()
        market = rets["SPY"]

        results = {}

        # Rebalance dates: the last trading day of each period
        rebal_dates = rets.resample(freq).last().index

        for date in rebal_dates:
            # Trailing window ending at this rebalance date
            window_rets = rets.loc[:date].tail(self._WINDOW).dropna(how="all")

            # Skip if insufficient history
            if len(window_rets) < self._WINDOW // 2:
                continue

            # Market returns aligned to the same window
            mkt_w   = market.loc[window_rets.index]
            mkt_var = mkt_w.var()
            if mkt_var == 0:
                continue

            # Compute beta for each ticker at this rebalance date
            betas = {}
            for col in window_rets.columns:
                if col == "SPY":
                    continue
                cov        = window_rets[col].cov(mkt_w)
                betas[col] = cov / mkt_var

            results[date] = pd.Series(betas)

        if not results:
            return pd.DataFrame()

        # Assemble: rows = rebalance dates, columns = tickers
        return pd.DataFrame(results).T
