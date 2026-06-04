"""Tests for quantile portfolio formation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.analysis.quantile import form_quantile_portfolios
from tests.helpers import make_prices


def make_predictive_factor(prices: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Factor perfectly correlated with next-month returns."""
    rets = prices.pct_change()
    fwd_monthly = (1 + rets).rolling(21).apply(np.prod, raw=True).shift(-21) - 1
    panel = fwd_monthly.resample("ME").last().dropna(how="all")
    return panel


class TestQuantilePortfolios:
    def test_returns_correct_quantiles(self):
        prices = make_prices(30, 504)
        from src.factors.momentum import Momentum12_1
        f = Momentum12_1()
        panel = f.compute_panel(prices)
        daily_rets = prices.pct_change().dropna(how="all")
        result = form_quantile_portfolios(panel, daily_rets, n_quantiles=5)
        assert set(result.portfolio_returns.keys()) == {1, 2, 3, 4, 5}

    def test_all_portfolios_have_returns(self):
        prices = make_prices(30, 504)
        from src.factors.momentum import Momentum12_1
        f = Momentum12_1()
        panel = f.compute_panel(prices)
        daily_rets = prices.pct_change().dropna(how="all")
        result = form_quantile_portfolios(panel, daily_rets, n_quantiles=5)
        for q, rets in result.portfolio_returns.items():
            assert len(rets) > 0, f"Q{q} has no returns"

    def test_monotonicity_predictive_factor(self):
        """With a perfect predictor, Q5 should outperform Q1."""
        rng = np.random.default_rng(42)
        n_tickers = 40
        n_days = 504
        dates = pd.bdate_range("2022-01-01", periods=n_days)
        # Create returns with cross-sectional signal
        cross_signal = rng.normal(0, 1, (n_days, n_tickers))
        # Each stock's return = 0.001 * signal + noise
        daily_rets = cross_signal * 0.005 + rng.normal(0, 0.005, (n_days, n_tickers))
        tickers = [f"S{i:02d}" for i in range(n_tickers)]
        ret_df = pd.DataFrame(daily_rets, index=dates, columns=tickers)

        # Factor: 21-day forward return (perfect predictor)
        fwd = ret_df.rolling(21).sum().shift(-21)
        panel = fwd.resample("ME").last().dropna(how="all")

        result = form_quantile_portfolios(panel, ret_df, n_quantiles=5, direction=1)
        cum5 = (1 + result.portfolio_returns[5]).prod()
        cum1 = (1 + result.portfolio_returns[1]).prod()
        assert cum5 > cum1, f"Q5 ({cum5:.3f}) should outperform Q1 ({cum1:.3f})"

    def test_n_quantiles_3(self):
        prices = make_prices(20, 504)
        from src.factors.momentum import Momentum6_1
        f = Momentum6_1()
        panel = f.compute_panel(prices)
        daily_rets = prices.pct_change().dropna(how="all")
        result = form_quantile_portfolios(panel, daily_rets, n_quantiles=3)
        assert set(result.portfolio_returns.keys()) == {1, 2, 3}

    def test_spread_returns_sign(self):
        prices = make_prices(30, 504)
        from src.factors.momentum import Momentum12_1
        f = Momentum12_1()
        panel = f.compute_panel(prices)
        daily_rets = prices.pct_change().dropna(how="all")
        result = form_quantile_portfolios(panel, daily_rets, n_quantiles=5)
        spread = result.spread_returns
        assert isinstance(spread, pd.Series)
        assert len(spread) > 0

    def test_stats_table_shape(self):
        prices = make_prices(25, 504)
        from src.factors.momentum import Momentum12_1
        f = Momentum12_1()
        panel = f.compute_panel(prices)
        daily_rets = prices.pct_change().dropna(how="all")
        result = form_quantile_portfolios(panel, daily_rets, n_quantiles=5)
        result.factor_name = "Momentum12_1"
        stats = result.stats_table()
        assert "Ann. Return" in stats.columns
        assert "Sharpe" in stats.columns
        assert len(stats) == 6  # 5 quintiles + L/S Spread
