"""Tests for factor computation correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import src.factors  # noqa: register all factors

from src.factors.base import get_registry, list_factors
from tests.helpers import make_prices, make_prices_with_spy, make_fundamentals


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_populated():
    registry = get_registry()
    assert len(registry) >= 7, f"Expected at least 7 registered factors, got {len(registry)}"


def test_factor_attributes():
    for name, factor in get_registry().items():
        assert hasattr(factor, "name"), f"{name} missing 'name'"
        assert hasattr(factor, "label"), f"{name} missing 'label'"
        assert hasattr(factor, "category"), f"{name} missing 'category'"
        assert factor.direction in (1, -1), f"{name}: direction must be ±1"
        assert isinstance(factor.requires_fundamentals, bool)
        assert isinstance(getattr(factor, "requires_edgar", False), bool)


# ---------------------------------------------------------------------------
# Momentum factors
# ---------------------------------------------------------------------------

class TestMomentum12_1:
    def test_compute_shape(self):
        from src.factors.momentum import Momentum12_1
        prices = make_prices(20, 504)
        f = Momentum12_1()
        scores = f.compute(prices)
        assert isinstance(scores, pd.Series)
        assert len(scores) > 0

    def test_compute_no_nan(self):
        from src.factors.momentum import Momentum12_1
        prices = make_prices(20, 504)
        f = Momentum12_1()
        scores = f.compute(prices)
        assert scores.isna().sum() == 0, "compute() should not return NaN values"

    def test_compute_insufficient_history(self):
        from src.factors.momentum import Momentum12_1
        prices = make_prices(10, 100)  # only 100 days
        f = Momentum12_1()
        scores = f.compute(prices)
        assert scores.empty, "Should return empty Series with insufficient history"

    def test_panel_shape(self):
        from src.factors.momentum import Momentum12_1
        prices = make_prices(20, 504)
        f = Momentum12_1()
        panel = f.compute_panel(prices, freq="ME")
        assert isinstance(panel, pd.DataFrame)
        assert panel.shape[1] == 20
        assert panel.shape[0] >= 12, "Should have at least 12 monthly rows for 504 days"

    def test_higher_price_higher_score(self):
        """Stock with higher 12m return should have higher momentum score."""
        from src.factors.momentum import Momentum12_1
        prices = make_prices(5, 504, seed=99)
        # Manually override last stock to have very high recent appreciation
        prices_modified = prices.copy()
        prices_modified.iloc[-22:, 0] = prices_modified.iloc[-252, 0] * 2.0  # 100% return
        prices_modified.iloc[-22:, 1] = prices_modified.iloc[-252, 1] * 0.5  # -50% return
        f = Momentum12_1()
        scores = f.compute(prices_modified)
        assert scores.iloc[0] > scores.iloc[1], "High-return stock should rank higher"


class TestShortTermReversal:
    def test_negated_return(self):
        """STR should be negated recent return — a recent loser gets high score."""
        from src.factors.momentum import ShortTermReversal
        prices = make_prices(10, 100)
        f = ShortTermReversal()
        scores = f.compute(prices)
        # Compute manual 5-day return
        manual_ret = prices.iloc[-1] / prices.iloc[-5] - 1
        # STR = -manual_ret (tickers with negative return get higher score)
        for t in scores.index:
            if t in manual_ret.index:
                assert abs(scores[t] - (-manual_ret[t])) < 1e-9


# ---------------------------------------------------------------------------
# Risk factors
# ---------------------------------------------------------------------------

class TestRealizedVol:
    def test_positive_vol(self):
        from src.factors.risk import RealizedVol60
        prices = make_prices(15, 300)
        f = RealizedVol60()
        scores = f.compute(prices)
        assert (scores > 0).all(), "Volatility must always be positive"

    def test_panel_positive(self):
        from src.factors.risk import RealizedVol60
        prices = make_prices(15, 400)
        f = RealizedVol60()
        panel = f.compute_panel(prices)
        # dropna before asserting: rolling warmup produces NaN in early months
        valid = panel.stack().dropna()
        assert valid.gt(0).mean() > 0.9, "Most non-NaN panel values should be positive"

    def test_constant_prices_zero_vol(self):
        from src.factors.risk import RealizedVol60
        dates = pd.bdate_range("2022-01-01", periods=200)
        prices = pd.DataFrame(
            np.ones((200, 5)) * 100,
            index=dates,
            columns=[f"S{i}" for i in range(5)],
        )
        f = RealizedVol60()
        scores = f.compute(prices)
        assert (scores == 0.0).all(), "Constant prices → zero volatility"


class TestBeta252:
    def test_spy_required(self):
        from src.factors.risk import Beta252
        prices = make_prices(10, 400)  # no SPY column
        f = Beta252()
        scores = f.compute(prices)
        assert scores.empty, "Should return empty without SPY"

    def test_beta_range(self):
        from src.factors.risk import Beta252
        prices = make_prices_with_spy(10, 400)
        f = Beta252()
        scores = f.compute(prices)
        assert not scores.empty
        assert (scores > -2) .all() and (scores < 5).all(), "Betas should be in reasonable range"


# ---------------------------------------------------------------------------
# Value factors
# ---------------------------------------------------------------------------

class TestFiftyTwoWeekHigh:
    def test_ratio_between_0_and_1(self):
        from src.factors.value import FiftyTwoWeekHighRatio
        prices = make_prices(10, 400)
        f = FiftyTwoWeekHighRatio()
        scores = f.compute(prices)
        assert (scores > 0).all() and (scores <= 1.0 + 1e-9).all()

    def test_at_high_score_is_one(self):
        """A stock at its 52-week high should score exactly 1.0."""
        from src.factors.value import FiftyTwoWeekHighRatio
        prices = make_prices(5, 400)
        # Force last value to be the maximum for first ticker
        prices_mod = prices.copy()
        prices_mod.iloc[-1, 0] = prices_mod.iloc[-252:, 0].max() * 2  # set new all-time high
        f = FiftyTwoWeekHighRatio()
        scores = f.compute(prices_mod)
        assert abs(scores.iloc[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Fundamental factors (snapshot only)
# ---------------------------------------------------------------------------

class TestPriceToBook:
    def test_inverse_pb(self):
        from src.factors.value import PriceToBook
        prices = make_prices(10, 300)
        fund = make_fundamentals(prices.columns.tolist())
        f = PriceToBook()
        scores = f.compute(prices, fundamentals=fund)
        assert not scores.empty
        # Verify: score ≈ 1/pb for each ticker
        for t in scores.index:
            if t in fund.index:
                expected = 1.0 / fund.loc[t, "pb"]
                assert abs(scores[t] - expected) < 1e-9

    def test_no_fundamentals_returns_empty(self):
        from src.factors.value import PriceToBook
        prices = make_prices(5, 300)
        f = PriceToBook()
        scores = f.compute(prices)
        assert scores.empty

    def test_no_panel_without_edgar(self):
        from src.factors.value import PriceToBook
        prices = make_prices(5, 300)
        f = PriceToBook()
        # Without edgar_panel kwarg, compute_panel returns empty DataFrame (graceful)
        result = f.compute_panel(prices)
        assert result.empty


# ---------------------------------------------------------------------------
# BaseFactor utilities
# ---------------------------------------------------------------------------

def test_winsorize():
    from src.factors.base import BaseFactor
    import src.factors  # noqa

    from src.factors.momentum import Momentum12_1
    prices = make_prices(20, 504)
    f = Momentum12_1()
    scores = f.compute(prices)
    w = f.winsorize(scores, pct=0.01)
    assert w.min() >= scores.quantile(0.01) - 1e-9
    assert w.max() <= scores.quantile(0.99) + 1e-9


def test_z_score():
    from src.factors.momentum import Momentum12_1
    prices = make_prices(20, 504)
    f = Momentum12_1()
    scores = f.compute(prices)
    z = f.z_score(scores)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 1e-10
