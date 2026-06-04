"""Tests for IC computation — validates the core analytical pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.analysis.ic import (
    compute_ic, compute_ic_series, compute_rolling_ic,
    compute_icir, compute_ic_decay, compute_forward_returns,
)
from tests.helpers import make_prices


def make_perfect_factor_panel(prices: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """Create a factor that perfectly predicts next-period returns (IC = 1.0)."""
    rets = prices.pct_change()
    # Perfect factor: scores ARE the future returns
    fwd = (1 + rets).rolling(horizon_days).apply(lambda x: x.prod(), raw=True).shift(-horizon_days) - 1
    monthly = fwd.resample("ME").last().dropna(how="all")
    return monthly


# ---------------------------------------------------------------------------
# compute_ic (single cross-section)
# ---------------------------------------------------------------------------

def test_ic_perfect_rank():
    """IC of perfectly ranked factor should be 1.0."""
    n = 30
    factor = pd.Series(np.arange(n, dtype=float), index=[f"S{i}" for i in range(n)])
    fwd = pd.Series(np.arange(n, dtype=float) * 0.01 + 0.001,
                    index=[f"S{i}" for i in range(n)])
    ic = compute_ic(factor, fwd, min_obs=5)
    assert abs(ic - 1.0) < 1e-9


def test_ic_reverse_rank():
    """IC of reversed factor should be -1.0."""
    n = 30
    factor = pd.Series(np.arange(n, dtype=float), index=[f"S{i}" for i in range(n)])
    fwd = pd.Series(np.arange(n - 1, -1, -1, dtype=float) * 0.01,
                    index=[f"S{i}" for i in range(n)])
    ic = compute_ic(factor, fwd, min_obs=5)
    assert abs(ic + 1.0) < 1e-9


def test_ic_random_near_zero():
    """IC of random factor vs random returns should be near zero."""
    rng = np.random.default_rng(0)
    n = 200
    factor = pd.Series(rng.normal(size=n), index=[f"S{i}" for i in range(n)])
    fwd = pd.Series(rng.normal(size=n), index=[f"S{i}" for i in range(n)])
    ic = compute_ic(factor, fwd, min_obs=10)
    assert abs(ic) < 0.25, f"Random IC should be near zero, got {ic:.3f}"


def test_ic_insufficient_data():
    """IC with fewer than min_obs valid pairs should return NaN."""
    factor = pd.Series([1.0, 2.0, 3.0])
    fwd = pd.Series([0.1, 0.2, 0.3])
    ic = compute_ic(factor, fwd, min_obs=10)
    assert np.isnan(ic)


# ---------------------------------------------------------------------------
# compute_ic_series
# ---------------------------------------------------------------------------

def test_ic_series_length():
    """IC series should have one entry per rebalance date with sufficient forward data."""
    prices = make_prices(20, 504)
    from src.factors.momentum import Momentum12_1
    f = Momentum12_1()
    panel = f.compute_panel(prices, freq="ME")
    daily_rets = prices.pct_change().dropna(how="all")
    ic_s = compute_ic_series(panel, daily_rets, horizon_days=21)
    # Should have multiple IC observations
    assert len(ic_s) >= 5


def test_ic_series_bounds():
    """IC values must be in [-1, 1]."""
    prices = make_prices(20, 504)
    from src.factors.momentum import Momentum12_1
    f = Momentum12_1()
    panel = f.compute_panel(prices, freq="ME")
    daily_rets = prices.pct_change().dropna(how="all")
    ic_s = compute_ic_series(panel, daily_rets, horizon_days=21)
    assert (ic_s >= -1.0).all() and (ic_s <= 1.0).all()


# ---------------------------------------------------------------------------
# compute_icir
# ---------------------------------------------------------------------------

def test_icir_positive_ic():
    ic = pd.Series([0.05, 0.06, 0.04, 0.07, 0.05, 0.06])
    icir = compute_icir(ic)
    assert icir > 0


def test_icir_zero_std():
    ic = pd.Series([0.05, 0.05, 0.05])
    icir = compute_icir(ic)
    assert np.isnan(icir)


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

def test_forward_returns_shape():
    prices = make_prices(10, 200)
    rets = prices.pct_change()
    fwd = compute_forward_returns(rets, horizon_days=21)
    assert fwd.shape == rets.shape


def test_forward_returns_no_lookahead():
    """Forward return at date t should use prices after t, not at t."""
    prices = make_prices(5, 100)
    rets = prices.pct_change()
    fwd = compute_forward_returns(rets, horizon_days=5)
    # At last 5 dates, forward return should be NaN (no future data)
    last_5 = fwd.iloc[-5:]
    assert last_5.isna().all().all()


# ---------------------------------------------------------------------------
# compute_ic_decay
# ---------------------------------------------------------------------------

def test_ic_decay_shape():
    prices = make_prices(20, 504)
    from src.factors.momentum import Momentum12_1
    f = Momentum12_1()
    panel = f.compute_panel(prices, freq="ME")
    daily_rets = prices.pct_change().dropna(how="all")
    decay = compute_ic_decay(panel, daily_rets, horizons=[1, 5, 21])
    assert set(decay.index) == {1, 5, 21}
    assert "IC_mean" in decay.columns
    assert "ICIR" in decay.columns
