"""Shared helper functions for tests (not pytest fixtures)."""
import numpy as np
import pandas as pd


def make_prices(n_tickers: int = 20, n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    log_returns = rng.normal(0.0003, 0.015, size=(n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    tickers = [f"S{i:03d}" for i in range(n_tickers)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_prices_with_spy(n_tickers: int = 20, n_days: int = 504) -> pd.DataFrame:
    df = make_prices(n_tickers, n_days)
    spy_rets = df.pct_change().mean(axis=1).fillna(0)
    df["SPY"] = 300 * (1 + spy_rets).cumprod()
    return df


def make_fundamentals(tickers: list) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(tickers)
    return pd.DataFrame(
        {
            "sector": ["Technology"] * n,
            "pb": rng.uniform(1.0, 20.0, n),
            "pe": rng.uniform(10.0, 50.0, n),
            "roe": rng.uniform(0.05, 0.40, n),
            "gross_margins": rng.uniform(0.20, 0.80, n),
            "beta": rng.uniform(0.5, 2.0, n),
        },
        index=tickers,
    )
