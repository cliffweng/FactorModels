"""Data loading layer: prices and fundamentals via yfinance with pickle cache."""
from __future__ import annotations

import warnings
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.cache import cached


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

@cached(ttl_days=1.0)
def get_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices. Returns DataFrame (dates × tickers).

    Args:
        tickers: sorted tuple of ticker symbols (tuple for cache-key stability)
        start: YYYY-MM-DD
        end:   YYYY-MM-DD

    Returns:
        DataFrame of adjusted close prices, columns = tickers, index = DatetimeIndex.
    """
    logger.info(f"Downloading prices for {len(tickers)} tickers [{start} → {end}]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if raw.empty:
        raise ValueError("yfinance returned empty DataFrame — check tickers / date range.")

    # Handle single vs multi-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker: columns are OHLCV field names
        prices = raw[["Close"]].copy()
        prices.columns = list(tickers)

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    logger.info(f"Prices loaded: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    return prices


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price DataFrame."""
    return prices.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Fundamental data (snapshot — not time-series)
# ---------------------------------------------------------------------------

@cached(ttl_days=1.0)
def get_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch point-in-time fundamental data from yfinance.info.

    NOTE: This returns CURRENT values only — no historical fundamentals.
    Suitable for cross-sectional factor construction, not time-series IC.

    Returns:
        DataFrame indexed by ticker with columns:
        sector, industry, market_cap, pb, pe, forward_pe, roe,
        gross_margins, beta, peg, dividend_yield
    """
    logger.info(f"Fetching fundamentals for {len(tickers)} tickers")
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap"),
                    "pb": info.get("priceToBook"),
                    "pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "roe": info.get("returnOnEquity"),
                    "gross_margins": info.get("grossMargins"),
                    "beta": info.get("beta"),
                    "peg": info.get("pegRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "short_ratio": info.get("shortRatio"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    logger.info(f"Fundamentals loaded for {len(df)} tickers")
    return df
