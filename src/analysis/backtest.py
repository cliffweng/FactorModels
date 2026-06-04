"""Long-short factor backtest engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.quantile import QuantileResult, form_quantile_portfolios
from src.analysis.stats import (
    annualize_return, annualize_vol, sharpe_ratio,
    max_drawdown, calmar_ratio, drawdown_series, summary_stats
)


@dataclass
class BacktestResult:
    factor_name: str
    long_returns: pd.Series       # daily returns of long leg
    short_returns: pd.Series      # daily returns of short leg
    ls_returns: pd.Series         # long minus short (net zero investment)
    benchmark_returns: pd.Series  # SPY daily returns over same period

    @property
    def cumulative_ls(self) -> pd.Series:
        return (1 + self.ls_returns).cumprod()

    @property
    def cumulative_long(self) -> pd.Series:
        return (1 + self.long_returns).cumprod()

    @property
    def cumulative_benchmark(self) -> pd.Series:
        return (1 + self.benchmark_returns.reindex(self.ls_returns.index).fillna(0)).cumprod()

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.cumulative_ls)

    def stats(self) -> pd.DataFrame:
        rows = [
            summary_stats(self.ls_returns, "L/S Strategy"),
            summary_stats(self.long_returns, "Long Leg"),
            summary_stats(self.short_returns, "Short Leg (raw)"),
        ]
        if len(self.benchmark_returns) > 0:
            bm = self.benchmark_returns.reindex(self.ls_returns.index).fillna(0)
            rows.append(summary_stats(bm, "SPY Benchmark"))
        return pd.DataFrame(rows).set_index("Name")

    def annual_returns(self) -> pd.DataFrame:
        """Annual L/S returns by calendar year."""
        ls_annual = (1 + self.ls_returns).resample("YE").prod() - 1
        bm = self.benchmark_returns.reindex(self.ls_returns.index).fillna(0)
        bm_annual = (1 + bm).resample("YE").prod() - 1
        df = pd.DataFrame({
            "L/S Strategy": ls_annual,
            "SPY": bm_annual,
        })
        df.index = df.index.year
        return df


def run_backtest(
    factor_panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    n_quantiles: int = 5,
    direction: int = 1,
    transaction_cost_bps: float = 10.0,
) -> BacktestResult:
    """Run a long-short factor backtest.

    Args:
        factor_panel:          (rebal_dates × tickers) factor scores.
        daily_returns:         (dates × tickers) daily returns including 'SPY'.
        n_quantiles:           Portfolio buckets (default 5).
        direction:             +1 or -1 as defined by the factor.
        transaction_cost_bps:  Round-trip cost in bps applied at each rebalance.

    Returns:
        BacktestResult with daily returns for each leg.
    """
    # Strip SPY from cross-sectional analysis
    stock_cols = [c for c in daily_returns.columns if c != "SPY"]
    stock_returns = daily_returns[stock_cols]
    spy_returns = daily_returns["SPY"] if "SPY" in daily_returns.columns else pd.Series(dtype=float)

    result = form_quantile_portfolios(factor_panel, stock_returns, n_quantiles, direction)

    top_q = result.portfolio_returns[n_quantiles]
    bot_q = result.portfolio_returns[1]

    # Align indices
    idx = top_q.index.union(bot_q.index).sort_values()
    top_q = top_q.reindex(idx).fillna(0)
    bot_q = bot_q.reindex(idx).fillna(0)

    # Transaction cost: applied on rebalance days
    tc = transaction_cost_bps / 10_000
    rebal_mask = pd.Series(False, index=idx)
    for d in result.rebal_dates:
        # Find next trading day at or after rebal date
        nearest = idx[idx >= d]
        if len(nearest):
            rebal_mask[nearest[0]] = True

    cost_series = rebal_mask.astype(float) * tc

    ls_raw = top_q - bot_q
    ls_net = ls_raw - cost_series.reindex(ls_raw.index).fillna(0)

    return BacktestResult(
        factor_name=result.factor_name,
        long_returns=top_q,
        short_returns=bot_q,
        ls_returns=ls_net,
        benchmark_returns=spy_returns,
    )
