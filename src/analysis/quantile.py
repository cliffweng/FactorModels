"""Quantile portfolio analysis.

Forms N equal-weight portfolios sorted by factor score at each rebalance date.
Portfolio returns are computed from the rebalance date to the next rebalance date.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.stats import (
    annualize_return, annualize_vol, sharpe_ratio, max_drawdown, summary_stats
)


@dataclass
class QuantileResult:
    factor_name: str
    n_quantiles: int
    portfolio_returns: dict[int, pd.Series]   # {1: daily_returns, ..., N: daily_returns}
    rebal_dates: pd.DatetimeIndex

    @property
    def cumulative(self) -> dict[int, pd.Series]:
        return {q: (1 + r).cumprod() for q, r in self.portfolio_returns.items()}

    @property
    def spread_returns(self) -> pd.Series:
        """Q_top minus Q_bottom daily returns (long top, short bottom)."""
        top = self.portfolio_returns[self.n_quantiles]
        bot = self.portfolio_returns[1]
        return (top - bot).rename("L/S Spread")

    def stats_table(self) -> pd.DataFrame:
        rows = []
        for q, rets in self.portfolio_returns.items():
            label = f"Q{q}" + (" (Top)" if q == self.n_quantiles else " (Bot)" if q == 1 else "")
            rows.append(summary_stats(rets, label))
        rows.append(summary_stats(self.spread_returns, "L/S Spread"))
        return pd.DataFrame(rows).set_index("Name")


def form_quantile_portfolios(
    factor_panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    n_quantiles: int = 5,
    direction: int = 1,
) -> QuantileResult:
    """Form quantile portfolios and compute their daily returns.

    Args:
        factor_panel:  (rebal_dates × tickers) panel of factor scores.
                       Rows are the ENTRY dates; positions are held until the
                       next rebal date.
        daily_returns: (dates × tickers) daily return DataFrame.
        n_quantiles:   Number of portfolios (default 5 = quintiles).
        direction:     +1 → Q5 is best, -1 → Q1 is best.

    Returns:
        QuantileResult with portfolio daily returns.
    """
    # Ensure consistent ticker universe
    common_tickers = factor_panel.columns.intersection(daily_returns.columns).tolist()
    factor_panel = factor_panel[common_tickers]
    daily_returns = daily_returns[common_tickers]

    rebal_dates = factor_panel.index.sort_values()
    portfolio_rets: dict[int, list[pd.Series]] = {q: [] for q in range(1, n_quantiles + 1)}

    for i, entry_date in enumerate(rebal_dates):
        # Determine exit date = next rebalance date (or end of data)
        exit_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_returns.index[-1]

        # Factor scores at entry; shift by +1 day to avoid look-ahead
        # (we use scores known at close of entry_date, applied from next open)
        scores = factor_panel.loc[entry_date].dropna()
        if len(scores) < n_quantiles * 2:
            continue

        # Assign quantile labels 1..N (1 = lowest score)
        quantile_labels = pd.qcut(
            scores.rank(method="first"),
            n_quantiles,
            labels=range(1, n_quantiles + 1),
        )

        # Daily returns for the holding period (entry+1 : exit inclusive)
        period_rets = daily_returns.loc[
            (daily_returns.index > entry_date) & (daily_returns.index <= exit_date)
        ]

        for q in range(1, n_quantiles + 1):
            tickers_in_q = quantile_labels[quantile_labels == q].index.tolist()
            if not tickers_in_q:
                continue
            # Equal-weight portfolio daily return
            q_rets = period_rets[tickers_in_q].mean(axis=1)
            portfolio_rets[q].append(q_rets)

    # Concatenate each portfolio's daily return series
    final = {}
    for q in range(1, n_quantiles + 1):
        if portfolio_rets[q]:
            combined = pd.concat(portfolio_rets[q]).sort_index()
            # If direction is -1, flip: Q1 becomes the "best" portfolio
            if direction == -1:
                actual_q = n_quantiles + 1 - q
                final[actual_q] = combined
            else:
                final[q] = combined
        else:
            final[q] = pd.Series(dtype=float)

    return QuantileResult(
        factor_name="",
        n_quantiles=n_quantiles,
        portfolio_returns=final,
        rebal_dates=rebal_dates,
    )
