"""CompositeModel: weighted combination of multiple BaseFactor instances.

Design notes:
- NOT a BaseFactor subclass — it's an aggregator that mirrors the panel API.
- Each factor is direction-adjusted and cross-sectionally z-scored before combining,
  so CompositeModel.direction is always +1.
- NaN handling: per ticker, only non-NaN factors contribute; the weight is renormalized
  over that ticker's available factors. A ticker is NaN only if ALL factors are NaN.
- Negative weights are rejected — use factor.direction=-1 to invert a signal.
"""
from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.factors.base import BaseFactor


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: standardise each row independently."""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sigma, axis=0).fillna(0)


class CompositeModel:
    """Weighted composite of multiple factors.

    Args:
        factors:  List of BaseFactor instances (price-based for panel support).
        weights:  Raw non-negative weights, one per factor (will be normalised to sum=1).
        name:     Display name for the composite.
    """

    direction: int = 1  # always +1; direction is absorbed during z-scoring

    def __init__(
        self,
        factors: list[BaseFactor],
        weights: list[float],
        name: str = "Composite",
    ) -> None:
        if len(factors) != len(weights):
            raise ValueError("factors and weights must have the same length.")
        if any(w < 0 for w in weights):
            raise ValueError(
                "Negative weights are not allowed. Set factor.direction=-1 to invert a signal."
            )
        self.factors = factors
        self._raw_weights = list(weights)
        self.name = name
        self.label = name
        self.category = "Composite"

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    @property
    def normalized_weights(self) -> list[float]:
        """Weights normalised to sum to 1."""
        total = sum(self._raw_weights)
        if total < 1e-12:
            n = len(self._raw_weights)
            return [1.0 / n] * n
        return [w / total for w in self._raw_weights]

    def weights_dict(self) -> dict[str, float]:
        """Mapping factor.name → normalised weight."""
        return {f.name: w for f, w in zip(self.factors, self.normalized_weights)}

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_scores(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional composite score at the most recent date in `prices`.

        Each factor is direction-adjusted then z-scored before weighting.
        Tickers with no factor coverage are dropped from the result.
        """
        weights = self.normalized_weights
        # Accumulate weighted z-scores and total weight per ticker
        numerator: pd.Series | None = None
        denominator: pd.Series | None = None

        for factor, w in zip(self.factors, weights):
            if w < 1e-12:
                continue
            try:
                raw = factor.compute(prices, **kwargs)
            except Exception:
                continue
            if raw.empty:
                continue
            # Direction-adjust
            raw = raw * factor.direction
            # Cross-sectional z-score
            mu, sigma = raw.mean(), raw.std()
            z = (raw - mu) / sigma if sigma > 1e-12 else raw * 0

            wz = z * w
            wt = pd.Series(w, index=z.index)

            numerator = wz if numerator is None else numerator.add(wz, fill_value=0)
            denominator = wt if denominator is None else denominator.add(wt, fill_value=0)

        if numerator is None or denominator is None:
            return pd.Series(dtype=float)

        # Renormalise per ticker so NaN factors don't dilute the score
        composite = numerator / denominator
        return composite.dropna()

    def compute_panel(
        self,
        prices: pd.DataFrame,
        freq: str = "ME",
        min_periods: int = 252,
        **kwargs,
    ) -> pd.DataFrame:
        """Rolling panel of composite scores (rebal_dates × tickers).

        Price-based factors are always included. FMP-backed factors (requires_edgar=True)
        are included when ``fmp_panel`` is present in kwargs; otherwise skipped.
        Factors with requires_fundamentals=True (yfinance snapshot only) are always skipped.

        Raises ValueError if no factors support panel computation.
        """
        weights = self.normalized_weights
        z_panels: list[tuple[pd.DataFrame, float]] = []
        has_fmp = "edgar_panel" in kwargs and kwargs["edgar_panel"]

        unsupported = []
        for factor, w in zip(self.factors, weights):
            if w < 1e-12:
                continue
            if factor.requires_fundamentals:
                unsupported.append(factor.label)
                continue
            if getattr(factor, "requires_edgar", False) and not has_fmp:
                unsupported.append(factor.label)
                continue
            try:
                panel = factor.compute_panel(prices, freq=freq, min_periods=min_periods, **kwargs)
            except Exception:
                unsupported.append(factor.label)
                continue
            if panel.empty:
                continue
            # Direction-adjust before z-scoring
            panel = panel * factor.direction
            z_panel = _zscore_rows(panel)
            z_panels.append((z_panel, w))

        if not z_panels:
            msg = "No price-based factors available for panel computation."
            if unsupported:
                msg += f" Unsupported: {unsupported}"
            raise ValueError(msg)

        # Align all panels to common dates × tickers
        common_idx = reduce(lambda a, b: a.intersection(b), [p.index for p, _ in z_panels])
        common_cols = reduce(lambda a, b: a.intersection(b), [p.columns for p, _ in z_panels])

        if common_idx.empty or common_cols.empty:
            return pd.DataFrame()

        # Coverage-weighted combination: renormalise per cell over non-NaN factors
        num = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        denom = pd.DataFrame(0.0, index=common_idx, columns=common_cols)

        for z_panel, w in z_panels:
            aligned = z_panel.reindex(index=common_idx, columns=common_cols)
            mask = aligned.notna().astype(float)
            num += aligned.fillna(0) * w
            denom += mask * w

        denom = denom.replace(0, np.nan)
        return (num / denom).where(denom.notna())

    # ------------------------------------------------------------------
    # Contribution decomposition (for stacked bar chart)
    # ------------------------------------------------------------------

    def factor_contributions(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Per-ticker weighted z-score contribution from each factor.

        Returns:
            DataFrame(index=tickers, columns=factor labels) where each cell
            is w_i * z_i(ticker). Rows sum to the composite score.
            Tickers sorted by composite score ascending (for bar chart).
        """
        weights = self.normalized_weights
        parts: dict[str, pd.Series] = {}

        for factor, w in zip(self.factors, weights):
            if w < 1e-12:
                continue
            try:
                raw = factor.compute(prices, **kwargs)
            except Exception:
                continue
            if raw.empty:
                continue
            raw = raw * factor.direction
            mu, sigma = raw.mean(), raw.std()
            z = (raw - mu) / sigma if sigma > 1e-12 else raw * 0
            parts[factor.label] = z * w

        if not parts:
            return pd.DataFrame()

        contrib_df = pd.DataFrame(parts)
        # Sort by composite score ascending so the bar chart reads bottom-to-top
        composite = contrib_df.sum(axis=1)
        return contrib_df.loc[composite.sort_values().index]
