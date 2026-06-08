"""Base class and registry for the pluggable factor system."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import pandas as pd

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, "BaseFactor"] = {}


def register_factor(cls: Type["BaseFactor"]) -> Type["BaseFactor"]:
    """Class decorator: instantiates the factor and adds it to the registry."""
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_registry() -> Dict[str, "BaseFactor"]:
    return dict(_REGISTRY)


def list_factors(category: str | None = None) -> list[str]:
    if category is None:
        return list(_REGISTRY.keys())
    return [n for n, f in _REGISTRY.items() if f.category == category]


def get_factor(name: str) -> "BaseFactor":
    if name not in _REGISTRY:
        raise KeyError(f"Factor '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseFactor(ABC):
    """Abstract base class for all factors.

    Class-level attributes (must be set in subclasses):
        name (str):  Unique identifier used as registry key.
        label (str): Human-readable display name.
        description (str): One-sentence description.
        category (str): "Momentum" | "Risk" | "Value" | "Quality"
        direction (int): +1 if higher score is better, -1 if lower is better.
        requires_fundamentals (bool): True for factors sourced from yfinance.info.
            These factors only support cross-sectional (snapshot) computation,
            not rolling panel computation needed for time-series IC analysis.
    """

    name: str
    label: str
    description: str
    category: str
    direction: int = 1
    requires_fundamentals: bool = False
    requires_edgar: bool = False        # True → needs EDGAR historical data for compute_panel()

    # Optional educational metadata — shown in the Factor Lab "About" panel.
    # Subclasses should override these; empty strings are silently skipped in the UI.
    formula: str = ""         # display formula, e.g. "score = p(t-21)/p(t-252) − 1"
    academic_ref: str = ""    # key citation, e.g. "Jegadeesh & Titman (1993)"
    interpretation: str = ""  # plain-English: what high/low scores mean

    # When False, the factor is excluded from the default active set shown on
    # first visit.  Users must enable it manually in the Factor Library page.
    # Set to False for experimental, niche, or slow-to-compute factors.
    enabled_by_default: bool = True

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        """Cross-sectional factor scores at the most recent date.

        Args:
            prices: Close price DataFrame (dates × tickers). Use the full
                    history available; the factor reads the lookback it needs.
            **kwargs: Optional overrides (e.g., fundamentals DataFrame).

        Returns:
            pd.Series indexed by ticker, float scores (higher = better if direction=+1).
        """

    def compute_panel(
        self,
        prices: pd.DataFrame,
        freq: str = "ME",
        min_periods: int = 252,
        **kwargs,
    ) -> pd.DataFrame:
        """Rolling panel of factor scores (dates × tickers).

        Default implementation raises for fundamental factors. Price-based
        factors should override with vectorized pandas operations.

        Args:
            prices: Full Close price history (dates × tickers).
            freq:   Rebalance frequency — pandas offset alias ('ME', 'W-FRI', etc.)
            min_periods: Minimum trading-day history required before first score.

        Returns:
            pd.DataFrame (rebalance dates × tickers), NaN where score unavailable.
        """
        if self.requires_fundamentals:
            raise NotImplementedError(
                f"Factor '{self.name}' requires fundamentals and does not support "
                "panel computation. Use compute() for a snapshot cross-section."
            )
        raise NotImplementedError(
            f"Factor '{self.name}' has not implemented compute_panel()."
        )

    def winsorize(self, scores: pd.Series, pct: float = 0.01) -> pd.Series:
        """Winsorize factor scores at pct/1-pct quantiles to reduce outlier impact."""
        lo = scores.quantile(pct)
        hi = scores.quantile(1 - pct)
        return scores.clip(lo, hi)

    def z_score(self, scores: pd.Series) -> pd.Series:
        """Standardize factor scores to zero-mean unit-variance cross-sectionally."""
        mu, sigma = scores.mean(), scores.std()
        if sigma == 0:
            return scores * 0
        return (scores - mu) / sigma

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Factor name={self.name!r} category={self.category!r}>"
