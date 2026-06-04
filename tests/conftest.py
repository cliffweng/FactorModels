"""Shared test fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tests.helpers import make_prices, make_prices_with_spy, make_fundamentals


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def prices_with_spy():
    return make_prices_with_spy()


@pytest.fixture
def fundamentals(prices):
    return make_fundamentals(prices.columns.tolist())
