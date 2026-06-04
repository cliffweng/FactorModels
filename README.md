# Factor Models Research Platform

An interactive Streamlit app that demonstrates how a quant research team investigates equity factor models — from raw price data through to backtested long-short portfolios. Designed to be modular and pluggable so new factors can be added without touching the app layer.

## Quick Start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt  # macOS/Linux

.venv/Scripts/streamlit run app.py
```

Data is fetched from [yfinance](https://github.com/ranaroussi/yfinance) on first load and cached to `.cache/` as pickle files. Subsequent page loads are instant.

---

## Research Workflow

The app follows the standard quant factor research pipeline:

```
Universe → Price Data → Factor Scores → IC Analysis → Quantile Portfolios → Backtest
```

Each stage corresponds to a page in the app.

---

## Pages

### Home (`app.py`)
Landing page. Shows the full factor registry (all registered factors with their category, direction, and whether they support time-series analysis), a methodology reference, and cache status in the sidebar with a one-click clear button.

### 1 — Universe Explorer
Explore the 75-stock investment universe across 11 GICS sectors.

- **Sector Breakdown** — pie chart of sector composition; box plot of annualised volatility per sector
- **Price Performance** — indexed price chart (base = 100) for all tickers or sector medians, with SPY overlay
- **Correlation Matrix** — return correlation heatmap with tickers sorted by sector, revealing intra-sector clustering
- **Data Table** — YTD return, annualised return, annualised vol, latest price for every stock

The universe is filterable by sector via the sidebar. SPY is always downloaded as a benchmark but excluded from cross-sectional analysis.

### 2 — Factor Lab
Compute and inspect cross-sectional factor scores for any registered factor as of the most recent date.

- **Factor Scores** — horizontal bar chart sorted by score, coloured by sector
- **Distribution** — box plots per sector showing how factor exposure varies across the universe
- **Score vs. Return** — scatter of factor score against forward return (configurable horizon: 1d to 3m), with a trend line; confirms or refutes the factor's cross-sectional predictive relationship

Options: winsorisation (1–99%) and z-score standardisation. The Top 10 / Bottom 10 tables show the stocks with the highest and lowest factor exposure.

### 3 — IC Analysis
The core quantitative evaluation of a factor's predictive power over time.

**What IC measures:** At each rebalance date *t*, the Information Coefficient is the Spearman rank correlation between the factor scores and the realised forward returns from *t* to *t + horizon*. IC ∈ [-1, 1]. An IC consistently above 0 means the factor reliably ranks winners above losers.

- **IC Time Series** — bar chart coloured green/red by sign; dotted line = 12-period rolling mean IC; annotated with mean IC and ICIR
- **IC Distribution** — histogram of IC values with mean line; a good factor's distribution is shifted right of zero
- **IC Decay** — bar chart of mean IC at horizons from 1 day to 6 months; shows how quickly the signal degrades. Momentum factors typically peak at 1–3 months then reverse; reversal factors peak at 1 week
- **Cumulative IC** — running sum of IC over time; a rising trend confirms persistent efficacy; flat or declining sections indicate regime changes

Key statistics displayed: Mean IC, ICIR (= mean IC / std IC), IC Std, % positive periods.

A collapsible **Factor Panel Heatmap** shows the raw factor score matrix (last 24 rebalance periods × all tickers) as a red-green heatmap, giving an intuitive view of how scores evolve over time.

### 4 — Backtest
Evaluate factor strength through simulated trading.

**How portfolios are formed:** At each rebalance date, stocks are sorted by factor score and divided into *N* equal-size buckets (default: 5 quintiles). Each bucket is an equal-weight portfolio held until the next rebalance date. The `direction` attribute of each factor determines which quintile is "top" — for low-volatility factors `direction = -1` so Q5 means lowest volatility.

- **Quintile Fan** — five equity curves from Q1 (red) to Q5 (blue); clear separation confirms factor efficacy
- **L/S Strategy** — long top quintile, short bottom quintile (zero net investment); SPY shown as reference
- **Drawdown** — filled area chart of L/S peak-to-trough drawdown over time
- **Annual Returns** — grouped bar chart comparing L/S vs SPY year by year

Performance table: annualised return, annualised vol, Sharpe ratio, max drawdown, Calmar ratio for each quintile and the spread.

Sidebar controls: factor, number of portfolios (3–10), rebalance frequency (monthly/weekly/quarterly), round-trip transaction cost in bps, lookback period.

### 5 — Factor Correlation
Understand how diversified the factor library is.

- **Correlation Matrix** — Spearman rank correlation between all factor cross-sections (snapshot); annotated heatmap with RdBu colour scale
- **Factor Scatter** — user-selectable X/Y factor pair, scatter coloured by sector with trend line; highlights whether two factors are driven by the same stocks
- **Panel Correlation** — time-averaged correlation computed from monthly cross-sections rather than the single-date snapshot, giving a more stable estimate

High correlation (|ρ| > 0.7) flags redundancy; low correlation signals genuine diversification across factors.

---

## Architecture

```
FactorModels/
├── app.py                        # Home page; triggers factor registration on import
├── pages/
│   ├── 1_Universe_Explorer.py
│   ├── 2_Factor_Lab.py
│   ├── 3_IC_Analysis.py
│   ├── 4_Backtest.py
│   └── 5_Factor_Correlation.py
│
├── src/
│   ├── data/
│   │   ├── cache.py              # @cached(ttl_days) decorator — pickle to .cache/
│   │   ├── loader.py             # get_prices(), get_fundamentals() via yfinance
│   │   └── universe.py           # 75-stock SECTOR_MAP, UNIVERSE list, TICKER_SECTOR map
│   │
│   ├── factors/
│   │   ├── base.py               # BaseFactor ABC + @register_factor + global registry
│   │   ├── momentum.py           # Momentum12_1, Momentum6_1, ShortTermReversal
│   │   ├── risk.py               # RealizedVol60, Beta252
│   │   ├── value.py              # FiftyTwoWeekHighRatio, PriceToBook, PriceToEarnings
│   │   └── quality.py            # ROEFactor, GrossMarginFactor
│   │
│   ├── analysis/
│   │   ├── stats.py              # annualize_return, sharpe_ratio, max_drawdown, etc.
│   │   ├── ic.py                 # compute_ic_series, compute_ic_decay, compute_icir
│   │   ├── quantile.py           # form_quantile_portfolios → QuantileResult
│   │   └── backtest.py           # run_backtest → BacktestResult
│   │
│   └── viz/
│       ├── theme.py              # DARK_LAYOUT, colour palettes, apply_dark()
│       ├── ic_charts.py          # plot_ic_bar, plot_ic_decay, plot_cumulative_ic
│       ├── portfolio_charts.py   # plot_quintile_fans, plot_cumulative_ls, plot_drawdown
│       └── factor_charts.py      # plot_factor_bar, plot_factor_scatter, plot_correlation_matrix
│
└── tests/
    ├── helpers.py                # make_prices(), make_fundamentals() — no yfinance
    ├── conftest.py               # pytest fixtures wrapping helpers
    ├── test_factors.py           # Factor compute/panel shape, monotonicity, edge cases
    ├── test_ic.py                # IC perfect rank = 1.0, random ≈ 0, bounds, decay shape
    └── test_quantile.py          # Portfolio formation, monotonicity with synthetic signal
```

---

## The Factor System

### Adding a New Factor

Every factor is a class that inherits `BaseFactor` and is decorated with `@register_factor`. The decorator instantiates the class and inserts it into the global registry. Every page in the app queries the registry at runtime — no changes needed outside the factor file itself.

**Minimal example:**

```python
# src/factors/my_factors.py
from src.factors.base import BaseFactor, register_factor
import pandas as pd

@register_factor
class EarningsYield(BaseFactor):
    name        = "E_Y"
    label       = "Earnings Yield"
    description = "Trailing 12-month EPS / price — higher is cheaper"
    category    = "Value"
    direction   = 1

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        fundamentals = kwargs.get("fundamentals")
        pe = fundamentals["pe"].dropna()
        return (1.0 / pe[pe > 0]).rename("E_Y")

    def compute_panel(self, prices, freq="ME", min_periods=252):
        # price-based proxy or raise NotImplementedError for snapshot-only
        ...
```

Then import it in `src/factors/__init__.py` and it appears in all pages automatically.

### Factor Types

| Type | `requires_fundamentals` | `compute()` | `compute_panel()` | IC Analysis | Backtest |
|---|---|---|---|---|---|
| **Price-based** | `False` | Latest window of price history | Rolling panel (vectorised pandas) | Full time series | Supported |
| **Fundamental** | `True` | Current `yfinance.info` snapshot | Not supported | Single-period IC only | Not supported |

Price-based factors use vectorised `shift()` and `rolling()` operations across the entire price DataFrame — no Python loops over dates. Fundamental factors call `yf.Ticker(t).info` per ticker, cached for 24 hours.

---

## Data & Caching

### Price Data
`yfinance.download()` fetches adjusted close prices for the full universe in a single bulk request. Stored as a pickled DataFrame at `.cache/get_prices_<hash>.pkl`.

### Fundamental Data
`yf.Ticker(t).info` is called per ticker and cached together for the universe. Re-fetched after 24 hours. Fields used: `priceToBook`, `trailingPE`, `returnOnEquity`, `grossMargins`, `beta`, `sector`, `marketCap`.

### Cache Behaviour
`@cached(ttl_days=1.0)` wraps any data-loading function. On each call:
1. Computes a SHA-1 key from the function name + arguments
2. If the `.pkl` file exists and is younger than the TTL, returns the cached result immediately
3. Otherwise calls the real function, writes the result to disk, then returns it

A `force_refresh=True` kwarg (wired to the "Refresh Data" button in each page's sidebar) bypasses the TTL check. Cache files can be cleared via the "Clear All Cache" button on the Home page or by deleting `.cache/` manually.

---

## Analysis Details

### IC Computation (no look-ahead bias)

```
forward_return[t] = cum_return[t + horizon] / cum_return[t] - 1
IC[t] = spearmanr( factor_scores[t],  forward_return[t] )
```

Factor scores at date *t* predict returns **starting from t**, not including the same day's return. The cumulative return series is shifted forward by the horizon so that each row's forward return uses only future prices. Scores and returns are inner-joined on the rebalance date index before computing Spearman correlation.

### Quantile Portfolio Formation

At each rebalance date:
1. Drop tickers with missing factor scores
2. Rank remaining tickers and assign to *N* equal buckets using `pd.qcut`
3. Hold an equal-weight portfolio of each bucket until the next rebalance date
4. Portfolio daily return = mean of constituent daily returns over the holding period

For factors where `direction = -1` (lower score is better, e.g. volatility), the bucket labels are flipped so Q5 always represents the "best" bucket in the fan chart.

### Long-Short Backtest

- **Long leg**: Q5 (top bucket)
- **Short leg**: Q1 (bottom bucket)
- **Net return**: `long_return - short_return - transaction_cost`
- Transaction cost is applied on every rebalance day as a flat deduction in bps

---

## Tests

```bash
.venv/Scripts/pytest tests/ -v
```

37 tests covering:
- Factor registry population and attribute validation
- `compute()` shape, NaN-free output, edge cases (insufficient history, missing SPY, no fundamentals)
- Monotonicity: a stock with a higher 12-month return must rank higher on Momentum 12-1
- IC = 1.0 for a perfect-rank factor; IC ≈ 0 for random noise
- Forward return alignment: last *horizon* dates must be NaN (no future data)
- IC decay shape and bounds
- Quantile portfolio monotonicity: synthetic signal where Q5 must outperform Q1

All tests use synthetic price data generated in `tests/helpers.py` — no network calls.
