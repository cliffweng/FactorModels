# Factor Models Research App

## Overview
Streamlit app demonstrating quant factor model research workflow. Uses yfinance for free data with pickle caching.

## Stack
- Python 3.10+
- Streamlit 1.32+ (multipage)
- Plotly (dark theme)
- yfinance + pickle cache

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure
- `src/data/` — data loading + caching
- `src/factors/` — pluggable factor library (add factors via `@register_factor`)
- `src/analysis/` — IC, quantile, backtest computations
- `src/viz/` — reusable Plotly chart components
- `pages/` — Streamlit pages (auto-discovered)
- `.cache/` — pickle cache (gitignored)

## Adding a New Factor
1. Create or add to a file in `src/factors/`
2. Inherit from `BaseFactor`, set class-level `name`, `label`, `category`
3. Implement `compute(prices)` → `pd.Series`
4. For panel computation: implement `compute_panel(prices)` → `pd.DataFrame`
5. Decorate with `@register_factor`
6. Import in `src/factors/__init__.py`

## Tests
```bash
pytest tests/ -v
```

## Notes
- `.cache/` dir is gitignored; delete it to force data refresh
- Fundamental factors (P/B, ROE, etc.) are snapshot-only from yfinance.info
- Price-based factors support full time-series IC and backtest
- Always run `src/factors/__init__.py` imports to register all factors
