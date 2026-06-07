"""SEC EDGAR XBRL fundamental data loader — free, no API key required.

SEC rate limit: 10 req/s. We throttle to ~3 req/s via sleep inside cached functions
so the sleep only fires on cache miss (real network requests), not on cache hits.

User-Agent header is required by SEC policy:
    https://www.sec.gov/os/accessing-edgar-data

Data returned per ticker (indexed by filing date — point-in-time safe):
    bvps         — book value per share  (StockholdersEquity / SharesOutstanding)
    eps_ttm      — trailing 12-month EPS (quarterly NetIncomeLoss summed × 4)
    roe          — return on equity      (TTM NetIncomeLoss / StockholdersEquity)
    gross_margin — gross profit margin   (TTM GrossProfit / Revenue)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from loguru import logger

from src.data.cache import cached, CACHE_DIR, _cache_key

_EDGAR_BASE = "https://data.sec.gov"
_USER_AGENT = "FactorModels Research App research@factormodels.dev"
_HEADERS    = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# XBRL concepts to try in order (first non-empty wins)
_CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "shares": [
        ("dei",     "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    ],
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ],
    "gross_profit": [
        ("us-gaap", "GrossProfit"),
    ],
    "eps": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasic"),
    ],
}

_INCOME_STMT_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}


# ---------------------------------------------------------------------------
# CIK lookup (30-day cache — ticker changes are rare)
# ---------------------------------------------------------------------------

@cached(ttl_days=30.0)
def _get_cik_map() -> dict[str, str]:
    """Return {TICKER: '0000320193'} for all SEC-registered companies."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": _USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in resp.json().values()
    }


def ticker_to_cik(ticker: str) -> str | None:
    return _get_cik_map().get(ticker.upper())


# ---------------------------------------------------------------------------
# Company facts — one request per company, 7-day cache
# ---------------------------------------------------------------------------

@cached(ttl_days=7.0)
def _fetch_company_facts(cik10: str) -> dict:
    """Fetch all XBRL facts for one company. Sleep fires only on cache miss."""
    time.sleep(0.35)  # ~3 req/s rate-limit guard
    url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik10}.json"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            logger.warning(f"EDGAR: CIK {cik10} not found")
            return {}
        resp.raise_for_status()
        logger.info(f"EDGAR: fetched CIK {cik10}")
        return resp.json()
    except Exception as e:
        logger.warning(f"EDGAR: failed CIK {cik10}: {type(e).__name__}: {e}")
        return {}


# ---------------------------------------------------------------------------
# XBRL extraction helpers
# ---------------------------------------------------------------------------

def _get_records(facts: dict, alternatives: list[tuple[str, str]]) -> list[dict]:
    """Return raw XBRL unit records for the first matching concept."""
    for taxonomy, concept in alternatives:
        units = (
            facts.get("facts", {})
            .get(taxonomy, {})
            .get(concept, {})
            .get("units", {})
        )
        for unit_key, records in units.items():
            if unit_key in ("USD", "shares", "USD/shares", "pure") and records:
                return records
    return []


def _to_df(records: list[dict]) -> pd.DataFrame:
    """Normalise raw records to a clean DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "filed" not in df.columns or "val" not in df.columns:
        return pd.DataFrame()
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["end"]   = pd.to_datetime(df.get("end",   pd.Series(dtype="datetime64[ns]")), errors="coerce")
    df["start"] = pd.to_datetime(df.get("start", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    df["val"]   = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna(subset=["filed", "end", "val"])


def _latest_instant(df: pd.DataFrame) -> pd.Series:
    """Balance-sheet items: most recent value per period end, indexed by filing date.

    Deduplication is two-pass to guarantee a unique filed-date index:
      1. per period-end: keep the latest filing for each fiscal period
      2. per filing-date: if two different periods were filed on the same date
         (e.g. a 10-Q/A amendment co-filed with a new 10-Q), keep the row
         with the most recent period end so the index stays unique.
    """
    if df.empty:
        return pd.Series(dtype=float)
    keep = df[df.get("form", pd.Series()).isin(_INCOME_STMT_FORMS)] if "form" in df.columns else df
    if keep.empty:
        keep = df
    keep = keep.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    keep = keep.sort_values(["filed", "end"]).drop_duplicates(subset=["filed"], keep="last")
    return keep.set_index("filed")["val"].sort_index()


def _ttm_from_quarters(df: pd.DataFrame) -> pd.Series:
    """Income-statement items: TTM = sum of last 4 non-overlapping ~90-day quarters.

    Falls back to the most recent annual record if we can't assemble 4 quarters.
    """
    if df.empty:
        return pd.Series(dtype=float)

    if "form" in df.columns:
        df = df[df["form"].isin(_INCOME_STMT_FORMS)]
    if df.empty:
        return pd.Series(dtype=float)

    # Quarterly records: period duration 60-100 days
    has_start = "start" in df.columns and df["start"].notna().any()
    if has_start:
        df = df.dropna(subset=["start"])
        df["dur"] = (df["end"] - df["start"]).dt.days
        quarterly = df[(df["dur"] >= 60) & (df["dur"] <= 100)].copy()
    else:
        quarterly = pd.DataFrame()

    if quarterly.empty:
        # Fallback: use annual records directly
        if has_start:
            annual = df[(df["dur"] >= 330) & (df["dur"] <= 400)].copy()
        else:
            annual = df.copy()
        if annual.empty:
            return pd.Series(dtype=float)
        annual = annual.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        return annual.set_index("filed")["val"].sort_index()

    quarterly = quarterly.sort_values("filed").drop_duplicates(subset=["end"], keep="last")

    results: dict = {}
    for fd in sorted(quarterly["filed"].unique()):
        avail = quarterly[quarterly["filed"] <= fd].sort_values("end")
        avail = avail.drop_duplicates(subset=["end"], keep="last").tail(4)
        if len(avail) < 4:
            continue
        span = (avail["end"].max() - avail["end"].min()).days
        if span < 240:  # need ~9 months of history to sum 4 quarters
            continue
        results[fd] = avail["val"].sum()

    return pd.Series(results).sort_index()


# ---------------------------------------------------------------------------
# Per-ticker fundamental series computation (cached at 7-day TTL)
# ---------------------------------------------------------------------------

@cached(ttl_days=7.0)
def _compute_ticker_fundamentals(ticker: str) -> dict[str, pd.Series]:
    """Compute all fundamental series for one ticker from EDGAR XBRL data.

    Returns {field: pd.Series(index=filing_date, val=float)}
    Fields: bvps, eps_ttm, roe, gross_margin.
    Empty Series for any field that can't be computed.
    """
    cik = ticker_to_cik(ticker)
    if cik is None:
        logger.warning(f"EDGAR: no CIK mapping for {ticker}")
        return {}

    facts = _fetch_company_facts(cik)
    if not facts:
        return {}

    # --- Balance sheet (instantaneous) ---
    equity_s = _latest_instant(_to_df(_get_records(facts, _CONCEPTS["equity"])))
    shares_s  = _latest_instant(_to_df(_get_records(facts, _CONCEPTS["shares"])))

    bvps_s = pd.Series(dtype=float)
    if not equity_s.empty and not shares_s.empty:
        idx = equity_s.index.union(shares_s.index).sort_values()
        eq = equity_s.reindex(idx).ffill()
        sh = shares_s.reindex(idx).ffill()
        bvps_raw = (eq / sh.replace(0, np.nan)).dropna()
        bvps_s = bvps_raw[bvps_raw > 0]

    # --- Income statement (TTM) ---
    ni_ttm  = _ttm_from_quarters(_to_df(_get_records(facts, _CONCEPTS["net_income"])))
    rev_ttm = _ttm_from_quarters(_to_df(_get_records(facts, _CONCEPTS["revenue"])))
    gp_ttm  = _ttm_from_quarters(_to_df(_get_records(facts, _CONCEPTS["gross_profit"])))

    # EPS TTM: prefer reported diluted EPS, fall back to net_income / shares
    eps_raw = _to_df(_get_records(facts, _CONCEPTS["eps"]))
    eps_ttm = _ttm_from_quarters(eps_raw)
    if eps_ttm.empty and not ni_ttm.empty and not shares_s.empty:
        sh_aligned = shares_s.reindex(ni_ttm.index.union(shares_s.index)).ffill().reindex(ni_ttm.index)
        eps_ttm = (ni_ttm / sh_aligned.replace(0, np.nan)).dropna()

    # ROE = net_income TTM / equity
    roe_s = pd.Series(dtype=float)
    if not ni_ttm.empty and not equity_s.empty:
        idx = ni_ttm.index.union(equity_s.index).sort_values()
        eq = equity_s.reindex(idx).ffill().reindex(ni_ttm.index)
        roe_s = (ni_ttm / eq.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()

    # Gross margin = gross_profit TTM / revenue TTM
    gm_s = pd.Series(dtype=float)
    if not gp_ttm.empty and not rev_ttm.empty:
        idx = gp_ttm.index.union(rev_ttm.index).sort_values()
        rev = rev_ttm.reindex(idx).ffill().reindex(gp_ttm.index)
        gm_raw = (gp_ttm / rev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        gm_s = gm_raw.clip(-1.0, 1.0)

    result: dict[str, pd.Series] = {}
    if not bvps_s.empty:   result["bvps"] = bvps_s
    if not eps_ttm.empty:  result["eps_ttm"] = eps_ttm
    if not roe_s.empty:    result["roe"] = roe_s
    if not gm_s.empty:     result["gross_margin"] = gm_s

    if result:
        sample = next(iter(result.values()))
        logger.info(
            f"EDGAR: {ticker} → {list(result.keys())}, "
            f"{len(sample)} periods, "
            f"{sample.index.min().date()} → {sample.index.max().date()}"
        )
    return result


# ---------------------------------------------------------------------------
# Batch panel assembly
# ---------------------------------------------------------------------------

def get_edgar_fundamentals_panel(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Assemble historical fundamental panels for all tickers.

    Returns dict mapping field → DataFrame(filing_dates × tickers):
        "bvps"         — book value per share (USD)
        "eps_ttm"      — trailing 12-month EPS (USD/share)
        "roe"          — return on equity (fraction)
        "gross_margin" — gross profit margin (fraction)

    Caching (two layers):
      1. Per-ticker pickle (7-day TTL on _compute_ticker_fundamentals).
         Survives Streamlit restarts; sleep in _fetch_company_facts fires only
         on cache miss so cached runs complete in milliseconds.
      2. @st.cache_data (24-hour in-memory TTL) in the calling page.
         Prevents even pickle reads during an active session.
    """
    per_ticker: dict[str, dict[str, pd.Series]] = {}
    for ticker in tickers:
        try:
            fields = _compute_ticker_fundamentals(ticker)
            if fields:
                per_ticker[ticker] = fields
        except Exception as e:
            logger.warning(f"EDGAR: skipping {ticker} — {type(e).__name__}: {e}")

    if not per_ticker:
        return {}

    result: dict[str, pd.DataFrame] = {}
    for field in ("bvps", "eps_ttm", "roe", "gross_margin"):
        series = {t: d[field] for t, d in per_ticker.items() if field in d}
        if not series:
            continue
        wide = pd.DataFrame(series).sort_index()
        # Safety: deduplicate index in case any individual series still has duplicates
        if not wide.index.is_unique:
            wide = wide[~wide.index.duplicated(keep="last")]
        result[field] = wide

    return result


# ---------------------------------------------------------------------------
# Cache introspection
# ---------------------------------------------------------------------------

def edgar_cache_info(tickers: tuple[str, ...]) -> dict:
    """Return cache status for each ticker without making any API calls."""
    import time as _time

    cik_map = _get_cik_map()
    cached_tickers, missing_tickers, ages = [], [], {}

    for ticker in tickers:
        key  = _cache_key("_compute_ticker_fundamentals", (ticker,), {})
        path = CACHE_DIR / f"_compute_ticker_fundamentals_{key}.pkl"
        if path.exists():
            age_days = (_time.time() - path.stat().st_mtime) / 86_400
            if age_days < 7.0:
                cached_tickers.append(ticker)
                ages[ticker] = round(age_days, 1)
            else:
                missing_tickers.append(ticker)
        else:
            if cik_map.get(ticker.upper()) is None:
                missing_tickers.append(f"{ticker}(no CIK)")
            else:
                missing_tickers.append(ticker)

    return {"cached": cached_tickers, "missing": missing_tickers, "ages_days": ages}
