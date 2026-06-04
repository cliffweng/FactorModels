"""Pickle-based disk cache with TTL for yfinance data."""
import pickle
import hashlib
import time
import json
from pathlib import Path
from functools import wraps

CACHE_DIR = Path(".cache")


def _cache_key(func_name: str, args, kwargs) -> str:
    payload = json.dumps(
        {"f": func_name, "a": str(args), "k": str(sorted(kwargs.items()))},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def cached(ttl_days: float = 1.0):
    """Decorator: cache function result to a pickle file. Re-fetches after TTL."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            force = kwargs.pop("force_refresh", False)
            CACHE_DIR.mkdir(exist_ok=True)
            key = _cache_key(func.__name__, args, kwargs)
            path = CACHE_DIR / f"{func.__name__}_{key}.pkl"

            if not force and path.exists():
                age = time.time() - path.stat().st_mtime
                if age < ttl_days * 86_400:
                    try:
                        with open(path, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        pass  # corrupt cache — fall through to re-fetch

            result = func(*args, **kwargs)
            with open(path, "wb") as f:
                pickle.dump(result, f)
            return result

        wrapper._cache_dir = CACHE_DIR
        wrapper._func_name = func.__name__
        return wrapper

    return decorator


def clear_cache(prefix: str | None = None) -> int:
    """Delete cache files. Pass prefix to target a specific function."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for p in CACHE_DIR.glob("*.pkl"):
        if prefix is None or p.name.startswith(prefix + "_"):
            p.unlink()
            count += 1
    return count


def list_cache() -> list[dict]:
    """Return metadata about all cached files, newest first."""
    if not CACHE_DIR.exists():
        return []
    result = []
    for p in sorted(CACHE_DIR.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        result.append(
            {
                "file": p.name,
                "size_kb": round(st.st_size / 1024, 1),
                "age_h": round((time.time() - st.st_mtime) / 3600, 1),
            }
        )
    return result
