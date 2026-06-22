"""
Data layer.

Pulls historical prices from Yahoo Finance via yfinance, caches them to
disk (so reruns are fast and don't hammer the API), and converts prices to
log-returns -- the quantity everything downstream actually models.

Why log-returns, not prices: predicting price *levels* is the classic
self-deception (a model that says "tomorrow approx today" scores a fake 0.99
R^2). Returns are the (closer to) stationary quantity worth modelling.

Why the cache is gitignored: the raw data is fully reproducible from this
code, so the repo commits the *fetcher*, not gigabytes of CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV for `ticker`, caching to data/raw/<ticker>.csv.

    Returns a DataFrame indexed by date. Requires the `yfinance` package and
    a network connection on first fetch; subsequent calls read the cache.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{ticker.upper()}.csv"

    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so the module loads without it

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}.")
    df.to_csv(cache)
    return df


def to_log_returns(prices: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Convert a price frame to a clean series of daily log-returns."""
    px = prices[price_col].astype(float)
    return np.log(px / px.shift(1)).dropna().rename("log_return")
