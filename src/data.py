"""
Data layer.

Pulls historical prices from Yahoo Finance via yfinance, caches them to
disk, and converts prices to log-returns -- the quantity everything
downstream actually models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Modern yfinance returns MultiIndex columns (e.g. ('Close','NVDA'))
    even for a single ticker. Collapse to the level holding the OHLCV names
    so downstream code can do df['Close'] and get a Series."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        df = df.copy()
        df.columns = df.columns.get_level_values(0 if "Close" in lvl0 else -1)
    return df


def fetch_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV for `ticker`, caching to data/raw/<ticker>.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{ticker.upper()}.csv"

    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so the module loads without it

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}.")
    df = _flatten_columns(df)
    df.to_csv(cache)
    return df


def to_log_returns(prices: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Convert a price frame to a clean series of daily log-returns."""
    px = prices[price_col]
    # Defensive: if a one-column DataFrame slipped through, squeeze to Series.
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.astype(float)
    return np.log(px / px.shift(1)).dropna().rename("log_return")