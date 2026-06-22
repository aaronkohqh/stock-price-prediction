"""
Feature engineering.

Intentionally thin for v1: pure GBM needs only mu and sigma, no features.
This module becomes relevant from v4/v5, when the volatility-regime model
needs inputs (e.g. realised vol over trailing windows, return autocorrelation,
volume signals). Keep features here so models stay focused on modelling.
"""

from __future__ import annotations

import pandas as pd


def realised_vol(log_returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing realised volatility (rolling std of log-returns)."""
    return log_returns.rolling(window).std().rename(f"realised_vol_{window}")
