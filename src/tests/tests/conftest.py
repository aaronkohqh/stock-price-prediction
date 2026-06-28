import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns():
    """Clean i.i.d.-Normal log-returns."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.02, 2000), name="log_return")


@pytest.fixture
def fat_returns():
    """Fat-tailed log-returns (a heavy-tail block mixed in)."""
    rng = np.random.default_rng(1)
    arr = np.concatenate([rng.normal(0.0005, 0.013, 1800),
                          rng.normal(0.0, 0.07, 200)])
    rng.shuffle(arr)
    return pd.Series(arr, name="log_return")


@pytest.fixture
def garch_returns():
    """Returns from a known GARCH(1,1) process, with the true params."""
    o, a, b = 3e-6, 0.09, 0.89
    rng = np.random.default_rng(0)
    n = 4000
    eps = np.empty(n); s2 = np.empty(n); s2[0] = o / (1 - a - b)
    for t in range(1, n):
        s2[t] = o + a * eps[t - 1] ** 2 + b * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.normal()
    return pd.Series(0.0005 + eps, name="log_return"), (o, a, b)
