"""Invariants for the Monte Carlo engine and SimulationResult."""

import numpy as np

from src.models import GBMGenerator
from src.engine import run_simulation


def _result(returns):
    g = GBMGenerator(drift_override=0.05).fit(returns)
    return run_simulation(g, 100.0, 252, n_paths=3000, seed=0)


def test_paths_start_at_start_price(returns):
    res = _result(returns)
    assert res.price_paths.shape == (3000, 253)
    assert np.allclose(res.price_paths[:, 0], 100.0)


def test_terminal_prices_positive(returns):
    assert (_result(returns).terminal_prices() > 0).all()


def test_percentiles_monotonic(returns):
    p = _result(returns).percentiles()
    assert p[5] <= p[25] <= p[50] <= p[75] <= p[95]


def test_cvar_at_least_var(returns):
    res = _result(returns)
    assert res.cvar(0.95) >= res.var(0.95)
    assert res.cvar(0.99) >= res.var(0.99)


def test_max_drawdown_is_a_fraction(returns):
    mdd = _result(returns).max_drawdown((50, 95))
    assert all(0.0 <= v <= 1.0 for v in mdd.values())


def test_prob_below_is_a_probability(returns):
    assert 0.0 <= _result(returns).prob_below(100.0) <= 1.0
