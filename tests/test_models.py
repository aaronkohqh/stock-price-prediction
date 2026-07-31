"""Core invariants for the four return generators (the contract + per-model)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from numpy.lib.stride_tricks import sliding_window_view

from src.models import (GBMGenerator, BlockBootstrapGenerator,
                        MertonJumpGenerator, GARCHGenerator, MLVolGenerator)
from src.models.ml_vol import _features, _LOOKBACK

ALL = [GBMGenerator, BlockBootstrapGenerator, MertonJumpGenerator,
       GARCHGenerator, MLVolGenerator]


# ---- contract: every generator behaves the same way ----

@pytest.mark.parametrize("Gen", ALL)
def test_generate_shape_and_finite(Gen, returns):
    out = Gen().fit(returns).generate(50, 100, np.random.default_rng(0))
    assert out.shape == (100, 50)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("Gen", ALL)
def test_reproducible_same_seed(Gen, returns):
    g = Gen().fit(returns)
    a = g.generate(30, 200, np.random.default_rng(5))
    b = g.generate(30, 200, np.random.default_rng(5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("Gen", ALL)
def test_generate_before_fit_raises(Gen):
    with pytest.raises(Exception):
        Gen().generate(10, 10, np.random.default_rng(0))


# ---- GBM ----

def test_gbm_params_match_data(returns):
    g = GBMGenerator().fit(returns)
    assert abs(g.mu_period - returns.mean()) < 1e-9
    assert abs(g.sigma_period - returns.std(ddof=1)) < 1e-9


def test_gbm_drift_override_recenters(returns):
    out = GBMGenerator(drift_override=0.07).fit(returns).generate(2520, 4000, np.random.default_rng(0))
    assert abs(out.mean() * 252 - 0.07) < 0.01


# ---- Block bootstrap ----

def test_bootstrap_preserves_fat_tails(fat_returns):
    out = BlockBootstrapGenerator(20).fit(fat_returns).generate(252, 3000, np.random.default_rng(0))
    assert stats.kurtosis(out.ravel()) > 2.0     # clearly non-Normal


def test_bootstrap_block_too_long_raises(returns):
    with pytest.raises(ValueError):
        BlockBootstrapGenerator(block_size=10000).fit(returns)


def test_bootstrap_drift_override_keeps_shape(fat_returns):
    out = BlockBootstrapGenerator(20, drift_override=0.05).fit(fat_returns).generate(2520, 4000, np.random.default_rng(0))
    assert abs(out.mean() * 252 - 0.05) < 0.02   # drift re-centred
    assert stats.kurtosis(out.ravel()) > 2.0     # tails preserved


# ---- Merton jump-diffusion ----

def test_merton_lambda_is_valid_rate(fat_returns):
    g = MertonJumpGenerator().fit(fat_returns)
    assert 0.0 <= g.lambda_jump <= 1.0


def test_merton_drift_override_recenters(returns):
    out = MertonJumpGenerator(drift_override=0.06).fit(returns).generate(2520, 4000, np.random.default_rng(0))
    assert abs(out.mean() * 252 - 0.06) < 0.02


# ---- GARCH ----

def test_garch_recovers_known_params(garch_returns):
    r, (o, a, b) = garch_returns
    g = GARCHGenerator().fit(r)
    assert abs(g.alpha - a) < 0.05
    assert abs(g.beta - b) < 0.05


def test_garch_stationarity_constraint(garch_returns):
    r, _ = garch_returns
    g = GARCHGenerator().fit(r)
    assert g.alpha + g.beta < 1.0


def test_garch_short_history_raises():
    with pytest.raises(ValueError):
        GARCHGenerator().fit(pd.Series(np.random.default_rng(0).normal(0, 0.02, 50)))


# ---- ML volatility-regime (v5) ----

def test_mlvol_tracks_latent_volatility(garch_returns):
    """The learner must track the true latent volatility it never observes.
    The ML analogue of GARCH recovering known parameters: it tests that the
    model learned the signal, not merely that it ran."""
    r, (o, a, b) = garch_returns
    eps = r.to_numpy() - 0.0005
    s2 = np.empty(eps.size)
    s2[0] = o / (1 - a - b)
    for t in range(1, eps.size):
        s2[t] = o + a * eps[t - 1] ** 2 + b * s2[t - 1]

    g = MLVolGenerator().fit(r)
    e = r.to_numpy() - g.mu
    hist = sliding_window_view(e, _LOOKBACK)
    ts = np.arange(_LOOKBACK, e.size - g.target_window + 1)
    pred = g.model.predict(_features(hist[ts - _LOOKBACK]))

    assert np.corrcoef(pred, np.sqrt(s2)[ts])[0, 1] > 0.6


def test_mlvol_features_have_no_lookahead():
    """Features for a sample must be invariant to anything at or after its
    timestamp -- the leakage check the whole evaluation depends on."""
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 0.02, 500)
    ts = np.arange(_LOOKBACK, eps.size - 21 + 1)
    X = _features(sliding_window_view(eps, _LOOKBACK)[ts - _LOOKBACK])

    corrupted = eps.copy()
    corrupted[300:] = 999.0
    X2 = _features(sliding_window_view(corrupted, _LOOKBACK)[ts - _LOOKBACK])

    past = ts <= 300
    assert np.allclose(X[past], X2[past])


def test_mlvol_drift_override_recenters(returns):
    out = MLVolGenerator(drift_override=0.06).fit(returns).generate(
        2520, 1500, np.random.default_rng(0))
    assert abs(out.mean() * 252 - 0.06) < 0.02


def test_mlvol_short_history_raises():
    with pytest.raises(ValueError):
        MLVolGenerator().fit(pd.Series(np.random.default_rng(0).normal(0, 0.02, 80)))
