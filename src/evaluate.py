"""
Evaluation harness.

Read this docstring before writing any model. It contains the most important
methodological idea in the project.

THE PROBLEM
-----------
The headline product is a 5-year terminal price distribution. You cannot
out-of-sample test that directly: history gives you exactly ONE realised
5-year path per (ticker, start date). One sample cannot validate a
distribution. Any claim that a model "predicts the 5-year outcome well" is
therefore unfalsifiable and should be treated with suspicion -- including
when it's your own model.

THE RESOLUTION
--------------
We do not validate the long-horizon distribution. We validate the *engine
that produces it*, at horizons where data is abundant, and then we are
explicit and humble about extrapolating to 5 years.

Concretely, evaluation operates at three levels:

  1. BASELINE (does any model beat doing nothing?)
     Every generator must be compared against trivial baselines -- e.g.
     "tomorrow's return is zero" or "tomorrow equals today". Most complex
     models fail to beat these on returns. Documenting *that* is a result,
     not a failure of the project.

  2. ONE-STEP CALIBRATION / COVERAGE (the honest core test)
     A generator implies a distribution for the next period's return. Over
     a long backtest, does its 90% interval actually contain the realised
     return ~90% of the time? This is a *calibration* check, and unlike
     point-accuracy it is the right question for a probabilistic model.
     We have thousands of one-step observations, so this is well-powered.

  3. SHORT-HORIZON TERMINAL CALIBRATION (the bridge)
     Repeat the calibration idea at 1-month and 3-month horizons using many
     non-overlapping windows. If the engine is well-calibrated at horizons
     we *can* test, we have a stated, defensible basis for extending it to
     horizons we cannot. The extrapolation assumption is named, not hidden.

What we explicitly DO NOT claim: that good short-horizon calibration proves
5-year accuracy. It does not. Regime change, structural breaks, and the
sheer dominance of the drift assumption all break that extrapolation. The
deliverable characterises *risk and dispersion under stated assumptions*,
not the future.

This module currently provides the baseline + coverage primitives. Fill in
walk-forward orchestration as v1 solidifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_zero_return_baseline(log_returns: pd.Series) -> float:
    """RMSE of always predicting a zero next-period log-return.

    Any model claiming predictive skill on returns must beat this.
    """
    r = log_returns.dropna().to_numpy()
    return float(np.sqrt(np.mean(r ** 2)))


def coverage(
    realised: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of realised values that fall inside [lower, upper].

    For a well-calibrated 90% interval this should be ~0.90. Systematic
    over- or under-coverage tells you the model's uncertainty is wrong --
    which matters far more here than its point error.
    """
    realised, lower, upper = map(np.asarray, (realised, lower, upper))
    inside = (realised >= lower) & (realised <= upper)
    return float(np.mean(inside))


def walk_forward_windows(
    n_obs: int,
    train: int,
    test: int,
    step: int | None = None,
):
    """Yield (train_slice, test_slice) index pairs with NO shuffling and no
    leakage: every test window is strictly after its training window.

    This is the time-series-correct alternative to random k-fold, which would
    leak the future into the past and inflate every score.
    """
    step = step or test
    start = 0
    while start + train + test <= n_obs:
        yield (slice(start, start + train),
               slice(start + train, start + train + test))
        start += step
