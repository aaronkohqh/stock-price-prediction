"""
Evaluation harness. Calibration is the honest test of a probabilistic model:
does its X% interval contain the realised value ~X% of the time? Unlike
point accuracy, that is the right question here. See docs/ for the full
rationale (why long horizons can't be tested directly, why we validate the
engine at short horizons and extrapolate explicitly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_zero_return_baseline(log_returns: pd.Series) -> float:
    """RMSE of always predicting a zero next-period log-return."""
    r = log_returns.dropna().to_numpy()
    return float(np.sqrt(np.mean(r ** 2)))


def coverage(realised, lower, upper) -> float:
    """Fraction of realised values inside [lower, upper]. ~0.90 for a good 90% interval."""
    realised, lower, upper = map(np.asarray, (realised, lower, upper))
    return float(np.mean((realised >= lower) & (realised <= upper)))


def walk_forward_windows(n_obs, train, test, step=None):
    """Yield (train_slice, test_slice) pairs, no shuffling, no leakage:
    every test window is strictly after its training window."""
    step = step or test
    start = 0
    while start + train + test <= n_obs:
        yield (slice(start, start + train),
               slice(start + train, start + train + test))
        start += step


def _interval(samples, level):
    lo = np.percentile(samples, 100 * (1 - level) / 2)
    hi = np.percentile(samples, 100 * (1 + level) / 2)
    return lo, hi


def one_step_calibration(
    generator,
    log_returns: pd.Series,
    levels=(0.50, 0.80, 0.90, 0.95),
    train: int = 500,
    step: int = 5,
    n_paths: int = 2000,
    seed: int = 0,
):
    """Walk forward: fit on a rolling window, ask the generator for its
    one-step return distribution, check whether the realised next return
    falls inside each nominal interval. Returns {level: empirical_coverage}
    and the number of test points.
    """
    r = log_returns.dropna()
    rv = r.to_numpy()
    rng = np.random.default_rng(seed)
    hits = {L: 0 for L in levels}
    total = 0
    for tr, te in walk_forward_windows(len(rv), train=train, test=1, step=step):
        generator.fit(r.iloc[tr])
        sim = generator.generate(horizon=1, n_paths=n_paths, rng=rng).ravel()
        realised = rv[te][0]
        total += 1
        for L in levels:
            lo, hi = _interval(sim, L)
            if lo <= realised <= hi:
                hits[L] += 1
    return {L: hits[L] / total for L in levels}, total


def horizon_calibration(
    generator,
    log_returns: pd.Series,
    horizon: int,
    levels=(0.50, 0.80, 0.90, 0.95),
    train: int = 500,
    n_paths: int = 2000,
    seed: int = 0,
):
    """Same idea at a multi-step horizon, using non-overlapping test windows:
    compare the simulated cumulative h-step return distribution against the
    realised cumulative h-step return. Bridges one-step calibration toward
    the (untestable) long horizons.
    """
    r = log_returns.dropna()
    rv = r.to_numpy()
    rng = np.random.default_rng(seed)
    hits = {L: 0 for L in levels}
    total = 0
    for tr, te in walk_forward_windows(len(rv), train=train, test=horizon, step=horizon):
        generator.fit(r.iloc[tr])
        sim = generator.generate(horizon=horizon, n_paths=n_paths, rng=rng)
        cum = sim.sum(axis=1)                 # cumulative h-step return per path
        realised = rv[te].sum()
        total += 1
        for L in levels:
            lo, hi = _interval(cum, L)
            if lo <= realised <= hi:
                hits[L] += 1
    return {L: hits[L] / total for L in levels}, total


def calibration_error(result: dict) -> float:
    """Mean absolute gap between nominal and empirical coverage. 0 = perfect."""
    return float(np.mean([abs(L - emp) for L, emp in result.items()]))


def calibration_report(result: dict, total: int, label: str = "") -> str:
    lines = [f"Calibration{(' — ' + label) if label else ''}  ({total} test points)",
             f"  {'nominal':>8}  {'empirical':>10}  {'gap':>7}"]
    for L, emp in sorted(result.items()):
        lines.append(f"  {L:>8.0%}  {emp:>10.1%}  {emp-L:>+7.1%}")
    lines.append(f"  mean abs calibration error: {calibration_error(result):.1%}")
    lines.append("  (empirical < nominal = overconfident; > nominal = underconfident)")
    return "\n".join(lines)