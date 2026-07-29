"""Invariants for the evaluation harness -- including a test that validates
the evaluator itself (it must be near-perfect on truly-Normal data)."""

import numpy as np

from src.models import GBMGenerator
from src.evaluate import (coverage, walk_forward_windows, one_step_calibration,
                          calibration_error, naive_zero_return_baseline)


def test_coverage_known_case():
    realised = np.array([0, 1, 2, 3, 4])
    lower = np.zeros(5)
    upper = np.full(5, 2)
    assert abs(coverage(realised, lower, upper) - 0.6) < 1e-9   # 0,1,2 inside


def test_walk_forward_no_leakage():
    windows = list(walk_forward_windows(1000, train=200, test=20, step=20))
    assert len(windows) > 0
    for tr, te in windows:
        assert tr.stop <= te.start      # test strictly after train


def test_baseline_is_positive(returns):
    assert naive_zero_return_baseline(returns) > 0


def test_evaluator_well_calibrated_on_normal(returns):
    # GBM on genuinely i.i.d.-Normal data MUST be well-calibrated; if this
    # fails, the harness (not the model) is wrong.
    res, n = one_step_calibration(GBMGenerator(), returns,
                                  train=400, step=10, n_paths=2000)
    assert calibration_error(res) < 0.05
