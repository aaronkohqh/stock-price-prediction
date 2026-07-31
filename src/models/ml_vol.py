"""ML volatility-regime generator (v5). A gradient-boosted regressor learns the
map from backward-looking return features to forward realised volatility; the
generator then draws returns from that predicted volatility, exactly as GARCH
does with its three-parameter recursion.

The ML predicts VOLATILITY only -- never returns, direction, or events.
Volatility is chosen because it is persistent, abundantly sampled and
measurable; the features include signed terms so the learner can express the
leverage effect (negative returns predicting higher future vol), which a
symmetric GARCH in r^2 structurally cannot. See docs/ for the rationale and
the honest comparison against v4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import GradientBoostingRegressor

from .base import ReturnGenerator

_WINDOWS = (5, 10, 21, 63)      # realised-vol lookback windows
_LOOKBACK = 63                  # longest window the features need
_EWMA_LAMBDA = 0.94             # RiskMetrics decay

FEATURE_NAMES = (
    "rv_5", "rv_10", "rv_21", "rv_63",
    "ewma_vol", "abs_last",
    "downside_21", "frac_down_21", "absret_dispersion_21",
)


def _features(buf: np.ndarray) -> np.ndarray:
    """Features from a buffer of recent de-meaned returns.

    buf : (n, _LOOKBACK), most recent observation LAST.
    Returns (n, len(FEATURE_NAMES)). Strictly backward-looking -- every column
    is a function of the buffer only, so no future information can enter.
    """
    feats = [buf[:, -w:].std(axis=1) for w in _WINDOWS]

    w = _EWMA_LAMBDA ** np.arange(_LOOKBACK)[::-1]
    w = w / w.sum()
    feats.append(np.sqrt((buf ** 2 * w).sum(axis=1)))

    feats.append(np.abs(buf[:, -1]))

    last21 = buf[:, -21:]
    feats.append(np.minimum(last21, 0.0).mean(axis=1))   # downside pressure (<= 0)
    feats.append((last21 < 0).mean(axis=1))              # fraction of down days
    feats.append(np.abs(last21).std(axis=1))             # dispersion of |r|

    return np.column_stack(feats)


class MLVolGenerator(ReturnGenerator):
    def __init__(
        self,
        target_window: int = 21,
        drift_override: float | None = None,
        periods_per_year: int = 252,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 0,
    ) -> None:
        """
        target_window : forward horizon (days) over which realised volatility is
            measured as the learning target. Chosen empirically, and the choice
            matters more than any other hyperparameter: on synthetic data,
            one-step calibration error fell monotonically from 20.4% (W=2) to
            10.4% (W=3), 5.4% (W=5), 2.1% (W=10) and 1.2% (W=21). A single
            day's volatility is too noisy a target to learn; a smoother forward
            estimate of the same latent quantity is far more learnable.
        """
        super().__init__(name="MLVol")
        self.target_window = target_window
        self.drift_override = drift_override
        self.periods_per_year = periods_per_year
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        )
        self.mu = None
        self._buf_seed = None
        self._sigma_floor = self._sigma_cap = None

    # ---- fitting ----

    def fit(self, log_returns: pd.Series) -> "MLVolGenerator":
        r = log_returns.dropna().to_numpy()
        need = _LOOKBACK + self.target_window + 50
        if r.size < need:
            raise ValueError(f"MLVol needs >={need} returns to fit.")

        self.mu = float(np.mean(r))
        eps = r - self.mu                      # vol modelling is drift-free
        n = eps.size

        # Sample t: features from eps[t-_LOOKBACK:t], target from eps[t:t+W].
        # The two windows do not overlap, so the target cannot leak into X.
        hist = sliding_window_view(eps, _LOOKBACK)          # row i = eps[i:i+LB]
        fwd = sliding_window_view(eps, self.target_window)  # row j = eps[j:j+W]
        ts = np.arange(_LOOKBACK, n - self.target_window + 1)

        X = _features(hist[ts - _LOOKBACK])
        y = fwd[ts].std(axis=1)

        self.model.fit(X, y)

        # Clip predictions into the range the data actually supports, so an
        # extrapolating tree cannot emit a degenerate or absurd volatility.
        self._sigma_floor = float(max(y.min() * 0.5, 1e-6))
        self._sigma_cap = float(y.max() * 3.0)
        self._buf_seed = eps[-_LOOKBACK:].copy()

        self._fitted = True
        return self

    def feature_importance(self) -> pd.Series:
        """Which features the learner actually leans on (transparency)."""
        self._check_fitted()
        return pd.Series(self.model.feature_importances_,
                         index=FEATURE_NAMES).sort_values(ascending=False)

    # ---- generation ----

    def generate(self, horizon, n_paths, rng):
        self._check_fitted()
        mu = (self.mu if self.drift_override is None
              else self.drift_override / self.periods_per_year)

        # Every path carries its own rolling buffer, so the predicted volatility
        # is conditional on that path's own simulated history.
        buf = np.tile(self._buf_seed, (n_paths, 1))
        out = np.empty((n_paths, horizon))

        for t in range(horizon):
            sigma = np.clip(self.model.predict(_features(buf)),
                            self._sigma_floor, self._sigma_cap)
            eps = sigma * rng.normal(size=n_paths)
            out[:, t] = mu + eps
            buf = np.column_stack([buf[:, 1:], eps])

        return out