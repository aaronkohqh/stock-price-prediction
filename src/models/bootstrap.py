"""
Block bootstrap return generator (v2).

GBM (v1) draws each day's return independently from Normal. Bakes in
two falsehoods about real markets where returns are NOT Normal because they have fat
tails and crashe, as well as the fact that they are NOT independent.

Block bootstrap fixes both without modelling either. Instead of fitting a
distribution, it resamples *actual contiguous chunks* of historical returns:
draw a random run of `block_size` consecutive real days, lay blocks end to
end until the horizon is filled. Because each block is a real slice of
history, the simulation inherits the true tail magnitudes AND the real
short-range clustering for free -- no distributional parameters estimated.

This is the honest realisation of the original goal ("model the frequency of
market shocks"): the frequency, size, and clustering of past shocks come
straight from the data.

DRIFT CONTROL (de-meaning):
    By default the bootstrap's drift is implicit -- it equals the historical
    mean return of the sampled window, so it silently extrapolates past
    performance. To make drift an explicit, user-controlled assumption (as the
    rest of the product treats it), pass `drift_override`. The generator then
    SHIFTS each resampled return by (target_mean - historical_mean), which
    re-centres the whole distribution on the chosen drift while leaving its
    variance, skew, kurtosis and within-block clustering untouched. You keep
    the realistic *shape* of history and replace only its *trend*.

HONEST LIMITATIONS (document, don't hide):
  * It can only resample crashes that ALREADY occurred in the history window.
    No 2008 in your data -> no 2008 in your sim. It captures *realised* risk,
    not unprecedented risk, and the next crisis won't resemble the last.
  * De-meaning shifts the marginal distribution but does not change the tail
    SHAPE, which is the point -- but it also cannot invent downside that the
    history never contained.
  * Block length is a real tradeoff (see below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ReturnGenerator


class BlockBootstrapGenerator(ReturnGenerator):
    def __init__(
        self,
        block_size: int = 20,
        drift_override: float | None = None,
        periods_per_year: int = 252,
    ) -> None:
        """
        block_size : length of each resampled run of consecutive returns.
            Too short (->1) destroys the clustering this model exists to
            capture and collapses toward i.i.d. GBM. Too long (-> close to the
            history length) leaves too few distinct blocks, so paths just
            replay history. ~20 (a trading month) is a sensible default.
        drift_override : annualised log-drift to re-centre on, INSTEAD of the
            historical mean. None keeps the historical (implicit) drift. This
            mirrors GBMGenerator so drift is a consistent user choice across
            generators.
        periods_per_year : for converting the annualised override to per-period.
        """
        super().__init__(name="BlockBootstrap")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        self.block_size = block_size
        self.drift_override = drift_override
        self.periods_per_year = periods_per_year
        self._returns: np.ndarray | None = None
        self.mu_hist_period: float | None = None  # historical per-period mean

    def fit(self, log_returns: pd.Series) -> "BlockBootstrapGenerator":
        r = log_returns.dropna().to_numpy()
        if r.size < self.block_size:
            raise ValueError(
                f"Need at least block_size ({self.block_size}) returns to fit; "
                f"got {r.size}. Use a shorter block or more history."
            )
        self._returns = r
        self.mu_hist_period = float(np.mean(r))
        self._fitted = True
        return self

    def generate(
        self,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        self._check_fitted()
        assert self._returns is not None
        pool = self._returns
        n = pool.size
        b = self.block_size

        max_start = n - b                       # inclusive last valid block start
        n_blocks = int(np.ceil(horizon / b))

        starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks))
        offsets = np.arange(b)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
        idx = idx[:, :horizon]

        returns = pool[idx]                     # (n_paths, horizon), real blocks

        # De-mean and re-centre on the chosen drift, preserving shape.
        if self.drift_override is not None:
            mu_target_period = self.drift_override / self.periods_per_year
            returns = returns - self.mu_hist_period + mu_target_period

        return returns