"""
Geometric Brownian Motion return generator (v1).

GBM is the deterministic growth model P_t = P_0 * exp(r t) with a random
shock added -- i.e. the continuous-time SDE  dS = mu*S*dt + sigma*S*dW.
In log-return space this is simply: each period's log-return is drawn
i.i.d. from a Normal(mu - 0.5*sigma^2, sigma^2) distribution.

This is the baseline engine. It is intentionally the simplest thing that
respects the probabilistic framing: it outputs a *distribution* of paths,
not a point forecast.

Honest limitations baked into this model (document these, don't hide them):
  * Returns are assumed i.i.d. Normal -- no fat tails, no volatility
    clustering, no crashes beyond what a thin-tailed Normal allows.
  * Over long horizons the terminal distribution is dominated by `mu`,
    the hardest input to estimate. See the note on drift below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ReturnGenerator


class GBMGenerator(ReturnGenerator):
    def __init__(
        self,
        periods_per_year: int = 252,
        drift_override: float | None = None,
    ) -> None:
        """
        drift_override : annualised drift to use INSTEAD of the historical
            estimate. This exists because estimating drift from a single
            asset's recent history is unreliable, and for a high-flier like
            NVDA, naively compounding historical drift over 5 years produces
            an economically absurd median. Pass a modest market-like value
            (or run a sweep) when you want the sim to characterise *risk*
            rather than assert an *expected return*.
        """
        super().__init__(name="GBM")
        self.periods_per_year = periods_per_year
        self.drift_override = drift_override
        self.mu_period: float | None = None      # per-period drift of log-returns
        self.sigma_period: float | None = None    # per-period vol of log-returns

    def fit(self, log_returns: pd.Series) -> "GBMGenerator":
        r = log_returns.dropna().to_numpy()
        if r.size < 2:
            raise ValueError("Need at least 2 log-returns to fit GBM.")

        self.sigma_period = float(np.std(r, ddof=1))

        if self.drift_override is not None:
            mu_annual = self.drift_override
            self.mu_period = mu_annual / self.periods_per_year
        else:
            # Mean of log-returns already corresponds to (mu - 0.5*sigma^2)
            # in the GBM log formulation, so we use it directly per period.
            self.mu_period = float(np.mean(r))

        self._fitted = True
        return self

    def generate(
        self,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        self._check_fitted()
        # i.i.d. Normal log-returns. Shape: (n_paths, horizon).
        return rng.normal(
            loc=self.mu_period,
            scale=self.sigma_period,
            size=(n_paths, horizon),
        )
