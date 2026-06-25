"""Merton jump-diffusion return generator (v3). Diffusion (Normal) plus
Poisson-timed jumps, so the sim can produce shocks beyond the historical
sample -- addressing the 95% tail both GBM and bootstrap under-cover. See
docs/ for the estimation method and its limitations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ReturnGenerator


class MertonJumpGenerator(ReturnGenerator):
    def __init__(
        self,
        jump_threshold: float = 3.0,
        drift_override: float | None = None,
        periods_per_year: int = 252,
    ) -> None:
        """
        jump_threshold : a day whose return is more than this many standard
            deviations from the mean is classified as a jump; the rest are the
            diffusion. This is a transparent threshold heuristic, not full MLE
            -- crude but explainable, and the threshold is an honest parameter.
        drift_override : annualised log-drift to re-centre on, as in the other
            generators (keeps drift a consistent user choice).
        """
        super().__init__(name="MertonJump")
        self.jump_threshold = jump_threshold
        self.drift_override = drift_override
        self.periods_per_year = periods_per_year
        # fitted params (per period)
        self.mu_diff = self.sigma_diff = None
        self.lambda_jump = self.mu_jump = self.sigma_jump = None

    def fit(self, log_returns: pd.Series) -> "MertonJumpGenerator":
        r = log_returns.dropna().to_numpy()
        if r.size < 30:
            raise ValueError("Need >=30 returns to separate jumps from diffusion.")

        mean_all, sd_all = float(np.mean(r)), float(np.std(r, ddof=1))
        is_jump = np.abs(r - mean_all) > self.jump_threshold * sd_all
        diffusion, jumps = r[~is_jump], r[is_jump]

        self.mu_diff = float(np.mean(diffusion))
        self.sigma_diff = float(np.std(diffusion, ddof=1))
        self.lambda_jump = float(jumps.size / r.size)             # jumps per day
        self.mu_jump = float(np.mean(jumps)) if jumps.size else 0.0
        self.sigma_jump = float(np.std(jumps, ddof=1)) if jumps.size > 1 else 0.0

        self._fitted = True
        return self

    def generate(self, horizon, n_paths, rng):
        self._check_fitted()
        shape = (n_paths, horizon)

        # Diffusion component.
        diff = rng.normal(self.mu_diff, self.sigma_diff, size=shape)

        # Jump component: N ~ Poisson(lambda); sum of N Normal(mu_J, sigma_J^2)
        # is Normal(N*mu_J, N*sigma_J^2). scale=0 when N=0 yields exactly 0.
        n_jumps = rng.poisson(self.lambda_jump, size=shape)
        jump = rng.normal(loc=n_jumps * self.mu_jump,
                          scale=np.sqrt(n_jumps) * self.sigma_jump)

        returns = diff + jump

        # Re-centre on the chosen drift, preserving jump structure & diffusion.
        if self.drift_override is not None:
            implied = self.mu_diff + self.lambda_jump * self.mu_jump
            target = self.drift_override / self.periods_per_year
            returns = returns - implied + target

        return returns