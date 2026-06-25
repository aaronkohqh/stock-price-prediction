"""GARCH(1,1) return generator (v4). Models volatility as a dynamic process —
today's variance depends on yesterday's shock and yesterday's variance — so
simulated paths show volatility clustering (calm and turbulent stretches)
that constant-sigma models can't produce. Parameters fit by maximum
likelihood (hand-rolled, scipy). See docs/ for the recursion and rationale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import ReturnGenerator


def _garch_variance(eps, omega, alpha, beta, sigma2_0):
    """Run the GARCH(1,1) variance recursion over a residual series."""
    n = eps.size
    sigma2 = np.empty(n)
    sigma2[0] = sigma2_0
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _neg_loglik(params, eps, sigma2_0):
    omega, alpha, beta = params
    sigma2 = np.maximum(_garch_variance(eps, omega, alpha, beta, sigma2_0), 1e-18)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)


class GARCHGenerator(ReturnGenerator):
    def __init__(
        self,
        drift_override: float | None = None,
        periods_per_year: int = 252,
    ) -> None:
        """drift_override: annualised log-drift to use as the mean return,
        instead of the historical mean (keeps drift a consistent user choice)."""
        super().__init__(name="GARCH")
        self.drift_override = drift_override
        self.periods_per_year = periods_per_year
        self.mu = None
        self.omega = self.alpha = self.beta = None
        self.sigma2_last = self.eps_last = None  # state to seed simulation

    def fit(self, log_returns: pd.Series) -> "GARCHGenerator":
        r = log_returns.dropna().to_numpy()
        if r.size < 100:
            raise ValueError("GARCH needs >=100 returns to fit reliably.")

        self.mu = float(np.mean(r))
        eps = r - self.mu

        # Rescale to unit variance so the MLE is well-conditioned (tiny return
        # variances otherwise stall the optimizer). alpha, beta are
        # scale-invariant; omega scales by 1/scale^2, recovered afterwards.
        scale = 1.0 / np.std(eps)
        z = eps * scale
        sigma2_0 = float(np.var(z))   # = 1.0 by construction

        x0 = [0.05, 0.10, 0.85]       # omega_z, alpha, beta on unit-variance data
        bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]
        cons = {"type": "ineq", "fun": lambda p: 1.0 - p[1] - p[2] - 1e-6}
        res = minimize(_neg_loglik, x0, args=(z, sigma2_0), method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        omega_z, self.alpha, self.beta = (float(v) for v in res.x)
        self.omega = omega_z / scale ** 2          # back to original units

        sigma2_full = _garch_variance(z, omega_z, self.alpha, self.beta, sigma2_0)
        self.sigma2_last = float(sigma2_full[-1]) / scale ** 2
        self.eps_last = float(eps[-1])
        self._fitted = True
        return self

    def long_run_vol(self) -> float:
        """Annualised long-run volatility implied by the fit (sanity check)."""
        lr_var = self.omega / max(1e-12, 1.0 - self.alpha - self.beta)
        return float(np.sqrt(lr_var * self.periods_per_year))

    def generate(self, horizon, n_paths, rng):
        self._check_fitted()
        mu = self.mu if self.drift_override is None \
            else self.drift_override / self.periods_per_year

        # Seed every path from the last observed in-sample state.
        sigma2_prev = np.full(n_paths, self.sigma2_last)
        eps_prev = np.full(n_paths, self.eps_last)
        out = np.empty((n_paths, horizon))

        # Path-dependent, but vectorised across paths: one step in time per loop.
        for t in range(horizon):
            sigma2 = self.omega + self.alpha * eps_prev ** 2 + self.beta * sigma2_prev
            eps = np.sqrt(sigma2) * rng.normal(size=n_paths)
            out[:, t] = mu + eps
            eps_prev, sigma2_prev = eps, sigma2

        return out
