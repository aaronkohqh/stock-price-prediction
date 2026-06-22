"""
Return generator contract.

This is the single most important file in the project. Every modelling
approach -- GBM, historical bootstrap, jump-diffusion, GARCH, an ML-driven
one -- implements this one interface. The simulation engine and the
evaluation harness never know which concrete generator they are holding.

That decoupling is what lets new capabilities be added as small, isolated
changes (a new subclass) instead of rewrites of the core.

A generator's job is narrow and well-defined: given the asset's return
history and some parameters, produce simulated *return* paths. It does NOT
deal in price levels (price is reconstructed downstream from a known start
price) and it does NOT decide horizons, path counts, or output formatting --
those belong to the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ReturnGenerator(ABC):
    """Abstract base class for all return-path generators.

    Concrete subclasses implement `fit` (estimate parameters from history)
    and `generate` (produce simulated return paths). Keeping these two steps
    separate makes it possible to fit once and simulate many times, and to
    inspect the fitted parameters for sanity-checking -- which matters a lot
    for drift, the parameter most likely to silently break long horizons.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self._fitted = False

    @abstractmethod
    def fit(self, log_returns: pd.Series) -> "ReturnGenerator":
        """Estimate parameters from a series of historical log-returns.

        Must set self._fitted = True and return self.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Produce simulated log-return paths.

        Parameters
        ----------
        horizon : number of periods (e.g. trading days) to simulate forward.
        n_paths : number of independent paths (the N of the Monte Carlo).
        rng     : a NumPy Generator, passed in so runs are reproducible and
                  the generator never owns global random state.

        Returns
        -------
        ndarray of shape (n_paths, horizon) of simulated log-returns.
        """
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}: call fit() before generate()."
            )
