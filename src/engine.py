"""
Monte Carlo simulation engine.

This is deliberately 'dumb': it takes ANY fitted ReturnGenerator, asks it for
n_paths x horizon log-returns, reconstructs price paths from a start price,
and summarises the terminal distribution. It contains no model logic, which
is exactly why swapping generators costs nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models.base import ReturnGenerator


@dataclass
class SimulationResult:
    price_paths: np.ndarray      # (n_paths, horizon + 1), includes start price
    start_price: float
    horizon: int

    def terminal_prices(self) -> np.ndarray:
        return self.price_paths[:, -1]

    def percentiles(self, qs=(5, 25, 50, 75, 95)) -> dict[int, float]:
        tp = self.terminal_prices()
        return {q: float(np.percentile(tp, q)) for q in qs}

    def prob_below(self, level: float) -> float:
        """P(terminal price < level). prob_below(start_price) = P(loss)."""
        return float(np.mean(self.terminal_prices() < level))


def run_simulation(
    generator: ReturnGenerator,
    start_price: float,
    horizon: int,
    n_paths: int = 10_000,
    seed: int | None = 42,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    log_rets = generator.generate(horizon=horizon, n_paths=n_paths, rng=rng)

    # Cumulative sum of log-returns -> log price relative to start.
    cum = np.cumsum(log_rets, axis=1)
    prices = start_price * np.exp(cum)
    # Prepend the known start price as column 0.
    price_paths = np.column_stack([np.full(n_paths, start_price), prices])

    return SimulationResult(
        price_paths=price_paths,
        start_price=start_price,
        horizon=horizon,
    )
