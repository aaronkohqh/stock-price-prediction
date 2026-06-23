"""
Monte Carlo simulation engine. Generator-agnostic: runs any fitted
ReturnGenerator's paths and summarises the result. No model logic here.
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

    def returns(self) -> np.ndarray:
        """Terminal return per path, relative to start (e.g. -0.2 = -20%)."""
        return self.terminal_prices() / self.start_price - 1.0

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk over the horizon, as a positive fractional loss.

        var(0.95) is the loss not exceeded with 95% probability. Returns 0 if
        the relevant quantile is a gain (no loss at that confidence).
        """
        q = (1.0 - confidence) * 100.0
        worst_return = np.percentile(self.returns(), q)
        return float(max(0.0, -worst_return))

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (expected shortfall): mean loss in the worst tail,
        as a positive fractional loss. Always >= var at the same confidence.
        """
        r = self.returns()
        q = (1.0 - confidence) * 100.0
        threshold = np.percentile(r, q)
        tail = r[r <= threshold]
        if tail.size == 0:
            return 0.0
        return float(max(0.0, -tail.mean()))

    def max_drawdown(self, qs=(50, 95)) -> dict[int, float]:
        """Distribution of per-path max drawdown (positive fractional decline).

        Max drawdown is the worst peak-to-trough drop along a path. Returns
        percentiles of that distribution, e.g. {95: 0.41} = a 41% drawdown at
        the 95th percentile of paths.
        """
        running_max = np.maximum.accumulate(self.price_paths, axis=1)
        dd = (self.price_paths - running_max) / running_max  # <= 0
        per_path_mdd = -dd.min(axis=1)                        # positive decline
        return {q: float(np.percentile(per_path_mdd, q)) for q in qs}

    def percentile_paths(self, qs=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
        """Percentile of price across paths at each time step (for fan charts)."""
        return {q: np.percentile(self.price_paths, q, axis=0) for q in qs}


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