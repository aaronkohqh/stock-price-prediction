"""
v1 end-to-end demo.

Run from the repo root:  python scripts/demo.py NVDA 1260

Fetches history, fits GBM, runs a Monte Carlo, prints the terminal
distribution. (1260 trading days ~= 5 years.) Requires network on first run
to populate the data cache.
"""

import sys

from src.data import fetch_prices, to_log_returns
from src.engine import run_simulation
from src.models import GBMGenerator


def main(ticker: str = "NVDA", horizon: int = 1260) -> None:
    prices = fetch_prices(ticker)
    returns = to_log_returns(prices)
    start_price = float(prices["Close"].iloc[-1])

    # NOTE: by default GBM uses historical drift. For a 5-year NVDA sim this
    # is almost certainly too high -- try drift_override=0.07 for a modest
    # market-like assumption, or sweep a range. See README caveat #1.
    gen = GBMGenerator().fit(returns)

    result = run_simulation(gen, start_price=start_price,
                            horizon=horizon, n_paths=10_000)

    print(f"\n{ticker}: start ${start_price:,.2f}, horizon {horizon} days")
    print(f"Fitted per-day drift={gen.mu_period:.5f}, vol={gen.sigma_period:.5f}")
    print("\nTerminal price percentiles:")
    for q, v in result.percentiles().items():
        print(f"  p{q:>2}: ${v:,.2f}")
    print(f"\nP(below start price) = {result.prob_below(start_price):.1%}")
    print("(GBM assumes i.i.d. Normal returns -- no fat tails or crashes.)")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 1260
    main(t, h)
