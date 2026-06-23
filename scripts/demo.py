"""
v1/v2 end-to-end demo: compare return generators on real data.

Run from the repo root:
    PYTHONPATH=. python scripts/demo.py NVDA 1260
    PYTHONPATH=. python scripts/demo.py NVDA 1260 --drift 0.07

Fetches history, then runs the SAME Monte Carlo engine through each generator
and prints them side by side. (1260 trading days ~= 5 years.) Requires network
on first run to populate the data cache.

The point of the comparison: GBM assumes i.i.d. Normal returns and so throws
away fat tails and volatility clustering; block bootstrap resamples real
historical blocks and keeps them. On a fat-tailed name the bootstrap's
downside percentiles should come out heavier -- that difference is v2.
"""

import argparse

import numpy as np
from scipy import stats

from src.data import fetch_prices, to_log_returns
from src.engine import run_simulation
from src.models import GBMGenerator, BlockBootstrapGenerator


def _row(label, res):
    p = res.percentiles()
    # excess kurtosis of the simulated one-step returns: 0 == Normal/thin-tailed
    daily = np.diff(np.log(res.price_paths), axis=1).ravel()
    ek = float(stats.kurtosis(daily))
    return (f"{label:<16} {ek:>8.2f}  "
            f"${p[5]:>10,.0f} ${p[50]:>11,.0f} ${p[95]:>12,.0f}  "
            f"{100*res.prob_below(res.start_price):>9.1f}%")


def main(ticker="NVDA", horizon=1260, drift=None, n_paths=10_000):
    prices = fetch_prices(ticker)
    returns = to_log_returns(prices)
    start_price = float(prices["Close"].iloc[-1])

    gbm = GBMGenerator(drift_override=drift).fit(returns)
    boot = BlockBootstrapGenerator(block_size=20, drift_override=drift).fit(returns)

    res_gbm = run_simulation(gbm, start_price, horizon, n_paths=n_paths)
    res_boot = run_simulation(boot, start_price, horizon, n_paths=n_paths)

    drift_note = f"drift_override={drift}" if drift is not None else "historical drift"
    print(f"\n{ticker}: start ${start_price:,.2f}, horizon {horizon} days "
          f"(~{horizon/252:.1f}y), {drift_note}")
    print(f"History excess kurtosis: {stats.kurtosis(returns.to_numpy()):.2f} "
          f"(0 = Normal; >0 = fat tails GBM cannot represent)\n")

    print(f"{'generator':<16} {'sim_ek':>8}  "
          f"{'p5':>11} {'p50':>12} {'p95':>13}  {'P(<start)':>10}")
    print("-" * 78)
    print(_row("GBM", res_gbm))
    print(_row("BlockBootstrap", res_boot))
    print("\nIf the bootstrap's p5 is lower and sim_ek is higher, it is "
          "representing\nthe fat tails GBM discards. Both share the same drift "
          "here, so tails are\nthe only difference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("horizon", nargs="?", type=int, default=1260)
    ap.add_argument("--drift", type=float, default=None,
                    help="annualised log-drift override applied to BOTH generators")
    args = ap.parse_args()
    main(args.ticker, args.horizon, args.drift)