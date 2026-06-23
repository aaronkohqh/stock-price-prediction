"""
v1/v2 end-to-end demo. Run from repo root:
    PYTHONPATH=. python scripts/demo.py NVDA 1260 --drift 0.07

Fetches history, runs the same Monte Carlo engine through GBM and block
bootstrap, prints a comparison + tail metrics, and saves a fan chart.
Rationale lives in docs/.
"""

import argparse

import numpy as np
from scipy import stats

from src.data import fetch_prices, to_log_returns
from src.engine import run_simulation
from src.models import GBMGenerator, BlockBootstrapGenerator
from src.viz import fan_chart


def _row(label, res):
    p = res.percentiles()
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

    # Tail risk (where the bootstrap's value actually shows) -- bootstrap result.
    print(f"\nTail risk (BlockBootstrap):")
    print(f"  VaR  95%: {100*res_boot.var(0.95):5.1f}%   99%: {100*res_boot.var(0.99):5.1f}%")
    print(f"  CVaR 95%: {100*res_boot.cvar(0.95):5.1f}%   99%: {100*res_boot.cvar(0.99):5.1f}%")
    mdd = res_boot.max_drawdown((50, 95))
    print(f"  Max drawdown  median: {100*mdd[50]:.1f}%   p95: {100*mdd[95]:.1f}%")

    path = fan_chart(res_boot, title=f"{ticker}: BlockBootstrap, ~{horizon/252:.1f}y")
    print(f"\nFan chart saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("horizon", nargs="?", type=int, default=1260)
    ap.add_argument("--drift", type=float, default=None,
                    help="annualised log-drift override applied to BOTH generators")
    args = ap.parse_args()
    main(args.ticker, args.horizon, args.drift)