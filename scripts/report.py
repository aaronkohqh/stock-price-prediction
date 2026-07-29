"""
Unified report: one ticker, one generator, everything the engine knows.
Assembly over the existing layers -- no new modelling logic lives here.

Run from repo root:
    PYTHONPATH=. python scripts/report.py NVDA 252 --model garch
    PYTHONPATH=. python scripts/report.py AAPL 1260 --model bootstrap --drift 0.07
    PYTHONPATH=. python scripts/report.py MSFT 252 --model gbm --skip-calib
"""

import argparse

from src.data import fetch_prices, to_log_returns
from src.engine import run_simulation
from src.evaluate import calibration_report, one_step_calibration
from src.models import (BlockBootstrapGenerator, GARCHGenerator,
                        GBMGenerator, MertonJumpGenerator)
from src.viz import fan_chart

MODELS = {
    "gbm": lambda d: GBMGenerator(drift_override=d),
    "bootstrap": lambda d: BlockBootstrapGenerator(block_size=20, drift_override=d),
    "merton": lambda d: MertonJumpGenerator(drift_override=d),
    "garch": lambda d: GARCHGenerator(drift_override=d),
}


def main(ticker, horizon, model, drift, n_paths, skip_calib):
    prices = fetch_prices(ticker)
    returns = to_log_returns(prices)
    start = float(prices["Close"].iloc[-1])

    gen = MODELS[model](drift).fit(returns)
    res = run_simulation(gen, start, horizon, n_paths=n_paths)

    drift_note = f"drift={drift:.0%}" if drift is not None else "historical drift"
    print(f"\n{ticker} — {gen.name}, {horizon}d (~{horizon/252:.1f}y), {drift_note}")
    print(f"start ${start:,.2f}   |   {len(returns)} days of history\n")

    p = res.percentiles()
    print("Distribution")
    print(f"  p5  ${p[5]:>10,.0f}    p50 ${p[50]:>10,.0f}    p95 ${p[95]:>10,.0f}")
    print(f"  P(below start): {100*res.prob_below(start):.1f}%\n")

    mdd = res.max_drawdown((50, 95))
    print("Tail risk")
    print(f"  VaR   95% {100*res.var(0.95):5.1f}%     99% {100*res.var(0.99):5.1f}%")
    print(f"  CVaR  95% {100*res.cvar(0.95):5.1f}%     99% {100*res.cvar(0.99):5.1f}%")
    print(f"  Max drawdown   median {100*mdd[50]:.1f}%     p95 {100*mdd[95]:.1f}%\n")

    if not skip_calib:
        cal, n = one_step_calibration(MODELS[model](None), returns)
        print(calibration_report(cal, n, f"{gen.name}, one-step"))
        print()

    path = fan_chart(res, title=f"{ticker} — {gen.name}, ~{horizon/252:.1f}y",
                     outpath=f"results/report_{ticker}_{model}.png")
    print(f"Fan chart: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("horizon", nargs="?", type=int, default=252)
    ap.add_argument("--model", choices=list(MODELS), default="garch")
    ap.add_argument("--drift", type=float, default=None,
                    help="annualised log-drift override")
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--skip-calib", action="store_true",
                    help="skip walk-forward calibration (it is the slow part)")
    a = ap.parse_args()
    main(a.ticker, a.horizon, a.model, a.drift, a.paths, a.skip_calib)