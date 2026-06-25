"""
Calibration runner. Walk-forward one-step calibration for GBM vs bootstrap on
a real ticker. Run from repo root:
    PYTHONPATH=. python scripts/calibrate.py NVDA
    PYTHONPATH=. python scripts/calibrate.py AAPL --horizon 21
"""

import argparse

from src.data import fetch_prices, to_log_returns
from src.models import GBMGenerator, BlockBootstrapGenerator
from src.evaluate import one_step_calibration, horizon_calibration, calibration_report


def main(ticker="NVDA", horizon=1):
    returns = to_log_returns(fetch_prices(ticker))
    print(f"\n{ticker}: {len(returns)} daily returns, "
          f"excess kurtosis {returns.kurtosis():.1f}\n")

    runner = (lambda g: one_step_calibration(g, returns)) if horizon == 1 \
        else (lambda g: horizon_calibration(g, returns, horizon=horizon))
    tag = "one-step" if horizon == 1 else f"{horizon}-step"

    res_g, n = runner(GBMGenerator())
    print(calibration_report(res_g, n, f"GBM, {tag}"))
    print()
    res_b, n = runner(BlockBootstrapGenerator(block_size=20))
    print(calibration_report(res_b, n, f"Bootstrap, {tag}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("--horizon", type=int, default=1,
                    help="calibration horizon in trading days (default 1)")
    args = ap.parse_args()
    main(args.ticker, args.horizon)
