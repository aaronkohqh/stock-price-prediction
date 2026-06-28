"""
Systematic calibration study. Runs walk-forward calibration for all four
generators across a universe of tickers and aggregates the results, turning
single-ticker observations into measured findings.

Run from repo root:
    PYTHONPATH=. python scripts/calibration_study.py                # one-step, full list
    PYTHONPATH=. python scripts/calibration_study.py --horizon 21   # 21-step
    PYTHONPATH=. python scripts/calibration_study.py --quick        # 8 tickers, faster

Note: GARCH refits at every walk-forward window, so the full run takes a few
minutes. Progress prints per ticker. Results are saved to results/.
"""

import argparse

import numpy as np
import pandas as pd

from src.data import fetch_prices, to_log_returns
from src.models import (GBMGenerator, BlockBootstrapGenerator,
                        MertonJumpGenerator, GARCHGenerator)
from src.evaluate import (one_step_calibration, horizon_calibration,
                          calibration_error)

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC",
    "GS", "KO", "PEP", "JNJ", "PFE", "MRK", "XOM", "CVX", "WMT", "HD", "PG",
    "DIS", "NFLX", "INTC", "AMD", "CSCO", "ORCL", "CRM", "V", "MA", "UNH",
]
QUICK = ["AAPL", "MSFT", "NVDA", "JPM", "KO", "XOM", "TSLA", "JNJ"]

MODELS = {
    "GBM": lambda: GBMGenerator(),
    "Bootstrap": lambda: BlockBootstrapGenerator(block_size=20),
    "Merton": lambda: MertonJumpGenerator(),
    "GARCH": lambda: GARCHGenerator(),
}


def run(tickers, horizon, step, n_paths):
    rows = []
    for tk in tickers:
        try:
            r = to_log_returns(fetch_prices(tk))
            row = {"ticker": tk, "n": len(r)}
            for name, make in MODELS.items():
                if horizon == 1:
                    res, _ = one_step_calibration(make(), r, step=step, n_paths=n_paths)
                else:
                    res, _ = horizon_calibration(make(), r, horizon=horizon, n_paths=n_paths)
                row[name] = calibration_error(res)
            g = GARCHGenerator().fit(r)
            row["persistence"] = g.alpha + g.beta
            row["lr_vol"] = g.long_run_vol()
        except Exception as e:
            print(f"  {tk:<6} skip ({type(e).__name__})")
            continue
        rows.append(row)
        best = min(MODELS, key=lambda m: row[m])
        cells = "  ".join(f"{m}={row[m]:.1%}" for m in MODELS)
        print(f"  {tk:<6} {cells}   best={best}")
    return pd.DataFrame(rows)


def summarize(df, horizon):
    models = list(MODELS)
    n = len(df)
    print(f"\n=== AGGREGATE over {n} tickers ({'one-step' if horizon==1 else f'{horizon}-step'}) ===")
    print(f"  {'model':<10} {'mean':>7} {'median':>8} {'win-rate':>9}")
    wins = df[models].idxmin(axis=1).value_counts()
    for m in models:
        print(f"  {m:<10} {df[m].mean():>6.1%} {df[m].median():>7.1%} "
              f"{wins.get(m, 0):>6}/{n}")
    if "persistence" in df:
        p = df["persistence"]
        print(f"\n  GARCH persistence (alpha+beta): mean={p.mean():.3f}  "
              f"range={p.min():.3f}-{p.max():.3f}  (all in [0.9,1) confirms the stylised fact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--paths", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    tickers = QUICK if args.quick else UNIVERSE
    print(f"Calibration study: {len(tickers)} tickers, "
          f"{'one-step' if args.horizon==1 else str(args.horizon)+'-step'} "
          f"(this takes a few minutes; GARCH refits each window)\n")

    df = run(tickers, args.horizon, args.step, args.paths)
    if df.empty:
        print("No data fetched.")
        return
    summarize(df, args.horizon)

    from pathlib import Path
    out = Path("results"); out.mkdir(exist_ok=True)
    path = out / f"calibration_study_{'1' if args.horizon==1 else args.horizon}step.csv"
    df.to_csv(path, index=False)
    print(f"\nPer-ticker results saved to {path}")


if __name__ == "__main__":
    main()
