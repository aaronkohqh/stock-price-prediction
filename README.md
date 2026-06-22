# Stock Forecasting Engine

A research project investigating what statistical modelling can and cannot
say about the future distribution of a stock's price. Built as a
pre-university project ahead of a CS (AI) degree, with an eye toward
quantitative finance.

**This is a research project, not a trading system.** The methodology and
the honest treatment of uncertainty matter more than any prediction. The
project deliberately predicts a *distribution of outcomes*, never a single
price.

## The question it answers

> Given an asset's historical statistical behaviour, what is the plausible
> *range* of its price over a chosen horizon?

It does **not** answer "what will the company do" -- it has no knowledge of
NVIDIA's competition, margins, or regulation. It models statistical risk and
dispersion, and it labels that scope honestly.

## Conceptual spine

The project follows a deterministic -> probabilistic arc:

  * Deterministic skeleton: `P_t = P_0 * exp(r t)` (compound growth).
  * Add a random shock and you get Geometric Brownian Motion,
    `dS = mu*S*dt + sigma*S*dW` -- the v1 engine.
  * Successive versions relax GBM's unrealistic assumptions (thin tails,
    no volatility clustering, no crashes).

## Architecture

The project is split by **layers with stable interfaces**, not by model.

```
Product surface   input: ticker + horizon  ->  fan chart, percentiles, P(loss)
Simulation engine generator-agnostic Monte Carlo over N paths
Return generator  ONE interface, many implementations (the key abstraction)
Parameters        drift + volatility; ML enters here as a volatility-regime model
Foundation        data layer + evaluation harness (built once, the stable spine)
```

Every modelling approach implements the same `ReturnGenerator` contract:
`generate(history, params) -> return paths`. Adding a capability means adding
a subclass, never rewriting the core.

## Roadmap

| Version | Capability            | What it buys                          |
|---------|-----------------------|---------------------------------------|
| v1      | GBM                   | end-to-end plumbing, baseline engine  |
| v2      | Block bootstrap       | real crash frequency & clustering     |
| v3      | Jump-diffusion        | explicit shocks (Merton)              |
| v4      | GARCH / regime        | volatility clustering                 |
| v5      | ML volatility-regime  | learned conditional volatility        |
| v6+     | Ensemble of generators| combine the above                     |

## Two honest caveats, stated up front

1. **Drift dominates long horizons.** Over 5 years the terminal distribution
   is governed by the assumed drift `mu`, the hardest input to estimate.
   Naively compounding a high-flier's recent returns gives an absurd median.
   The engine supports a drift override and drift sensitivity sweeps.
2. **The 5-year output is not directly testable.** History gives one realised
   5-year path; one sample cannot validate a distribution. We validate the
   engine at short horizons (where data is abundant) via calibration/coverage
   tests, and are explicit about the extrapolation. See `src/evaluate.py`.

## Setup

```bash
conda env create -f environment.yml
conda activate stock-forecast
```

## Status

v1 in progress. Foundation and the GBM generator are scaffolded.
