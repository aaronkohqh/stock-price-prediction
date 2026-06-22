# CLAUDE.md

Project context and working principles. Read this fully before proposing any
code. It encodes deliberate decisions — deviating from them is almost always a
mistake, not an improvement.

## What this project is

A research project investigating what statistical modelling can and cannot say
about the future *distribution* of a stock's price. The two core user inputs
are **stock choice** and **time horizon**. The output is always a
**distribution of outcomes** (fan chart, percentile bands, probability of
loss) — never a single predicted price.

It is a **research project, not a trading system**. Methodology and the honest
treatment of uncertainty matter more than any prediction.

## Non-negotiable principles

1. **Distributional, not oracle.** Never produce or imply a point forecast of a
   future price as if it were "the answer". A 5-year price is among the most
   stochastic objects in finance; a single number is a category error. Always
   work in distributions and intervals.

2. **Honesty over apparent accuracy.** The quality metric for this project is
   *calibration* (do the intervals cover reality at the stated rate?), not
   point accuracy. Do not chase R^2 on price levels — predicting price levels
   yields a fake ~0.99 R^2 because "tomorrow ≈ today". Model **log-returns**,
   not prices.

3. **Do NOT use an LLM to predict prices.** Claude (this agent) helps *write*
   the project. It must never be wired in as the predictive model. LLMs have no
   numerical edge on returns and would undermine the project's credibility. The
   math does the forecasting; language models only explain/interpret, if at all.

4. **Drift is the silent killer.** Over long horizons the terminal distribution
   is dominated by the assumed drift mu, the hardest input to estimate. Never
   naively compound a high-flier's historical drift over years — it produces
   absurd medians. Prefer a modest market-like drift or an explicit sensitivity
   sweep, and say which.

5. **Scope honesty.** This models *statistical* risk only. It knows nothing
   about a company's fundamentals (competition, margins, regulation). Label the
   question precisely: "given the asset's statistical behaviour, what is the
   plausible range?" — not "what will the company do".

## Architecture (do not break these seams)

The project is split by **layers with stable interfaces**, not by model.

- **Foundation** (`src/data.py`, `src/evaluate.py`): built once, changed
  rarely. Data fetching + the evaluation harness. The harness is a first-class
  citizen, NOT something buried inside models.
- **Return generator contract** (`src/models/base.py`): the heart of the
  project. Every model — GBM, bootstrap, jump-diffusion, GARCH, ML — subclasses
  `ReturnGenerator` and implements `fit()` + `generate()`. Same interface,
  always.
- **Simulation engine** (`src/engine.py`): deliberately "dumb". It runs N paths
  through whatever generator it's handed and never contains model logic. Do not
  put model-specific behaviour here.
- **Parameters**: drift + volatility estimation. This is the ONLY place ML
  belongs — and only to predict the *volatility regime* (calm vs turbulent),
  never to predict events or prices.
- **Product surface**: input (ticker + horizon) -> fan chart, percentiles,
  P(loss).

**The golden rule:** add a capability by adding a new `ReturnGenerator`
subclass, never by rewriting the engine or the foundation.

## Evaluation discipline

- You **cannot** out-of-sample test a single long-horizon path (history gives
  one realised path). Do not claim long-horizon accuracy.
- Validate the engine at SHORT horizons where data is abundant: one-step
  calibration / coverage tests (does the 90% interval cover ~90% of realised
  returns?).
- Every model must be compared against trivial baselines (e.g. zero next-period
  return). Most complex models fail to beat them — documenting that is a result,
  not a failure.
- Time-series splits only. **Never** shuffle or use random k-fold — it leaks
  the future into the past and inflates every score.

## Conventions

- Python 3.12, conda env in `environment.yml` (env name `stock-forecast`).
- Generators take an injected `numpy` `Generator` (`rng`) for reproducibility;
  never use global random state.
- Keep new models in `src/models/`, register them in `src/models/__init__.py`.
- Cached price data lives in `data/raw/` and is gitignored — commit the fetcher,
  not the CSVs. Never commit data.
- If adding an API key for any reason, it goes in `.env` (gitignored), never in
  committed code. This is a public repo.

## Roadmap (each version = one clean addition behind the contract)

- v1 — GBM (done): end-to-end plumbing, baseline engine.
- v2 — Block bootstrap: real crash frequency & clustering, for free.
- v3 — Jump-diffusion (Merton): explicit shocks.
- v4 — GARCH / regime-switching: volatility clustering.
- v5 — ML volatility-regime: learned conditional volatility (the defensible
  form of "ML predicting market behaviour").
- v6+ — Ensemble of generators.

## How to work here

- Propose changes as reviewable diffs; explain the *why*, not just the *what*.
- Prefer the smallest change that respects the architecture over a clever
  rewrite.
- When something can't be done honestly (e.g. "make it predict the real price
  accurately"), say so plainly and offer the honest alternative, rather than
  quietly building the thing that looks impressive but isn't sound.
