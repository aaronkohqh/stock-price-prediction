# Stock Forecasting Engine

A research project investigating what statistical modelling can and cannot say
about the future *distribution* of a stock's price. Built as a pre-university
project ahead of a CS (AI) degree, with an eye toward quantitative finance.

**This is a research project, not a trading system.** The methodology and the
honest treatment of uncertainty matter more than any prediction. The project
deliberately outputs a *distribution of outcomes*, never a single price.

## The question it answers

> Given an asset's historical statistical behaviour, what is the plausible
> *range* of its price over a chosen horizon?

It does **not** answer "what will the company do" — it has no knowledge of a
company's competition, margins, or regulation. It models statistical risk and
dispersion, and labels that scope honestly. A point forecast of a future price
is treated as a category error: a 5-year price is among the most stochastic
objects in finance, and any tool claiming to predict it accurately is either
lucky or lying (a public, accurate predictor would destroy its own signal).



## Conceptual spine

The project follows a deterministic → probabilistic arc:

- Deterministic skeleton: Pₜ = P₀ · exp(rt) (compound growth).
- Add a random shock and you get Geometric Brownian Motion,
  dS = μS dt + σS dW — the v1 engine.
- Successive versions relax GBM's unrealistic assumptions (thin tails, no
  volatility clustering, no crashes).

In the fan chart, the **median line is the deterministic model** and the
**shaded bands are the probabilistic model** — uncertainty made visible.

## Mathematical background

The math the engine rests on, in the order it is used.

**Log-returns** — the (more stationary) quantity everything models:

&nbsp;&nbsp;&nbsp;&nbsp;rₜ = ln(Pₜ / Pₜ₋₁)

**Geometric Brownian Motion (v1)** as a stochastic differential equation:

&nbsp;&nbsp;&nbsp;&nbsp;dS = μS dt + σS dW

with W a Wiener process. In log-return space each step is drawn from a Normal:

&nbsp;&nbsp;&nbsp;&nbsp;rₜ ~ 𝒩( (μ − σ²/2) dt , σ² dt )

The −σ²/2 term is volatility drag — variance erodes arithmetic return over time.

**Price reconstruction** — exponentiate the cumulative sum of log-returns:

&nbsp;&nbsp;&nbsp;&nbsp;Pₜ = P₀ · exp( Σ rᵢ )&nbsp;&nbsp;(i = 1 … t)

**Annualisation** — drift scales with time, volatility with its square root:

&nbsp;&nbsp;&nbsp;&nbsp;μₐ = μ · 252&nbsp;&nbsp;&nbsp;&nbsp;σₐ = σ · √252

**Monte Carlo estimate** — the bands are quantiles counted across N simulated
paths, not a closed form. The estimate is a sample mean; by the Law of Large
Numbers its error shrinks as:

&nbsp;&nbsp;&nbsp;&nbsp;standard error ∝ 1 / √N

**Widening cone** — variance of a sum of t returns grows with t, so dispersion
grows with √t:

&nbsp;&nbsp;&nbsp;&nbsp;Var( Σ rᵢ ) = t · σ²&nbsp;&nbsp;⟹&nbsp;&nbsp;spread ∝ σ√t

**Lognormal terminal distribution** — summed log-returns, then exponentiated,
give a right-skewed price bounded below by 0 (the fan chart's up/down asymmetry):

&nbsp;&nbsp;&nbsp;&nbsp;ln(Pₜ / P₀) ~ Normal&nbsp;&nbsp;⟹&nbsp;&nbsp;Pₜ ~ Lognormal

**Block bootstrap de-meaning (v2 drift control)** — shift resampled returns to
re-centre on a chosen drift while preserving variance, skew, and kurtosis:

&nbsp;&nbsp;&nbsp;&nbsp;rᵢ′ = rᵢ − μ_hist + μ_target

**Fat tails (excess kurtosis)** — departure from Normal; > 0 = heavier tails
than GBM can represent:

&nbsp;&nbsp;&nbsp;&nbsp;κ = E[(r − μ)⁴] / σ⁴ − 3

**Risk metrics** — reported because the model-choice difference lives in the
tail, not the centre. For terminal return R at confidence c:

&nbsp;&nbsp;&nbsp;&nbsp;VaR = −Q₁₋c(R)&nbsp;&nbsp;(loss not exceeded with probability c)

&nbsp;&nbsp;&nbsp;&nbsp;CVaR = −E[ R | R ≤ Q₁₋c(R) ]&nbsp;&nbsp;(mean loss in that tail; CVaR ≥ VaR)

&nbsp;&nbsp;&nbsp;&nbsp;MDD = maxₜ (peakₜ − Pₜ) / peakₜ&nbsp;&nbsp;(worst peak-to-trough drop)

**Calibration (evaluation)** — the honest test: a c-level interval should
contain the realised value about a fraction c of the time:

&nbsp;&nbsp;&nbsp;&nbsp;coverage = (# realised inside interval) / total ≈ c

## Architecture

Split by **layers with stable interfaces**, not by model.

```
Product surface   input: ticker + horizon + drift mode  ->  fan chart, percentiles, P(loss), VaR/CVaR
Advisory layer    rules-based credibility flags on the output (not ML)
Simulation engine generator-agnostic Monte Carlo over N paths
Return generator  ONE interface, many implementations (the key abstraction)
Foundation        data layer + evaluation harness (built once, the stable spine)
```

Every model implements the same `ReturnGenerator` contract:
`generate(history, params) → return paths`. Drift and volatility are parameters
passed *into* a generator; the engine and evaluation never know which generator
is plugged in. Adding a capability means adding a subclass, never rewriting the
core — proven when v2 dropped in with zero engine changes.

## Where machine learning is — and isn't — used

The most important AI decision in the project is *where not to apply ML*.

- **Used (v5):** conditional volatility. Volatility clustering is a real,
  abundantly-sampled, measurable pattern — legitimate ML doing a job it is good
  at. This is the AI centerpiece, implemented as one more generator.
- **Not used for predicting events** (crashes, wars): exogenous, rare, and
  structurally novel — no training set exists.
- **Not used for drift:** long-run return is not learnable from one asset's
  price history; an explicit assumption + sensitivity sweep is the honest tool.
- **Not used for the advisory layer:** transparent rules beat a black-box
  credibility score in a project whose whole point is auditability, and there is
  no labelled "credible vs not" dataset to train on.

Using ML where a learnable signal and measurable target exist, and refusing it
where they don't, is treated as a contribution — not an omission.

## Roadmap

| Version | Capability | Status |
|---------|-----------|--------|
| v1 | GBM | **done** — baseline engine |
| v2 | Block bootstrap (+ drift control) | **done** — fat tails & clustering |
| — | Fan chart + VaR/CVaR/max-drawdown | **done** |
| — | Advisory layer (rules-based) | scaffolded |
| — | Evaluation harness (walk-forward/coverage) | scaffolded |
| v3 | Jump-diffusion (Merton) | **done** — explicit shocks beyond the sample |
| v4 | GARCH / regime | planned — volatility clustering |
| v5 | ML volatility-regime | planned — learned conditional volatility |
| v6+ | Ensemble of generators | planned |
| — | Streamlit app | planned — built last, over a stable backend |

## Findings (so far)

Both results come from running the engine and calibration harness, not from assumption.

1. The model difference is horizon- and shape-dependent. GBM vs bootstrap barely differ in the central percentiles at long horizons (CLT smooths the daily fat tails away), but diverge at shorter horizons and in the extreme tail (p1, worst day, max drawdown) — which is why those tail metrics are reported, not just p5–p95.

2. Which model is better calibrated also flips with horizon. Mean absolute coverage error across {50, 80, 90, 95}% intervals:

Ticker	Horizon	GBM	Bootstrap
NVDA	one-step	4.5%	1.1%
MSFT	one-step	4.7%	0.7%
AAPL	21-step	2.0%	5.1%
MSFT	21-step	4.2%	5.1%
One-step: bootstrap wins — daily returns are fat-tailed (kurtosis ≈ 6–8), GBM's Normal can't represent that, so it's underconfident in the centre.
21-step: GBM wins — summing returns pulls the distribution toward Normal (CLT), while the bootstrap's 20-day blocks over-disperse.
No model dominates across horizons; the right tool depends on the timescale.

Validation/caveats: the harness is near-perfect on synthetic i.i.d.-Normal data (≈1.8% error, as GBM should be), so the results are trusted. The 21-step tests use ~173 windows (noisier than ~728 one-step); the bootstrap's over-dispersion is partly a block_size artifact. Both models under-cover the 95% tail at one step — motivating v3 (jump-diffusion).

3. The deepest tail resists all three models. Merton adds explicit
Poisson jumps to reach shocks beyond the historical sample, and on synthetic
fat-tailed data it calibrates the 95% tail almost perfectly. But on real NVDA,
all three models are still under-cover the 95% interval (GBM −2.3%, bootstrap
−2.3%, Merton −3.5%), and tuning Merton's jump threshold improves the centre
without fixing the deep tail. The persistent under-coverage is structural where
real extreme days are fatter than a Normal-diffusion-plus-Normal-jump family
can represent — not a tuning failure. Naming that limit honestly matters more
than hiding it behind a tuned number.

The takeaway across all three: no single model dominates. Bootstrap is the
generalist (best one-step calibration), GBM is the long-horizon CLT play,
Merton fixes GBM's centre via explicit jumps — each earns its place for a
different question, and the calibration harness is what lets you say which.

4. GARCH is the consistent all-rounder across horizons. One-step
calibration (mean abs coverage error) is consistent across tickers — NVDA and
MSFT both rank: bootstrap (1.1% / 0.7%) < Merton (2.8% / 2.4%) < GARCH
(3.0% / 2.9%) < GBM (4.5% / 4.7%). The bootstrap wins one-step, but it
over-disperses at 21-step (5.1%), where GARCH is best (1.1%) — its volatility
mean-reverts over the horizon instead of compounding block noise. So GARCH
rarely wins a single horizon outright but is never badly wrong, because it is
the only model conditioning on the current volatility regime.

5. Volatility persistence is near-universal and independent of volatility
level. Fitting GARCH across eight names, persistence (α+β) sits in 0.91–0.99
regardless of how calm or wild the stock is — a known stylised fact, recovered
independently. Persistence and volatility level are separate axes: KO and JNJ
share ~17% long-run vol but differ in persistence; TSLA is both the most
volatile (~57%) and the most persistent (0.991, near a unit root).

## Two caveats, stated up front

1. **Drift dominates long horizons.** Over 5 years the terminal distribution is
   governed by the assumed drift μ, the hardest input to estimate. Naively
   compounding a high-flier's recent returns gives an absurd median. The engine
   supports a drift override and drift sensitivity sweeps across all generators.
2. **The long-horizon output is not directly testable.** History gives one
   realised multi-year path; one sample cannot validate a distribution. The
   engine is validated at short horizons (abundant data) via calibration/coverage
   tests, with the extrapolation stated explicitly. See `src/evaluate.py`.

## Setup

```bash
conda env create -f environment.yml
conda activate stock-forecast
```

## Usage

```bash
# compare GBM vs bootstrap, print tail metrics, save a fan chart
PYTHONPATH=. python scripts/demo.py NVDA 1260 --drift 0.07

# short vs long horizon (watch the bootstrap-vs-GBM tail gap change)
PYTHONPATH=. python scripts/demo.py TSLA 504 --drift 0.05
PYTHONPATH=. python scripts/demo.py MSFT 252 --drift 0.05
```

`ticker` and `horizon` (in trading days; 252 ≈ 1 year) are positional; `--drift`
sets an annualised log-drift applied to both generators. The fan chart is saved
to `results/fan_chart.png`.

## Status

v1 and v2 and v3 are built, tested, and committed, along with the fan chart and tail
metrics.