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
| v5 | ML volatility-regime (GradientBoosting) | **done** - ties GARCH one-step, higher variance at horizon |
| v6+ | Ensemble of generators | planned |
| — | Streamlit app | planned — built last, over a stable backend |

## Findings

All results are produced by the calibration harness across a 30-ticker
universe (`scripts/calibration_study.py`), not by assumption.

### Which model is best depends entirely on the horizon

Mean absolute coverage error across {50, 80, 90, 95}% intervals, 30 tickers:

| Model | One-step mean | wins | 21-step mean | wins |
|-------|--------------|------|--------------|------|
| Bootstrap | **1.3%** | **26/30** | 3.5% | 7/30 |
| Merton | 2.7% | 3/30 | **2.6%** | 10/30 |
| GARCH | 3.8% | 1/30 | **2.6%** | **11/30** |
| GBM | 5.0% | 0/30 | 2.9% | 2/30 |

The ranking reverses. The bootstrap dominates one-step (26/30) and is *worst*
at 21-step; the adaptive-volatility models take over at horizon.

- **One-step:** daily returns are fat-tailed (excess kurtosis 6-8). The
  bootstrap resamples the real return shape, so it wins almost everywhere;
  GBM's Normal never wins once. The three exceptions are instructive - Merton
  takes TSLA, MRK and AMD, all names where discrete jumps dominate.
- **21-step:** the bootstrap's 20-day blocks leave too few independent draws
  over a 21-day horizon and over-disperse the cumulative distribution. GARCH's
  volatility mean-reverts across the horizon instead, and GBM improves as the
  CLT pulls multi-day returns toward Normal.

**No model dominates; the right tool depends on the timescale of the question.**
That is the project's main empirical result.

### Volatility persistence is near-universal

Fitting GARCH(1,1) across the same 30 tickers, persistence (alpha + beta) has
mean **0.951** and range **0.832-0.999** - high for every name regardless of its
volatility level, a known stylised fact recovered independently. Persistence
and volatility level are separate axes: KO and JNJ share ~17% long-run vol but
differ in persistence, while TSLA is both the most volatile (~57%) and the most
persistent (0.991, near a unit root).

### Honest limits

All four models under-cover the deepest 95% tail on real data, and tuning
Merton's jump threshold improves the centre without fixing it - the shortfall
is structural to a Normal-diffusion family, not a tuning failure. This is what
motivates a Student-t innovation variant (GARCH-t) as future work. The 21-step
tests also use fewer non-overlapping windows (~173) than the one-step tests
(~728), so their per-ticker numbers are noisier.



### v5: the ML model was built, measured, and did not earn the default slot

v5 replaces GARCH's three-parameter recursion with a gradient-boosted
regressor (scikit-learn) that learns the map from nine backward-looking
features to forward realised volatility. It predicts *volatility only* -
never returns, direction, or events - and slots in behind the same
`ReturnGenerator` contract as everything else.

It was evaluated with the same harness as v1-v4:

| Horizon | GARCH | MLVol |
|---------|-------|-------|
| one-step (8 tickers) | 3.3% mean / 3.3% median | 3.4% / 3.3% |
| 21-step (30 tickers) | 2.6% / 2.3%, wins 8/30 | 3.4% / 2.8%, wins 7/30 |

**At one step the two are indistinguishable.** A 150-tree ensemble on nine
engineered features recovers what a three-parameter model already captures and
nothing more - evidence the GARCH form is well specified for this task, not
that the learner is broken.

**At 21 steps MLVol wins almost as often as GARCH but fails far more often.**
Counting tickers with error above 4%: MLVol fails on 9 of 30 (AAPL 8.5%, AMD
and CSCO 7.9%), GARCH on 3. MLVol also produces the single best result in the
study (PEP, 0.3%). It is the highest-variance model tested.

For a risk tool that is the wrong trade. You cannot know in advance whether a
ticker will behave like PEP or like AAPL, and the failure mode of a risk model
is understating risk. A model that is reliably decent beats one that is
sometimes excellent and occasionally badly miscalibrated. **GARCH stays the
default; MLVol ships as an available generator, not the recommended one.**

Two findings worth keeping from building it:

- **The learning target mattered more than the architecture.** One-step
  calibration error fell monotonically as the forward window defining the
  target was lengthened: 20.4% (W=2), 10.4% (W=3), 5.4% (W=5), 2.1% (W=10),
  1.2% (W=21). A single day's volatility is too noisy to learn; the same
  latent quantity measured over a smoother window is far more learnable. No
  amount of tuning trees would have recovered that.
- **The leverage effect is real signal.** `downside_21` (mean of negative
  returns over 21 days) carries 15% of feature importance on NVDA, and its
  importance rises from 0.22 to 0.59 on synthetic data when an asymmetry is
  injected - the learner detects it only when present. Symmetric GARCH in r^2
  structurally cannot express this. But real signal did not translate into
  better calibration, which is the distinction that matters.

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

## Testing

    pytest

32 tests covering generator contract invariants, per-model correctness (GBM
parameter recovery, bootstrap kurtosis preservation, GARCH MLE recovering known
parameters), engine invariants (CVaR >= VaR, monotonic percentiles), and a check
that the calibration harness itself is near-perfect on synthetic Normal data.


## Status

v1 and v2 and v3 and v4 are built, tested, and committed.