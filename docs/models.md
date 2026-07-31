# Models: rationale, assumptions, limitations

Reference for the five return generators and the evaluation harness. The README
gives results; this gives reasoning. Every generator implements the same
`ReturnGenerator` contract — `fit(log_returns)` then
`generate(horizon, n_paths, rng) -> (n_paths, horizon)` — so the engine never
knows which one it holds.

Drift is a *user assumption*, never an assertion. Where a generator would
otherwise inherit whatever drift its sample happened to contain, it exposes
`drift_override` and re-centres without distorting shape.

---

## v1 — Geometric Brownian Motion (`gbm.py`)

Log-returns drawn i.i.d. from a Normal fitted to history.

**Assumes:** constant volatility, independent returns, Normal tails.

**Why it exists:** the baseline. Every later model is an argument against one of
its assumptions, and the calibration harness quantifies how much each relaxation
buys.

**Limitations:** cannot represent fat tails (real equities show excess kurtosis
of roughly 6-8) or volatility clustering. It is the *worst* one-step calibrator
in the 30-ticker study and never wins a single ticker. It becomes competitive at
21 steps, because summing returns pulls the cumulative distribution toward
Normal by the CLT — the one regime where its assumption is closest to true.

---

## v2 — Block bootstrap (`bootstrap.py`)

Resamples contiguous blocks of real historical returns rather than assuming any
distribution. Default block length 20 trading days.

**Assumes:** the future resembles a re-shuffling of the past; dependence beyond
the block length is ignorable.

**Why blocks, not single returns:** drawing individual returns i.i.d. would
destroy volatility clustering. A block preserves the local dependence structure
inside it, so turbulent stretches survive resampling.

**Drift control by de-meaning.** Shifting every resampled return by
`r' = r - mu_hist + mu_target` moves the mean to the chosen drift while leaving
variance, skew and kurtosis untouched. Verified empirically: re-centring NVDA
from ~35%/yr to 7%/yr left excess kurtosis unchanged.

**Limitations:** it can only replay crashes that already happened — it cannot
generate a shock larger than the worst day in its sample. And at multi-step
horizons the block length becomes a liability: over a 21-day horizon, 20-day
blocks leave too few independent draws, so the cumulative distribution
over-disperses. This is why the best one-step model (1.3% mean error, 26/30
wins) is the *worst* at 21 steps (3.5%). A stationary bootstrap with random
block lengths would likely fix this and has not been tried.

---

## v3 — Merton jump-diffusion (`jumpdiff.py`)

Normal diffusion plus Poisson-timed jumps, so simulated paths can contain shocks
absent from the historical sample.

**Estimation is a threshold heuristic, not MLE.** A day more than `k` standard
deviations from the mean (default k=3) is labelled a jump; the rest are
diffusion; the jump rate is the fraction of such days. This is the model's
honest soft spot — proper Merton estimation is a maximum-likelihood problem.
The threshold was chosen for interpretability, and lowering it to 2.5 improved
the centre of the distribution while leaving the deep tail no better.

**Where it wins:** it takes 3 of 30 tickers at one step — TSLA, MRK and AMD, all
names where discrete jumps genuinely dominate the return process. It also fixes
GBM's over-inflated centre by pulling extreme days out into a separate
component.

---

## v4 — GARCH(1,1) (`garch.py`)

Volatility as a process rather than a constant:

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

with `alpha + beta < 1` so variance mean-reverts to `omega / (1 - alpha - beta)`.

**Fitted by hand-rolled maximum likelihood** via `scipy.optimize` (SLSQP), not a
library. One implementation detail matters: returns are rescaled to unit
variance before optimising. Raw daily variances are around 1e-4, which leaves
the likelihood badly conditioned and the optimiser sitting on its starting
values. `alpha` and `beta` are scale-invariant; `omega` is recovered afterwards
by dividing by the squared scale. Validated by recovering known parameters from
synthetic data (alpha 0.080 -> 0.076, beta 0.900 -> 0.906).

**Path-dependence stays inside the contract.** Each step's variance depends on
the previous step *within the same path*, so `generate` loops over time while
vectorising across paths. The engine is unaffected.

**Why it is the default:** it is never the best by much and never badly wrong.
At 21 steps it ties Merton for the lowest mean error (2.6%) and fails on only 3
of 30 tickers. For a risk tool, reliability beats peak performance.

**Limitations:** Normal innovations mean it still under-covers the deepest tail.
GARCH-t (Student-t innovations) is the natural fix and is not yet built. It is
also symmetric in `r^2`, so it structurally cannot express the leverage effect.

---

## v5 — ML volatility-regime (`ml_vol.py`)

A `GradientBoostingRegressor` learns the map from nine backward-looking features
to forward realised volatility.

**It predicts volatility only** — never returns, direction, or events. Those are
either unlearnable from price history alone (drift) or have no training set
(crashes, regulation, wars).

**Features** (all functions of a 63-day buffer of de-meaned returns): realised
vol over 5/10/21/63 days, EWMA vol, last absolute return, mean of negative
returns over 21 days, fraction of down days, and dispersion of absolute returns.
The signed terms exist so the learner *can* express the leverage effect that
GARCH cannot.

**The learning target mattered more than the architecture.** One-step
calibration error fell monotonically as the forward window defining the target
was lengthened: 20.4% (W=2), 10.4% (W=3), 5.4% (W=5), 2.1% (W=10), 1.2% (W=21).
A single day's volatility is too noisy to learn; the same latent quantity
measured over a smoother window is far more learnable. Default is W=21.

**Leakage discipline.** For a sample at time t, features come from
`eps[t-63:t]` and the target from `eps[t:t+W]`. The windows do not overlap, and
`tests/test_models.py::test_mlvol_features_have_no_lookahead` asserts that
corrupting everything from index t onward leaves earlier features bit-identical.

**Verdict: it did not earn the default slot.** At one step it is
indistinguishable from GARCH (3.4% vs 3.3%) — a 150-tree ensemble recovers what
three parameters already capture. At 21 steps it wins nearly as often (7/30 vs
8/30) but fails on 9 tickers versus GARCH's 3, making it the highest-variance
model tested. For a risk tool that is the wrong trade, because you cannot know
in advance which kind of ticker you are holding.

The leverage effect *is* real signal — `downside_21` carries 15% of feature
importance on NVDA, and rises from 0.22 to 0.59 on synthetic data when an
asymmetry is injected. But real signal did not translate into better
calibration, which is the distinction that matters.

---

## Evaluation (`evaluate.py`)

**Calibration, not point accuracy.** The right question for a distributional
forecast is whether a c-level interval contains the realised value about a
fraction c of the time. Empirical coverage below nominal means overconfident
(intervals too narrow); above means underconfident.

**Walk-forward, never shuffled.** Each test window sits strictly after its
training window. Random k-fold would leak future information backwards and
inflate every score.

**The harness is validated before it is trusted.** On synthetic i.i.d.-Normal
data, GBM calibrates to roughly 1.8% mean error — as it must, since the model
is then correctly specified. `test_evaluate.py` encodes this: if it fails, the
evaluator is wrong, not the model.

**Known caveat:** 21-step tests use non-overlapping windows, giving ~173 points
per ticker against ~728 at one step. Per-ticker 21-step numbers are
correspondingly noisier, which is why conclusions are drawn from the 30-ticker
aggregate rather than any single name.
