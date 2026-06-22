<!-- Copied into repo to guide AI coding agents. Keep concise and actionable. -->
# Copilot instructions for contributors and AI agents

This project is a research-grade Monte Carlo stock forecasting engine. Below are the minimal, practical facts an AI coding agent needs to be productive working here.

## Big picture
- Product: produce a terminal price *distribution* for a ticker + horizon (fan chart, percentiles, P(loss)).
- Layers: data -> return generators -> simulation engine -> evaluation. See `src/engine.py` and `src/models/base.py` for the central contract.

## Key interfaces & invariants
- `ReturnGenerator` (see src/models/base.py): implement `fit(log_returns: pd.Series) -> self` and `generate(horizon, n_paths, rng) -> np.ndarray`.
  - `fit()` must set `self._fitted = True`.
  - `generate()` receives a NumPy `Generator` and must not touch global RNG state.
  - `generate()` returns shape `(n_paths, horizon)` of log-returns (not prices).
- `run_simulation(generator, start_price, horizon, n_paths, seed)` (see src/engine.py): the engine cum-sums log-returns and reconstructs price paths; do not duplicate price logic in generators.

## Data & caching
- Historical prices are fetched with `src/data.py::fetch_prices()` (uses `yfinance` lazily) and cached to `data/raw/<TICKER>.csv`. Tests and demos use the cache by default. First run needs network.
- Use `src/data.py::to_log_returns()` to obtain the log-return series that generators expect.

## Existing runnable demo & common commands
- Create environment: `conda env create -f environment.yml && conda activate stock-forecast`.
- Run end-to-end demo: `python scripts/demo.py NVDA 1260` (defaults shown in script). This fetches data, fits `GBMGenerator`, runs 10k MC paths, and prints percentiles.

## Project-specific patterns & conventions
- Separation of concerns is deliberate: generators model returns only; engine handles horizons, path counts and price reconstruction. Follow this pattern for new generators.
- RNG is explicit: always accept a `numpy.random.Generator` in `generate()` so runs are reproducible.
- Drift handling: many models include an explicit `drift_override` (see `src/models/gbm.py`) — do not silently replace historical drift; prefer explicit override or sweeps.
- Feature engineering belongs in `src/features.py` (e.g. `realised_vol`) and should be passed into generators rather than imported inside modelling code.
- Evaluation focuses on calibration/coverage (`src/evaluate.py`) not point forecasts. New model work should include one-step calibration checks.

## Where to look for examples
- Baseline generator: [src/models/gbm.py](src/models/gbm.py)
- Engine usage & result handling: [src/engine.py](src/engine.py)
- Demo flow: [scripts/demo.py](scripts/demo.py)
- Data and cache semantics: [src/data.py](src/data.py)
- Evaluation primitives (coverage, walk-forward): [src/evaluate.py](src/evaluate.py)

## Small implementation checklist for new generators
1. Add subclass in `src/models/` with clear name.
2. Implement `fit()` returning `self` and set `_fitted = True`.
3. Implement `generate(horizon, n_paths, rng)` returning `(n_paths, horizon)` log-returns.
4. Write a short demo or unit that uses `scripts/demo.py` pattern (fitting once, simulating many times).
5. Add coverage/calibration checks using `src/evaluate.py` functions.

## Do / Don't (quick)
- Do: keep generators stateless wrt global RNG, return log-returns, document drift choices.
- Don't: reconstruct prices inside generators, assume availability of cached CSVs on first run, or change the engine's signature.

---
If anything above is unclear or you'd like more examples (unit tests, a notebook snippet, or a guided PR template), say which part to expand. 
