# Contributing to `skin-temperature-cardiovascular-perfusion`

Thanks for your interest! This guide gets you productive in a few minutes.

## Development setup

```bash
git clone <this-repo> && cd skin-temperature-cardiovascular-perfusion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # editable install + pytest, ruff, jupyter
# optional: pip install -e ".[te,dev]"  for the pyinform transfer-entropy backend
```

Requires Python ≥ 3.10.

## The three commands you'll use

```bash
pytest                              # run the test suite (fast)
ruff check src scripts tests        # lint (config in pyproject.toml)
ruff format src scripts tests       # auto-format (optional but encouraged)
```

CI runs `ruff check` and `pytest` on every push and pull request (see
`.github/workflows/ci.yml`), so run them locally before opening a PR.

## How the repo is organized

One module per pipeline stage under `src/skin_temp_perfusion/` — see the table in
[`README.md`](README.md) and the equation/figure cross-reference in
[`docs/METHODS.md`](docs/METHODS.md). The dependency chain is:

```
sleep windows → { global relationships, perturbation trajectories }
trajectories  → { linear models (Eqs 1–3), mechanistic fits (Eqs 4–6) }
mechanistic   → cohort effect sizes + parameter multilinear models
```

Conventions worth knowing:

- **Computation is pure and config-driven.** Functions take a `PipelineConfig`
  rather than reading globals, so an experiment is fully described by one object.
- **Plotting is separate from analysis** (`plotting.py` only draws).
- **No hard third-party dependency for the core estimators.** Transfer entropy
  ships a pure-NumPy estimator; `pyinform` is an optional exact-parity backend.

### Running on real data

Every stage script takes `--data-dir`, so the same commands run on real Oura
exports placed in `data/raw/` (schema in [`data/raw/README.md`](data/raw/README.md)):

```bash
python scripts/02_extract_perturbations.py --data-dir data/raw
```

## Testing philosophy

The two-stage mechanistic estimator is a *consistent* (deterministic, monotone)
inversion, not an exact one, so tests assert **invariants and effect recovery**,
not exact numbers:
- estimator invariants (monotone parameter recovery, nested-OLS R² is
  non-decreasing, transfer entropy is directional and non-negative);
- that the pipeline recovers the *planted* condition effects on a small synthetic
  dataset (`tests/test_pipeline.py`).

Keep tests fast (seconds): small signals, few shuffles/bootstraps, fixed seeds.

## Data policy

**Never commit participant data.** `data/raw/` is git-ignored and is only a
drop-in location for real records; all shipped/generated data are synthetic.

## Pull requests

- Keep changes focused; update `docs/METHODS.md` if you touch the method↔paper
  mapping, and add a line to `CHANGELOG.md`.
- Make sure `ruff check` and `pytest` are green.
