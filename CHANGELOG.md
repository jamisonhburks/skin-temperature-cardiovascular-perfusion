# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026

Initial public release: a clean, modular reference implementation of the analysis
in *Wearable-derived skin temperature dynamics during sleep reveal cardiovascular
perfusion deficits through mechanistic modeling* (npj Digital Medicine, 2026).

### Added
- Pipeline stages, one module each: sleep-window segmentation; Pearson /
  cross-correlation global relationships; directed transfer entropy (pure-NumPy
  estimator with an optional `pyinform` backend); isolated-perturbation
  trajectory selection; incremental OLS models (Eqs. 1–3); the state-dependent
  mechanistic model (Eqs. 4–6); and cohort effect-size + Table 1 analyses.
- A synthetic Oura-like data generator that plants condition effects via the
  paper's own forward model, so the full pipeline runs and recovers them without
  the (non-redistributable) real data.
- CLI stage scripts, figure generation, a narrated tutorial notebook, and a
  pytest suite of estimator invariants and effect-recovery checks.
