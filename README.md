# skin-temperature-cardiovascular-perfusion

**Physics-informed modeling of wearable skin-temperature and activity coupling during sleep.**

A clean, modular reference implementation of the analysis in:

> Burks, Hartogensis, Dilchert, Mason & Smarr.
> *Wearable-derived skin temperature dynamics during sleep reveal cardiovascular
> perfusion deficits through mechanistic modeling.*
> **npj Digital Medicine** 9:464 (2026). <https://doi.org/10.1038/s41746-026-02633-2>

---

## The idea in one paragraph

Average skin temperature is a noisy population marker because temperature is
pushed around by many causes. Instead of treating minute-by-minute skin
temperature as independent samples, this work treats it as a **dynamical
system**: after an isolated burst of movement (an *activity perturbation*) during
sleep, distal skin temperature dips and recovers along a characteristic
trajectory. Modeling that trajectory with a state-dependent, physics-informed
model (Newton's law of heating + a transient activity residual) explains far more
variance than an equally-parsimonious linear model, and its four parameters
separate cohorts with cardiovascular conditions (coronary artery disease,
diabetes, hypertension, atrial fibrillation) that classical statistics miss.

The four **dynamical biomarkers**:

| symbol | name              | physiological reading                              |
|:------:|-------------------|----------------------------------------------------|
| `k₁`   | recovery decay    | thermal inertia + resting vascular tone            |
| `a`    | MET amplitude gain| sensitivity of the vasoconstrictive response       |
| `b`    | onset steepness   | vasoconstrictive elasticity                        |
| `k₂`   | offset steepness  | refractory / vasodilative recovery                 |

---

## Reproducibility without the private data

The Oura Ring data used in the paper **cannot be redistributed**. To keep the
whole pipeline runnable and its findings reproducible, this repo ships a
**synthetic data generator** that builds nights from the paper's own forward
model with planted age/sex/condition effects. Running the pipeline on it recovers
those effects (see *Results* below), which validates the estimators and the code.
Point any script at `data/raw/` to run on real exports instead — the input schema
is identical (see [`data/raw/README.md`](data/raw/README.md)).

---

## Install

```bash
git clone <this-repo> && cd skin-temperature-cardiovascular-perfusion
python -m venv .venv && source .venv/bin/activate
pip install -e .           # add ".[te,dev]" for pyinform + test/dev tools
```

Requires Python ≥ 3.10. Transfer entropy uses a bundled pure-NumPy estimator by
default; `pip install -e ".[te]"` enables the optional `pyinform` backend.

## Quickstart

Run the whole thing on synthetic data (a few minutes end-to-end):

```bash
make demo          # generate data → all stages → figures
```

or step by step:

```bash
python scripts/00_generate_synthetic_data.py --n-participants 600 --n-nights 14
python scripts/01_global_relationships.py      # Fig. 2  (Pearson, lead/lag, TE)
python scripts/02_extract_perturbations.py     # isolated-perturbation trajectories
python scripts/03_fit_linear_models.py         # Eqs. 1–3 (Fig. 3)
python scripts/04_fit_mechanistic.py           # Eqs. 4–6 (Figs. 4–5)
python scripts/05_cohort_analysis.py           # Fig. 5, Table 1
python scripts/make_figures.py                 # writes results/figures/*.png,*.pdf
```

Every script takes `--data-dir` (default `data/synthetic`), so the same commands
run on real data in `data/raw/`. See also the narrated walkthrough in
[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb).

## Use it as a library

```python
import numpy as np
from skin_temp_perfusion import extract_sleep_windows, extract_trajectories, fit_trajectory

windows = extract_sleep_windows(participant_df)          # contiguous ≥5 h nights
trajectories = extract_trajectories(windows, pid="P0001")# isolated perturbations
fit = fit_trajectory(trajectories.iloc[0][[f"T{i}" for i in range(20)]].to_numpy(),
                     met0=trajectories.iloc[0]["met0"])
print(fit.k1, fit.a, fit.b, fit.k2, fit.residual_r2)
```

---

## Repository layout

```
skin-temperature-cardiovascular-perfusion/
├── src/skin_temp_perfusion/       # the package — one module per pipeline stage
│   ├── config.py                 # constants + PipelineConfig (all hyper-parameters)
│   ├── io.py                     # schema-validated loading
│   ├── sleep.py                  # sleep-window segmentation
│   ├── correlations.py           # Pearson + cross-correlation lead/lag   (Fig. 2)
│   ├── transfer_entropy.py       # directed MET→ΔT information flow        (Fig. 2)
│   ├── perturbations.py          # isolated-perturbation trajectory select (Fig. 7)
│   ├── linear_models.py          # incremental OLS, Eqs. 1–3               (Fig. 3)
│   ├── mechanistic.py            # state-dependent model, Eqs. 4–6         (Figs. 4–5)
│   ├── cohorts.py                # cohort effect sizes + Table 1           (Figs. 5–6)
│   ├── stats_utils.py            # effect sizes, Bonferroni / BH corrections
│   ├── plotting.py               # paper-analogous figures
│   ├── pipeline.py               # high-level orchestration
│   └── synthetic/generate.py     # forward-model data generator
├── scripts/                      # thin CLI entry points, one per stage
├── notebooks/tutorial.ipynb      # narrated end-to-end walkthrough
├── tests/                        # pytest suite (estimator invariants + smoke)
├── docs/METHODS.md               # code ↔ paper equation/figure cross-reference
└── data/  results/               # git-ignored inputs/outputs (schema docs inside)
```

Design notes: computation is pure and config-driven (functions take a
`PipelineConfig`, not globals); plotting is fully separated from analysis; the
transfer-entropy estimator has no hard third-party dependency. Details in
[`docs/METHODS.md`](docs/METHODS.md).

---

## Results on synthetic data

Running the pipeline on 600 synthetic participants recovers the two headline
results of the paper:

**1. The mechanistic model beats the linear models** at the same parameter count
(median R², example run) — the whole-trajectory fit averages noise that a single
instantaneous difference cannot:

| model            | median R² |
|------------------|-----------|
| linear, 1-var    | 0.39 |
| linear, 2-var    | 0.40 |
| linear, 3-var    | 0.59 |
| **mechanistic**  | **0.89** |

**2. The biomarkers separate condition cohorts** — **87% of comparisons carry the
correct sign** (controls higher, since conditions lower the parameters) with
several surviving Bonferroni + CI, and the *pattern* mirrors the paper: CAD and
diabetes strongest on `k₁`, hypertension/AF on `b`/`k₂`, the H+D comorbid group
strongest overall. Median Cohen's *d* (control − condition):

| condition | `k₁` | `a`  | `b`  | `k₂` |
|-----------|------|------|------|------|
| CAD       | 0.48 | 0.04 | 0.29 | 0.28 |
| D         | 0.54 | −0.01| 0.19 | 0.37 |
| H         | 0.28 | 0.06 | 0.41 | 0.32 |
| AF        | 0.22 | 0.06 | 0.38 | 0.40 |
| H+D       | 0.57 | −0.08| 0.69 | 0.85 |

These validate the estimators and code path — not human physiology (the real
findings are in the paper). Synthetic condition prevalences and effect sizes are
set larger than the real cohort's so the effects are detectable at this repo's
modest sample size. See
[`docs/METHODS.md`](docs/METHODS.md#what-the-synthetic-data-does-and-does-not-show).

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The suite checks estimator **invariants** (transfer entropy is directional and
non-negative; the mechanistic fit recovers parameters monotonically; nested OLS
R² is non-decreasing) plus an end-to-end smoke test on a tiny synthetic dataset.

## Contributing

New collaborators: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for a 5-minute setup,
the three commands you'll use (`pytest`, `ruff check`, `ruff format`), and the
repo's conventions. CI runs lint + tests on every push and PR across Python
3.10–3.12.

## Citing

If you use this software or its methods, please cite the paper (see
[`CITATION.cff`](CITATION.cff)). Code is released under the [MIT License](LICENSE);
the manuscript is CC-BY-4.0.
