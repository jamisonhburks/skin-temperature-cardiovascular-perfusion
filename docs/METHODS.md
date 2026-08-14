# Methods — code ↔ paper cross-reference

This document maps every stage of the analysis to the module that implements it
and to the corresponding equation, figure, or Methods subsection of:

> Burks, Hartogensis, Dilchert, Mason & Smarr. *Wearable-derived skin
> temperature dynamics during sleep reveal cardiovascular perfusion deficits
> through mechanistic modeling.* npj Digital Medicine 9:464 (2026).
> https://doi.org/10.1038/s41746-026-02633-2

The scientific claim is that a **physics-informed, state-dependent model** of
skin-temperature trajectories after activity perturbations explains far more
variance than an equally-parsimonious linear model, and that its parameters
separate cohorts with cardiovascular conditions where linear coefficients do not.

---

## Pipeline overview

```
 participant parquet (temp_skin, met, is_awake)
        │
        ▼  sleep.extract_sleep_windows                 (Methods: participant selection)
 contiguous ≥5 h asleep windows
        │
        ├─▶ correlations + transfer_entropy            → Fig. 2   (global relationships)
        │
        ▼  perturbations.extract_trajectories          (Methods: TT selection; Fig. 7)
 isolated-perturbation temperature trajectories
        │
        ├─▶ linear_models.fit_participant_linear_models  Eqs. 1–3 → Fig. 3
        │
        ▼  mechanistic.fit_trajectory                    Eqs. 4–6 → Figs. 4–5
 per-trajectory (k₁, a, b, k₂)
        │
        ▼  cohorts.{aggregate,cohort_effect_sizes,parameter_multilinear_models}
 per-participant medians → Mann–Whitney U + Cohen's d → Fig. 5, Table 1
```

---

## Stage-by-stage

### 1. Sleep-window segmentation — `sleep.py`
A "night" is a maximal contiguous run of asleep samples of length ≥ 300 minutes
(5 h). Runs are found by incrementing a group id whenever `is_awake` flips.
*(Methods: "Selection of participant data for activity and temperature coupling".)*

### 2. Global relationships — `correlations.py`, `transfer_entropy.py`
- **Pearson** of MET vs. `T` and MET vs. `ΔT` (minute differences). ΔT is the
  better covariate for most participants. *(Fig. 2B–D.)*
- **Cross-correlation** lead/lag: the max-|corr| lag is negative → MET *leads*
  ΔT. *(Fig. 2E.)*
- **Transfer entropy** TE(MET→ΔT) vs. TE(ΔT→MET), with a shuffle-based null.
  TE is asymmetric and nonlinear, giving directional evidence. *(Fig. 2F–G.)*
  A self-contained plug-in estimator (nats) is the default; `pyinform` gives
  exact parity.

### 3. Perturbation-trajectory selection — `perturbations.py`
An **isolated activity perturbation (AP)** is a non-baseline MET sample with no
other activity within a 20-minute quiet zone before and after. Around each AP a
20-minute temperature trajectory is extracted. *(Methods: "Selection of
temperature trajectories after activity perturbations"; Fig. 7A.)*

### 4. Linear models (Eqs. 1–3) — `linear_models.py`
Three nested OLS models per participant predict ΔT₁ from MET₀, then + T₀, then
+ T_end. Fit improves (median R² 0.04 → 0.19 → 0.35) but T₀ and T_end are
colinear — motivating a state-dependent model. *(Fig. 3.)*

### 5. Mechanistic model (Eqs. 4–6) — `mechanistic.py`
Two-stage fit per trajectory:
1. **Recovery (Eq. 4):** `ΔTₜ = k₁·(T_end − Tₜ)`. `k₁` is estimated on the late
   "smooth recovery phase" (minutes ≥ 10) where the activity effect is
   negligible, then used to predict the whole trajectory. Kept only if the
   recovery fit reaches R² ≥ 0.8 (Eq. 6).
2. **Residual (Eq. 5):** the one-step prediction error is fit to
   `ε_T(t) = a·M₀·t^b·e^(−k₂·t)`.

The four parameters are interpreted physiologically: `k₁` recovery/thermal
inertia; `a` MET amplitude gain; `b` onset steepness; `k₂` offset steepness.
*(Figs. 4–5.)*

> **Note on the estimator.** The two-stage fit is a *simplified, deterministic
> inversion* of the forward model, not an exact one (it pins the steady state to
> the last observed sample and fits `k₁` on a short tail). Absolute parameter
> values are therefore biased, but the estimator is consistent, so condition
> effects propagate. The unit tests assert this consistency (monotone recovery)
> rather than exact value recovery.

### 6. Cohort comparison & Table 1 — `cohorts.py`, `stats_utils.py`
Each participant is reduced to a **median** parameter (IID comparisons), keeping
only trajectories with residual R² ≥ 0.5. Condition cohorts (AF, H, D, H+D, CAD)
are compared to sex/age-matched controls with **Mann–Whitney U** tests and
**Cohen's d**, with a bootstrap CI from size-matched control resampling. A result
is flagged significant when it passes **Bonferroni** correction (α/120) *and* the
CI excludes zero. Per-parameter OLS models (`param ~ age + sex + conditions`)
reproduce Table 1. *(Fig. 5, Table 1.)*

Both Bonferroni and Benjamini–Hochberg corrections are available in
`stats_utils` per the project's statistical standards.

---

## What the synthetic data does (and does not) show

`synthetic/generate.py` builds nights from the *forward* model with planted age,
sex, and condition effects. Running the pipeline on it recovers both headline
results: the mechanistic model's R² (~0.89) clearly exceeds the linear models'
(~0.39/0.40/0.59), and ~87% of cohort comparisons show the planted sign, with CAD
and diabetes strongest on `k₁`, hypertension/AF on `b`/`k₂`, and the H+D comorbid
group strongest overall — mirroring the paper's qualitative pattern. This
validates the **estimators and the code path**; it is not independent evidence
about human physiology.

Two synthetic knobs are deliberately un-realistic to make a small demo legible:
- **Steady-state steps + measurement noise** (`_STEADY_STEP`, `_TEMP_NOISE`) make
  a single-step ΔT₁ noise-dominated, so the linear models are weak while the
  20-point mechanistic fit still recovers the trajectory — reproducing the
  paper's central *mechanistic ≫ linear* gap.
- **Condition prevalences and effect sizes** are set larger than the real cohort
  so that each sex × age × condition cell is big enough to yield a stable,
  correctly-signed, sometimes-significant effect at a few hundred participants
  (the real study had thousands).
