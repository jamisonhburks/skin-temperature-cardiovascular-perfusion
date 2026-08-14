"""Synthetic Oura-like data generator.

The participant-level TemPredict/Oura data used in the paper are not
redistributable. To keep the repository runnable and its findings reproducible,
this module synthesizes data *from the paper's own forward model*: each night is
built from state-dependent Newton recovery (Eq. 4) plus the activity residual
(Eq. 5), with per-participant dynamical parameters shifted by planted age, sex,
and condition effects. Running the full pipeline on this data therefore recovers
the planted effects — a self-consistency check on the estimators, not a claim
about real physiology.

Output matches the real schema exactly:
    <out_dir>/<PID>.parquet      columns: temp_skin, met, is_awake, HR
    <out_dir>/demographics.parquet   index PID; age, sex, bmi, cd_* flags
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    COL_AWAKE,
    COL_HR,
    COL_MET,
    COL_PID,
    COL_TEMP,
    MET_BASELINE,
    TEMP_RESOLUTION_C,
)
from ..mechanistic import residual_effect

__all__ = ["DynParams", "generate_participant", "generate_dataset"]

# Population medians for the four biomarkers. Their *shape* follows the paper
# (sharp early residual, minute-scale recovery); absolute scales are chosen so
# the four parameters are comfortably identifiable above measurement noise on
# synthetic data — this generator demonstrates estimator recovery, not physical
# units. The recovery constant k1 is deliberately slow enough that the min-10→20
# "smooth recovery phase" retains curvature.
_POP_MEDIAN = {"k1": 0.15, "a": 6.0, "b": 6.0, "k2": 3.3}

# Planted multiplicative log-effects (negative ⇒ condition lowers the parameter,
# so condition-free controls sit higher — the Cohen's d > 0 sign in the paper).
# Magnitudes are set a little larger than the real study's so the effects are
# detectable at this repo's (necessarily small) synthetic cohort sizes.
_CONDITION_EFFECTS: dict[str, dict[str, float]] = {
    "cd_coronary":     {"k1": -0.20, "a": -0.16, "b": -0.20, "k2": -0.24},
    "cd_diabetes":     {"k1": -0.28, "a": -0.08, "b": -0.26, "k2": -0.40},
    "cd_hypertension": {"k1": -0.06, "a": -0.10, "b": -0.24, "k2": -0.28},
    "cd_afib":         {"k1": -0.04, "a": -0.04, "b": -0.26, "k2": -0.26},
}
_AGE_EFFECT = {"k1": -0.10, "a": -0.09, "b": -0.11, "k2": -0.11}  # per SD of age
_SEX_FEMALE_EFFECT = {"k1": -0.20, "a": 0.0, "b": -0.02, "k2": 0.03}

# Condition prevalences. Set higher than the real cohort so that, at this repo's
# modest synthetic sample sizes, each sex × age × condition cell is large enough
# to estimate a stable, correctly-signed effect (the real study had thousands of
# participants; here we trade epidemiological realism for cohort-cell size).
_PREVALENCE = {
    "cd_hypertension": 0.32,
    "cd_diabetes": 0.16,
    "cd_coronary": 0.12,
    "cd_afib": 0.10,
    "cd_myocardial": 0.04,
    "cd_congestive": 0.03,
    "cd_stroke": 0.03,
    "cd_sleepapnea": 0.12,
}


@dataclass(frozen=True)
class DynParams:
    """A participant's latent dynamical parameters used to synthesize nights."""

    k1: float
    a: float
    b: float
    k2: float


def _sample_demographics(rng: np.random.Generator) -> dict:
    """Draw age, sex, BMI and correlated condition flags for one participant."""
    age = float(np.clip(rng.normal(48, 15), 18, 90))
    sex = rng.choice(["male", "female"])
    bmi = float(np.clip(rng.normal(27, 5), 16, 50))

    demo = {"age": age, "sex": sex, "bmi": bmi}
    for cond, prev in _PREVALENCE.items():
        p = prev + 0.15 * demo.get("cd_diabetes", 0) if cond == "cd_hypertension" else prev
        demo[cond] = int(rng.random() < p)
    demo["cd_none"] = int(sum(demo[c] for c in _PREVALENCE) == 0)
    return demo


def _participant_params(demo: dict, rng: np.random.Generator) -> DynParams:
    """Map demographics + a random effect to latent (k1, a, b, k2)."""
    age_z = (demo["age"] - 48) / 15
    values = {}
    for name, median in _POP_MEDIAN.items():
        log_effect = _AGE_EFFECT[name] * age_z
        if demo["sex"] == "female":
            log_effect += _SEX_FEMALE_EFFECT[name]
        for cond, effects in _CONDITION_EFFECTS.items():
            log_effect += effects.get(name, 0.0) * demo[cond]
        log_effect += rng.normal(0, 0.10)  # idiosyncratic between-person spread
        values[name] = float(median * np.exp(log_effect))
    return DynParams(**values)


def _ou_drift(n: int, mu: float, rng: np.random.Generator) -> np.ndarray:
    """Ornstein–Uhlenbeck steady-state drift, giving a slowly wandering baseline."""
    theta, sigma = 0.02, 0.02
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return np.clip(x, 33.0, 37.0)


# Per-sample measurement + physiological noise. Large enough that a single
# minute-to-minute difference (ΔT₁) is noise-dominated — so MET₀ alone predicts
# it weakly, as in the paper — yet small versus the multi-minute perturbation
# dip, so the 20-point mechanistic fit still recovers the trajectory shape.
_TEMP_NOISE = 0.035
# Std of the persistent steady-state step at each perturbation ("the steady
# state changes through the night due to hand location"): this is what makes T₀
# and T_end informative predictors beyond MET₀.
_STEADY_STEP = 0.25


def _perturbation_trajectory(
    t0: float, t_end: float, magnitude: float, params: DynParams,
    traj_len: int, rng: np.random.Generator, noise: float = _TEMP_NOISE,
) -> np.ndarray:
    """One trajectory from the same discrete state-space model the fitter inverts.

    Generating from ``Tₜ = Tₜ₋₁ + k₁·(T_end − Tₜ₋₁) − ε_T(t)`` — the exact
    recursion :func:`mechanistic._one_step_residual` reconstructs — makes the
    planted (k₁, a, b, k₂) *consistently* recoverable: the two-stage estimator
    pins the steady state to the last sample and fits k₁ on a short tail, so
    absolute values carry a known bias, but condition effects propagate faithfully.
    """
    tau = np.arange(traj_len, dtype=float)
    resid = residual_effect(tau, magnitude, params.a, params.b, params.k2)
    y = np.empty(traj_len)
    y[0] = t0
    for i in range(1, traj_len):
        y[i] = y[i - 1] + params.k1 * (t_end - y[i - 1]) - resid[i]
    return y + rng.normal(0, noise, traj_len)


def _generate_night(
    params: DynParams,
    length: int,
    traj_len: int,
    rng: np.random.Generator,
    *,
    quantize: bool = False,
) -> pd.DataFrame:
    """Synthesize one night's temperature/MET/HR from the forward model.

    Steady-state temperature is a piecewise-constant level that steps at each
    activity perturbation; between perturbations it is flat (plus noise). Every
    perturbation is well isolated by construction — the isolation filter's
    rejection of clustered events is covered by the unit tests.
    """
    met = np.full(length, MET_BASELINE)
    temp = np.empty(length)
    level = float(rng.normal(35.0, 0.4))

    i = int(rng.integers(20, 40))                      # leading quiet baseline
    temp[:i] = level + rng.normal(0, _TEMP_NOISE, i)
    while i < length - traj_len - 5:
        magnitude = min(round(rng.exponential(0.15) + 0.1, 1), 0.6)
        met[i] = MET_BASELINE + magnitude
        # Steady state drifts to a new level over the trajectory (hand movement,
        # microclimate); mild mean reversion keeps it in a physiological band.
        new_level = float(np.clip(0.95 * level + 0.05 * 35.0
                                  + rng.normal(0, _STEADY_STEP), 33.0, 37.0))
        temp[i : i + traj_len] = _perturbation_trajectory(
            level, new_level, magnitude, params, traj_len, rng
        )
        level = new_level
        i += traj_len
        gap = int(rng.integers(8, 55))                 # quiet baseline before next
        end = min(i + gap, length)
        temp[i:end] = level + rng.normal(0, _TEMP_NOISE, end - i)
        i = end
    temp[i:] = level + rng.normal(0, _TEMP_NOISE, length - i)

    if quantize:  # optional: emulate the real 0.07 °C thermistor step
        temp = np.round(temp / TEMP_RESOLUTION_C) * TEMP_RESOLUTION_C
    hr = 55 + 2.0 * (temp - temp.mean()) + rng.normal(0, 2, length)  # loose coupling
    return pd.DataFrame({COL_TEMP: temp, COL_MET: met, COL_AWAKE: False, COL_HR: hr})


def generate_participant(
    pid: str,
    *,
    n_nights: int = 20,
    trajectory_len: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate a full record (many nights + awake gaps) and demographics for one PID."""
    rng = np.random.default_rng() if rng is None else rng
    demo = _sample_demographics(rng)
    params = _participant_params(demo, rng)

    blocks: list[pd.DataFrame] = []
    for _ in range(n_nights):
        night = _generate_night(params, int(rng.integers(320, 460)), trajectory_len, rng)
        blocks.append(night)
        # Short "awake" gap so sleep-window segmentation has boundaries to find.
        gap_len = int(rng.integers(20, 45))
        blocks.append(
            pd.DataFrame(
                {
                    COL_TEMP: rng.normal(33.5, 0.5, gap_len),
                    COL_MET: rng.uniform(0.9, 1.5, gap_len),
                    COL_AWAKE: True,
                    COL_HR: rng.normal(70, 5, gap_len),
                }
            )
        )
    record = pd.concat(blocks, ignore_index=True)
    demo[COL_PID] = pid
    return record, demo


def generate_dataset(
    n_participants: int,
    out_dir: str | Path,
    *,
    n_nights: int = 20,
    trajectory_len: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate ``n_participants`` synthetic records and a demographics table.

    Writes ``<PID>.parquet`` per participant plus ``demographics.parquet`` into
    ``out_dir`` and returns the demographics frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any prior synthetic files so runs with different seeds/sizes never
    # mix (a stale P0499.parquet must not survive a smaller regeneration).
    for stale in [*out_dir.glob("P*.parquet"), out_dir / "demographics.parquet"]:
        stale.unlink(missing_ok=True)
    master = np.random.default_rng(seed)

    demographics: list[dict] = []
    for i in range(n_participants):
        pid = f"P{i:04d}"
        # Independent child stream per participant → reproducible & order-free.
        rng = np.random.default_rng(master.integers(0, 2**63))
        record, demo = generate_participant(
            pid, n_nights=n_nights, trajectory_len=trajectory_len, rng=rng
        )
        record.to_parquet(out_dir / f"{pid}.parquet")
        demographics.append(demo)

    demo_df = pd.DataFrame(demographics).set_index(COL_PID)
    demo_df.to_parquet(out_dir / "demographics.parquet")
    return demo_df
