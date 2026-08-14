"""Central configuration, physical constants, and column conventions.

Every tunable in the analysis lives here so that a reader can see, in one
place, exactly which choices reproduce the paper and which are free parameters.
The defaults reproduce the pipeline described in:

    Burks et al., "Wearable-derived skin temperature dynamics during sleep
    reveal cardiovascular perfusion deficits through mechanistic modeling",
    npj Digital Medicine (2026). https://doi.org/10.1038/s41746-026-02633-2

Constants are grouped by pipeline stage and cross-referenced to the Methods
section of the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
#: Repository root, resolved from this file's location.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# --------------------------------------------------------------------------- #
# Device / signal conventions (Oura Ring Gen2)
# --------------------------------------------------------------------------- #
#: Lowest valid MET value the Oura data stream reports. All "activity above
#: baseline" is measured relative to this floor (Methods; MET step size 0.1).
MET_BASELINE: float = 0.9

#: A sample is an "activity event" when MET exceeds the baseline by more than
#: this tolerance. MET is quantized to 0.1, so anything > ~0 counts.
MET_EVENT_TOL: float = 0.01

#: Sampling period of the temperature / MET streams, in seconds.
SAMPLE_PERIOD_S: int = 60

#: Thermistor resolution in degrees Celsius (reported spec, used by the
#: synthetic generator to quantize temperature realistically).
TEMP_RESOLUTION_C: float = 0.07


# --------------------------------------------------------------------------- #
# Canonical column names
# --------------------------------------------------------------------------- #
COL_TEMP = "temp_skin"      # distal skin temperature, T (°C)
COL_MET = "met"             # metabolic equivalents (activity proxy)
COL_AWAKE = "is_awake"      # boolean sleep/wake label (Oura algorithm)
COL_HR = "HR"               # heart rate (optional; not used by core models)
COL_PID = "PID"             # participant identifier

#: Cardiovascular / cardiac condition columns expected in the demographics
#: table (self-reported survey items). `cd_none` == 1 marks the control cohort.
CONDITION_COLUMNS: tuple[str, ...] = (
    "cd_none",
    "cd_hypertension",
    "cd_diabetes",
    "cd_coronary",     # coronary artery disease (CAD)
    "cd_afib",         # atrial fibrillation
    "cd_myocardial",
    "cd_congestive",
    "cd_stroke",
    "cd_sleepapnea",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable bundle of every analysis hyper-parameter.

    Grouping the knobs in a frozen dataclass keeps functions pure (they take a
    config, not global state) and makes an experiment fully described by one
    printable object. Defaults reproduce the paper.

    Attributes
    ----------
    min_sleep_samples
        Minimum contiguous asleep samples (minutes) for a night to be analysed.
        300 samples == 5 hours (Methods, "Selection of participant data").
    trajectory_len
        Number of minutes of temperature followed after an activity
        perturbation (the "temperature trajectory", TT).
    pre_window, post_window
        A non-minimum MET sample is an isolated *activity perturbation* (AP)
        only if no other activity event occurs within ``pre_window`` minutes
        before or ``post_window`` minutes after it (Methods; the paper uses a
        symmetric 20-minute quiet zone).
    recovery_start_min
        Minute from which the Newton's-law recovery constant ``k1`` is fit;
        the earlier "perturbed phase" is excluded (Methods, Eq. 4 / Supp Fig 1
        found ~95% of trajectories fit best at ≥10 min).
    recovery_r2_min
        A trajectory is kept for residual modelling only if its Eq. 4 recovery
        fit reaches this R² (Eq. 6).
    residual_r2_min
        A trajectory's Eq. 5 parameters are pooled for cohort comparison only
        if the residual fit reaches this R².
    te_history
        History length ``k`` for the transfer-entropy estimator.
    te_dtemp_bins
        Number of quantile bins used to discretize ΔT for transfer entropy.
    te_n_shuffles
        Circular/permutation shuffles used to build the TE null distribution.
    te_min_events, te_max_events
        A night contributes to the TE analysis only when its number of activity
        events falls in ``[te_min_events, te_max_events)``.
    min_trajectories_per_participant
        Minimum trajectories a participant needs before an individualized OLS
        model (Eqs. 1–3) is fit.
    age_cutoff
        Age (years) splitting the "younger" and "older" cohorts (Methods; 45 y,
        chosen near the mean diagnosis age of diabetes/hypertension).
    min_cohort_size
        Minimum participants in a condition cohort before it is tested.
    n_comparisons
        Total planned comparisons, used for the Bonferroni threshold
        (paper: 120 → α ≈ 4.1e-4).
    alpha
        Family-wise significance level before correction.
    bootstrap_iters
        Control-cohort resamples used to build the effect-size confidence band.
    random_seed
        Global seed for every stochastic step (shuffles, bootstrap, synthetic
        data) so runs are reproducible.
    """

    # --- sleep window selection ---
    min_sleep_samples: int = 300

    # --- perturbation trajectory selection ---
    trajectory_len: int = 20
    pre_window: int = 20
    post_window: int = 20

    # --- mechanistic model (Eqs. 4–6) ---
    recovery_start_min: int = 10
    recovery_r2_min: float = 0.80
    residual_r2_min: float = 0.50

    # --- transfer entropy ---
    te_history: int = 2
    te_dtemp_bins: int = 5
    te_n_shuffles: int = 100
    te_min_events: int = 5
    te_max_events: int = 20

    # --- linear models (Eqs. 1–3) ---
    min_trajectories_per_participant: int = 50

    # --- cohort statistics ---
    age_cutoff: int = 45
    min_cohort_size: int = 10
    n_comparisons: int = 120
    alpha: float = 0.05
    bootstrap_iters: int = 1000

    # --- reproducibility ---
    random_seed: int = 0

    #: The four dynamical biomarkers, in paper order (k1, a, b, k2).
    dynamical_params: tuple[str, ...] = field(
        default=("k1", "a", "b", "k2"), repr=False
    )

    @property
    def bonferroni_alpha(self) -> float:
        """Bonferroni-corrected significance threshold, ``alpha / n_comparisons``."""
        return self.alpha / self.n_comparisons


#: A ready-to-use default configuration reproducing the manuscript.
DEFAULT_CONFIG = PipelineConfig()
