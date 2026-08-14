"""skin_temp_perfusion — physics-informed modeling of wearable skin-temperature
and activity coupling during sleep.

Reference implementation of the analysis in Burks et al., *npj Digital Medicine*
(2026), https://doi.org/10.1038/s41746-026-02633-2.

The package is organized as one module per pipeline stage:

    config          hyper-parameters, constants, column conventions
    io              schema-validated loading of participant/demographics data
    sleep           segmentation into contiguous sleep windows
    correlations    Pearson / cross-correlation global relationships (Fig. 2)
    transfer_entropy directed information flow MET → ΔT (Fig. 2)
    perturbations   isolated activity-event trajectory selection (Fig. 7)
    linear_models   incremental OLS models, Eqs. 1–3 (Fig. 3)
    mechanistic     state-dependent Newton + residual model, Eqs. 4–6 (Figs. 4–5)
    cohorts         condition-cohort effect sizes and Table 1 (Figs. 5–6)
    stats_utils     effect sizes and multiple-comparison corrections
    plotting        paper-analogous figures
    pipeline        high-level orchestration of the stages
    synthetic       forward-model data generator (no real data required)

A typical end-to-end run lives in :mod:`skin_temp_perfusion.pipeline`.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, PipelineConfig
from .mechanistic import TrajectoryFit, fit_trajectory, newton_recovery, residual_effect
from .perturbations import extract_trajectories, find_isolated_perturbations
from .sleep import extract_sleep_windows
from .transfer_entropy import transfer_entropy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "extract_sleep_windows",
    "find_isolated_perturbations",
    "extract_trajectories",
    "transfer_entropy",
    "fit_trajectory",
    "TrajectoryFit",
    "newton_recovery",
    "residual_effect",
]
