"""High-level orchestration tying the analysis stages into one flow.

These helpers iterate over a directory of participant parquet files and return
tidy DataFrames, so the command-line scripts stay thin. The dependency chain is:

    sleep windows → { global relationships, perturbation trajectories }
    trajectories  → { linear models (Eqs 1–3), mechanistic fits (Eqs 4–6) }
    mechanistic   → cohort effect sizes + parameter multilinear models
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .cohorts import aggregate_participant_parameters, cohort_effect_sizes
from .config import DEFAULT_CONFIG, PipelineConfig
from .correlations import cross_correlation_lag, night_correlations
from .io import iter_participants
from .linear_models import fit_participant_linear_models
from .mechanistic import fit_trajectory
from .perturbations import extract_trajectories, temp_columns
from .sleep import extract_sleep_windows
from .transfer_entropy import night_transfer_entropy

__all__ = [
    "run_global_relationships",
    "extract_all_trajectories",
    "fit_all_linear_models",
    "fit_all_mechanistic",
    "run_cohort_analysis",
]


def _participants(data_dir, progress, desc):
    # Default glob works for real (hashed-id) and synthetic files; the loader
    # skips demographics.parquet, so no custom pattern is needed.
    it = iter_participants(data_dir)
    return tqdm(it, desc=desc) if progress else it


def run_global_relationships(
    data_dir: str | Path,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    progress: bool = True,
) -> pd.DataFrame:
    """Per-night Pearson correlations, lead/lag, and transfer entropy (Fig. 2)."""
    rows: list[dict] = []
    for pid, df in _participants(data_dir, progress, "global"):
        for night_idx, night in enumerate(extract_sleep_windows(df, config=config)):
            corr = night_correlations(night)
            lag, lag_corr = cross_correlation_lag(night)
            te = night_transfer_entropy(night, config=config)
            rows.append(
                {
                    "PID": pid,
                    "night": night_idx,
                    "r_temp": corr.r_temp,
                    "p_temp": corr.p_temp,
                    "r_dtemp": corr.r_dtemp,
                    "p_dtemp": corr.p_dtemp,
                    "max_lag": lag,
                    "lag_corr": lag_corr,
                    "te_met_to_dtemp": te.te_met_to_dtemp if te else None,
                    "te_dtemp_to_met": te.te_dtemp_to_met if te else None,
                    "te_p": te.p_met_to_dtemp if te else None,
                }
            )
    return pd.DataFrame(rows)


def extract_all_trajectories(
    data_dir: str | Path,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    progress: bool = True,
) -> pd.DataFrame:
    """Extract isolated-perturbation trajectories for every participant."""
    frames = [
        extract_trajectories(extract_sleep_windows(df, config=config), pid, config=config)
        for pid, df in _participants(data_dir, progress, "trajectories")
    ]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def fit_all_linear_models(
    trajectories: pd.DataFrame, *, config: PipelineConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Fit the 1/2/3-variable OLS models per participant (Eqs. 1–3)."""
    fits = [
        fit_participant_linear_models(group, config=config)
        for _, group in trajectories.groupby("PID")
    ]
    return pd.DataFrame([f.to_record() for f in fits if f is not None])


def fit_all_mechanistic(
    trajectories: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    progress: bool = True,
) -> pd.DataFrame:
    """Fit the two-stage mechanistic model to every trajectory (Eqs. 4–6)."""
    cols = temp_columns(config)
    rows: list[dict] = []
    it = trajectories.itertuples(index=False)
    for row in tqdm(it, total=len(trajectories), desc="mechanistic", disable=not progress):
        rec = row._asdict()
        y = [rec[c] for c in cols]
        fit = fit_trajectory(
            y, rec["met0"], pid=rec["PID"], night=rec["night"], onset=rec["onset"],
            config=config,
        )
        if fit is not None:
            rows.append(fit.to_record())
    return pd.DataFrame(rows)


def run_cohort_analysis(
    mechanistic_fits: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate to participants, then compute effect sizes (Fig. 5) and Table 1.

    Returns ``(effect_sizes, participant_parameters)``.
    """
    participants = aggregate_participant_parameters(
        mechanistic_fits, demographics, config=config
    )
    effects = cohort_effect_sizes(participants, config=config)
    return effects, participants
