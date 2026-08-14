"""Segmentation of a participant record into contiguous sleep windows.

A "night" is a maximal run of samples labelled asleep by the device's own
sleep-staging algorithm. Only runs of at least ``min_sleep_samples`` minutes
(5 hours by default) are kept, biasing selection toward genuine overnight sleep
rather than transient naps (Methods, "Selection of participant data").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COL_AWAKE, COL_MET, COL_TEMP, DEFAULT_CONFIG, PipelineConfig

__all__ = ["extract_sleep_windows"]


def extract_sleep_windows(
    df: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    columns: tuple[str, ...] = (COL_TEMP, COL_MET),
) -> list[pd.DataFrame]:
    """Split a participant frame into qualifying asleep windows.

    Contiguous sleep/wake runs are found by a standard "change-point cumsum"
    trick: incrementing a group id whenever the ``is_awake`` label flips. Each
    asleep run is then reduced to the requested signal ``columns``, missing
    samples are dropped, and only sufficiently long windows are returned.

    Parameters
    ----------
    df
        Minute-resolution frame containing at least ``is_awake`` plus ``columns``.
    config
        Pipeline configuration; only ``min_sleep_samples`` is read.
    columns
        Signal columns to retain in each returned window (temperature, MET).

    Returns
    -------
    list of pandas.DataFrame
        One frame per qualifying night, index preserved, reset to a clean
        ``RangeIndex`` so downstream code can use positional access safely.
    """
    labelled = df.dropna(subset=[COL_AWAKE])
    if labelled.empty:
        return []

    # A new group starts each time the boolean sleep label changes.
    run_id = (labelled[COL_AWAKE] != labelled[COL_AWAKE].shift()).cumsum()

    windows: list[pd.DataFrame] = []
    for _, run in labelled.groupby(run_id):
        if bool(run[COL_AWAKE].iloc[0]):
            continue  # awake run
        night = run[list(columns)].dropna()
        if len(night) >= config.min_sleep_samples:
            windows.append(night.reset_index(drop=True))
    return windows


def activity_events(met: np.ndarray | pd.Series, *, tol: float | None = None) -> np.ndarray:
    """Return indices where MET rises above the device baseline (activity).

    Kept here as a small shared primitive used by both the transfer-entropy and
    perturbation-selection stages.
    """
    from .config import MET_BASELINE, MET_EVENT_TOL

    tol = MET_EVENT_TOL if tol is None else tol
    above = np.asarray(met) - MET_BASELINE
    return np.flatnonzero(above > tol)
