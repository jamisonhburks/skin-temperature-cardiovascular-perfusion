"""Selection of isolated activity perturbations and their temperature trajectories.

To study how a *single* burst of activity perturbs temperature, we isolate MET
events that are surrounded by quiet: a non-baseline MET sample with no other
activity in the ``pre_window`` minutes before or ``post_window`` minutes after
(Methods, "Selection of temperature trajectories after activity perturbations";
Fig. 7A). Around each such perturbation we extract a fixed-length temperature
trajectory that the linear and mechanistic models then describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COL_MET,
    COL_TEMP,
    DEFAULT_CONFIG,
    MET_BASELINE,
    MET_EVENT_TOL,
    PipelineConfig,
)

__all__ = [
    "PRE_CONTEXT",
    "temp_columns",
    "find_isolated_perturbations",
    "extract_trajectories",
    "trajectory_matrix",
]

#: Number of pre-perturbation temperature samples retained (for T₋₁ predictor
#: and for plotting the approach to the perturbation, Fig. 4).
PRE_CONTEXT = 5


def temp_columns(config: PipelineConfig = DEFAULT_CONFIG) -> list[str]:
    """Column labels ``T0 … T{L-1}`` for the absolute temperature trajectory."""
    return [f"T{i}" for i in range(config.trajectory_len)]


def find_isolated_perturbations(
    met: np.ndarray, *, config: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Indices of isolated activity perturbations within one night's MET series.

    An index ``i`` qualifies when MET[i] is above baseline and every sample in
    ``[i-pre_window, i-1]`` and ``[i+1, i+post_window]`` is at baseline, and a
    full ``trajectory_len`` window fits after it.

    Parameters
    ----------
    met
        Minute-resolution MET values for a single night.
    config
        Reads ``pre_window``, ``post_window`` and ``trajectory_len``.
    """
    met = np.asarray(met, dtype=float)
    n = met.size
    is_event = (met - MET_BASELINE) > MET_EVENT_TOL
    quiet_after = max(config.post_window, config.trajectory_len)

    out: list[int] = []
    for i in np.flatnonzero(is_event):
        if i < config.pre_window or i + config.trajectory_len > n:
            continue  # not enough context / trajectory room
        if is_event[i - config.pre_window : i].any():
            continue  # activity in the preceding quiet zone
        if is_event[i + 1 : i + quiet_after + 1].any():
            continue  # activity in the following quiet zone
        out.append(int(i))
    return np.asarray(out, dtype=int)


def extract_trajectories(
    windows: list[pd.DataFrame],
    pid: str,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Extract all isolated-perturbation temperature trajectories for a participant.

    Parameters
    ----------
    windows
        Per-night frames from :func:`~skin_temp_perfusion.sleep.extract_sleep_windows`.
    pid
        Participant identifier stamped onto every row.
    config
        Pipeline configuration.

    Returns
    -------
    pandas.DataFrame
        One row per trajectory with columns:

        - ``PID``, ``night`` (index within participant), ``onset`` (sample index)
        - ``met0`` — MET above baseline at the perturbation
        - ``t_prev`` — temperature one minute *before* the perturbation (the
          steady-state / T₀ predictor used by the linear models)
        - ``Tm5 … Tm1`` — pre-perturbation context temperatures (for plotting)
        - ``T0 … T{L-1}`` — absolute temperature trajectory (t=0 is perturbation)
        - ``dtemp1`` — ΔT between the first and second post-perturbation minute
    """
    cols = temp_columns(config)
    rows: list[dict] = []

    for night_idx, night in enumerate(windows):
        temp = night[COL_TEMP].to_numpy(dtype=float)
        met = night[COL_MET].to_numpy(dtype=float)
        for i in find_isolated_perturbations(met, config=config):
            traj = temp[i : i + config.trajectory_len]
            row = {
                "PID": pid,
                "night": night_idx,
                "onset": i,
                "met0": met[i] - MET_BASELINE,
                "t_prev": temp[i - 1],
                "dtemp1": traj[1] - traj[0],
            }
            for j in range(1, PRE_CONTEXT + 1):
                row[f"Tm{j}"] = temp[i - j]
            row.update(dict(zip(cols, traj)))
            rows.append(row)

    return pd.DataFrame(rows)


def trajectory_matrix(
    df: pd.DataFrame, *, config: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Return the ``(n_trajectories, trajectory_len)`` absolute-temperature matrix."""
    return df[temp_columns(config)].to_numpy(dtype=float)
