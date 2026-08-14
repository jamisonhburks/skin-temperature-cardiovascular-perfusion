"""Incremental linear models of the post-perturbation temperature change.

Three nested ordinary-least-squares models predict ΔT₁ (the temperature change
one minute after an activity perturbation) from progressively more of the
trajectory's state (Methods, Eqs. 1–3):

    ΔT₁ = β₀ + β₁·MET₀                         (Eq. 1)
    ΔT₁ = β₀ + β₁·MET₀ + β₂·T₀                  (Eq. 2)
    ΔT₁ = β₀ + β₁·MET₀ + β₂·T₀ + β₃·T_end       (Eq. 3)

Adding the starting temperature T₀ and the trajectory's final (≈ steady-state)
temperature T_end sharply improves fit (median R² 0.04 → 0.19 → 0.35), but T₀
and T_end are strongly colinear — motivating the state-dependent mechanistic
model in :mod:`skin_temp_perfusion.mechanistic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from .config import DEFAULT_CONFIG, PipelineConfig
from .perturbations import temp_columns

__all__ = ["MODEL_PREDICTORS", "LinearModelFit", "fit_participant_linear_models"]

#: Predictor sets for the 1-, 2- and 3-variable models. ``t_prev`` is T₀ (the
#: pre-perturbation temperature); ``t_end`` is the final trajectory sample.
MODEL_PREDICTORS: dict[int, list[str]] = {
    1: ["met0"],
    2: ["met0", "t_prev"],
    3: ["met0", "t_prev", "t_end"],
}


@dataclass(frozen=True)
class LinearModelFit:
    """Per-participant fit summary for the three nested models.

    ``r2`` maps model size → R². ``coef`` / ``pvalue`` hold the 3-variable
    model's terms (``const``, ``met0``, ``t_prev``, ``t_end``) — the model the
    paper carries forward for cohort comparison.
    """

    pid: str
    n_trajectories: int
    r2: dict[int, float]
    coef: dict[str, float]
    pvalue: dict[str, float]

    def to_record(self) -> dict[str, float | str]:
        """Flatten to a single dict row for pooling across participants."""
        rec: dict[str, float | str] = {"PID": self.pid, "n": self.n_trajectories}
        rec |= {f"r2_{k}var": v for k, v in self.r2.items()}
        rec |= {f"C({name})": v for name, v in self.coef.items()}
        rec |= {f"p({name})": v for name, v in self.pvalue.items()}
        return rec


def fit_participant_linear_models(
    df: pd.DataFrame, *, config: PipelineConfig = DEFAULT_CONFIG
) -> LinearModelFit | None:
    """Fit Eqs. 1–3 for one participant's pooled trajectories.

    Parameters
    ----------
    df
        Trajectory rows for a single participant (see
        :func:`~skin_temp_perfusion.perturbations.extract_trajectories`).
    config
        Reads ``min_trajectories_per_participant``.

    Returns
    -------
    LinearModelFit or None
        ``None`` if the participant has too few trajectories to fit reliably.
    """
    if len(df) < config.min_trajectories_per_participant:
        return None

    data = df.copy()
    data["t_end"] = data[temp_columns(config)[-1]]
    y = data["dtemp1"].to_numpy(dtype=float)

    r2: dict[int, float] = {}
    coef: dict[str, float] = {}
    pval: dict[str, float] = {}
    for size, predictors in MODEL_PREDICTORS.items():
        X = sm.add_constant(data[predictors].to_numpy(dtype=float))
        result = sm.OLS(y, X).fit()
        r2[size] = float(result.rsquared)
        if size == 3:  # keep the full model's coefficients
            names = ["const", *predictors]
            coef = dict(zip(names, result.params))
            pval = dict(zip(names, result.pvalues))

    return LinearModelFit(
        pid=str(df["PID"].iloc[0]),
        n_trajectories=len(df),
        r2=r2,
        coef=coef,
        pvalue=pval,
    )
