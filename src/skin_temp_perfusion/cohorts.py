"""Cohort construction and comparison of dynamical biomarkers.

Do the fitted parameters (k₁, a, b, k₂) separate participants with cardiac /
cardiovascular conditions from condition-free controls? This module builds the
condition cohorts, aggregates each participant to a median parameter value
(preserving IID comparisons), and quantifies separation with Mann–Whitney U
tests, Cohen's d, and bootstrap confidence bands (Methods, "Comparison of
dynamical model coefficients to reported conditions"; Figs. 5–6, Table 1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

from .config import DEFAULT_CONFIG, PipelineConfig
from .stats_utils import bootstrap_effect_ci, cohen_d

__all__ = [
    "CONDITIONS",
    "AGE_GROUPS",
    "condition_mask",
    "aggregate_participant_parameters",
    "cohort_effect_sizes",
    "parameter_multilinear_models",
]

#: Condition cohorts keyed by short label. Each value is a predicate over the
#: demographics columns. Hypertension (H) and diabetes (D) are split to isolate
#: their individual and comorbid (H+D) effects, exactly as in the paper.
CONDITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "AF": lambda d: d["cd_afib"] == 1,
    "H": lambda d: (d["cd_hypertension"] == 1) & (d["cd_diabetes"] == 0),
    "D": lambda d: (d["cd_diabetes"] == 1) & (d["cd_hypertension"] == 0),
    "H+D": lambda d: (d["cd_hypertension"] == 1) & (d["cd_diabetes"] == 1),
    "CAD": lambda d: d["cd_coronary"] == 1,
}

#: Age strata reported in Fig. 5 (A: all adults, B: older, C: younger).
AGE_GROUPS: dict[str, Callable[[pd.DataFrame, int], pd.Series]] = {
    "all": lambda d, cut: d["age"] >= 18,
    "older": lambda d, cut: d["age"] > cut,
    "younger": lambda d, cut: d["age"] <= cut,
}


def condition_mask(df: pd.DataFrame, condition: str) -> pd.Series:
    """Boolean mask selecting participants in a named condition cohort."""
    return CONDITIONS[condition](df)


def aggregate_participant_parameters(
    fits: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Reduce per-trajectory fits to one median row per participant, with demographics.

    Only trajectories whose Eq. 5 residual fit reaches ``residual_r2_min`` are
    pooled (a quality gate ensuring the parameters describe well-captured
    dynamics). The per-participant medians are then joined to the demographics /
    condition table.
    """
    good = fits[fits["residual_r2"] >= config.residual_r2_min]
    medians = good.groupby("PID")[list(config.dynamical_params)].median()
    return medians.join(demographics, how="inner")


def cohort_effect_sizes(
    participants: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Effect size of each biomarker for every (age × sex × condition) cohort.

    For each stratum, the condition cohort is compared to the size-matched
    condition-free control cohort. A comparison is flagged ``significant`` when
    it survives Bonferroni correction *and* the bootstrap CI excludes zero — the
    solid-marker criterion in Fig. 5.

    Returns
    -------
    pandas.DataFrame
        Tidy rows: ``age_group, sex, condition, param, n_control, n_cohort,
        cohen_d, p_value, ci_low, ci_high, significant``.
    """
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict] = []

    for age_group, age_pred in AGE_GROUPS.items():
        in_age = age_pred(participants, config.age_cutoff)
        for sex in ("male", "female"):
            stratum = participants[in_age & (participants["sex"] == sex)]
            control = stratum[stratum["cd_none"] == 1]
            if len(control) < config.min_cohort_size:
                continue

            for condition in CONDITIONS:
                cohort = stratum[condition_mask(stratum, condition)]
                if len(cohort) < config.min_cohort_size:
                    continue

                for param in config.dynamical_params:
                    c_vals = control[param].dropna().to_numpy()
                    e_vals = cohort[param].dropna().to_numpy()
                    _, p = mannwhitneyu(c_vals, e_vals)
                    d = cohen_d(c_vals, e_vals)
                    _, lo, hi = bootstrap_effect_ci(
                        c_vals, e_vals, n_iter=config.bootstrap_iters, rng=rng
                    )
                    rows.append(
                        {
                            "age_group": age_group,
                            "sex": sex,
                            "condition": condition,
                            "param": param,
                            "n_control": len(c_vals),
                            "n_cohort": len(e_vals),
                            "cohen_d": d,
                            "p_value": p,
                            "ci_low": lo,
                            "ci_high": hi,
                            "significant": bool(
                                p < config.bonferroni_alpha
                                and np.sign(lo) == np.sign(hi)
                            ),
                        }
                    )

    columns = [
        "age_group", "sex", "condition", "param", "n_control", "n_cohort",
        "cohen_d", "p_value", "ci_low", "ci_high", "significant",
    ]
    return pd.DataFrame(rows, columns=columns)


def parameter_multilinear_models(
    participants: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    predictors: tuple[str, ...] = (
        "age",
        "sex",
        "cd_afib",
        "cd_hypertension",
        "cd_diabetes",
        "cd_coronary",
    ),
) -> pd.DataFrame:
    """Fit one OLS model per biomarker to disentangle age/sex/condition effects.

    Reproduces Table 1: each dynamical parameter is regressed on age, sex, and
    condition flags to read off independent β weights. Continuous variables are
    z-scored and ``sex`` is coded male=0/female=1 so coefficients are comparable.

    Returns
    -------
    pandas.DataFrame
        MultiIndex (``param``, ``term``) with ``beta``, ``ci_low``, ``ci_high``,
        ``p_value`` and model ``r2``.
    """
    data = participants.copy()
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    for col in ("age", *config.dynamical_params):
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    records: list[dict] = []
    for param in config.dynamical_params:
        subset = data[[param, *predictors]].dropna()
        X = sm.add_constant(subset[list(predictors)].astype(float))
        result = sm.OLS(subset[param].astype(float), X).fit()
        ci = result.conf_int()
        for term in result.params.index:
            records.append(
                {
                    "param": param,
                    "term": term,
                    "beta": result.params[term],
                    "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1],
                    "p_value": result.pvalues[term],
                    "r2": result.rsquared,
                }
            )
    return pd.DataFrame(records).set_index(["param", "term"])
