#!/usr/bin/env python3
"""Regenerate paper-analogous figures from the processed artifacts.

Produces (as .png + .pdf in results/figures/):
    fig3_linear_r2          per-participant R² for the 1/2/3-variable models
    fig4_decomposition      one trajectory decomposed into recovery + residual
    fig4_parameters         distributions of the four dynamical biomarkers
    fig5_cohort_effects     Cohen's d per condition/biomarker with bootstrap CIs

These are qualitative analogues on synthetic data, not the paper's exact panels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _common import base_parser
from skin_temp_perfusion import plotting as viz
from skin_temp_perfusion.config import FIGURES_DIR, PipelineConfig
from skin_temp_perfusion.mechanistic import TrajectoryFit, reconstruct
from skin_temp_perfusion.perturbations import temp_columns


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = PipelineConfig()
    viz.apply_style()
    pdir = args.processed_dir

    trajectories = pd.read_parquet(pdir / "trajectories.parquet")
    linear = pd.read_parquet(pdir / "linear_fits.parquet")
    mech = pd.read_parquet(pdir / "mechanistic_fits.parquet")
    effects = pd.read_parquet(pdir / "cohort_effect_sizes.parquet")
    cols = temp_columns(config)

    # Fig 3 — linear model performance.
    fig, ax = plt.subplots(figsize=(4, 3))
    viz.plot_linear_model_r2(linear, ax=ax)
    viz.save_figure(fig, "fig3_linear_r2")

    # Fig 4G — the headline: mechanistic vs. linear model R².
    good_r2 = mech.loc[mech["residual_r2"] >= config.residual_r2_min, "residual_r2"]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    viz.plot_model_comparison(linear, good_r2, ax=ax)
    viz.save_figure(fig, "fig4g_model_comparison")

    # Fig 4A/B — decompose the trajectory whose fit is nearest the median.
    good = mech[mech["residual_r2"] >= config.residual_r2_min].copy()
    med_row = good.iloc[(good["residual_r2"] - good["residual_r2"].median()).abs().argmin()]
    match = trajectories[(trajectories["PID"] == med_row["PID"])
                         & (trajectories["night"] == med_row["night"])
                         & (trajectories["onset"] == med_row["onset"])].iloc[0]
    fields = {k: med_row[k] for k in TrajectoryFit.__dataclass_fields__ if k != "pid"}
    fit = TrajectoryFit(pid=med_row["PID"], **fields)
    curves = reconstruct(match[cols].to_numpy(dtype=float), fit)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    viz.plot_decomposition(curves, ax=ax)
    viz.save_figure(fig, "fig4_decomposition")

    # Fig 4C–F — biomarker distributions.
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5), layout="constrained")
    viz.plot_parameter_distributions(good, axes=axes)
    viz.save_figure(fig, "fig4_parameters")

    # Fig 5 — cohort effect sizes across age strata.
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharey=True, layout="constrained")
    for ax, group in zip(axes, ("all", "older", "younger")):
        viz.plot_cohort_effect_sizes(effects, age_group=group, ax=ax)
        ax.set_title(ax.get_title(), pad=12)
    viz.save_figure(fig, "fig5_cohort_effects")

    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
