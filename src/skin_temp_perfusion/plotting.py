"""Reusable plotting helpers for the paper-analogous figures.

Plotting is deliberately separate from computation: every function here takes
already-computed DataFrames/arrays and draws onto an Axes, so figures can be
re-styled or re-composed without rerunning any analysis. Figures are saved as
both ``.png`` and ``.pdf`` (project convention).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

# Consistent, colour-blind-safe roles: temperature (red), activity (blue).
TEMP_COLOR = "#d62728"
MET_COLOR = "#1f77b4"
# Sequential blues for the nested linear models (more variables → darker).
LINEAR_COLORS = {1: "#9ecae1", 2: "#4292c6", 3: "#08519c"}
MECH_COLOR = "#e6550d"  # mechanistic model, contrasting with the linear blues
PARAM_LABELS = {
    "k1": r"$T$ decay $k_1$",
    "a": r"MET gain $a$",
    "b": r"onset $b$",
    "k2": r"offset $k_2$",
}

__all__ = [
    "apply_style",
    "save_figure",
    "plot_mean_trajectory",
    "plot_decomposition",
    "plot_linear_model_r2",
    "plot_model_comparison",
    "plot_parameter_distributions",
    "plot_cohort_effect_sizes",
]


def apply_style() -> None:
    """Apply a clean, publication-oriented Matplotlib/Seaborn style."""
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig: plt.Figure, stem: str, *, directory: Path = FIGURES_DIR) -> None:
    """Save ``fig`` as ``<stem>.png`` and ``<stem>.pdf`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{ext}", bbox_inches="tight")


def plot_mean_trajectory(
    trajectories: pd.DataFrame, temp_cols: list[str], ax: plt.Axes | None = None
) -> plt.Axes:
    """Mean post-perturbation temperature deviation ± IQR (cf. Figs. 3A/4A)."""
    ax = ax or plt.gca()
    curves = trajectories[temp_cols].to_numpy(dtype=float)
    curves = curves - curves[:, [0]]  # align each trajectory to its t=0 value
    t = np.arange(curves.shape[1])
    mean, lo, hi = curves.mean(0), *np.percentile(curves, [25, 75], axis=0)

    ax.axhline(0, ls="--", c="gray", zorder=0)
    ax.plot(t, mean, color=TEMP_COLOR, marker="o", lw=2, label="mean ΔT")
    ax.fill_between(t, lo, hi, color=TEMP_COLOR, alpha=0.25)
    ax.set(xlabel="minutes after activity perturbation", ylabel="ΔT from perturbation (°C)")
    ax.legend()
    return ax


def plot_decomposition(curves: dict[str, np.ndarray], ax: plt.Axes | None = None) -> plt.Axes:
    """Observed vs. recovery prediction vs. activity residual (cf. Fig. 4A/B).

    ``curves`` is the dict returned by
    :func:`skin_temp_perfusion.mechanistic.reconstruct`.
    """
    ax = ax or plt.gca()
    t = np.arange(len(curves["observed"]))
    baseline = curves["observed"][0]  # plot temperature relative to the perturbation
    ax.plot(t, curves["observed"] - baseline, "k-", lw=2.5, label="observed $T$")
    ax.plot(t, curves["recovery"] - baseline, color=TEMP_COLOR, marker="o", ms=3,
            label="Eq. 4 recovery $\\hat{T}$")
    ax.plot(t, curves["residual"], color=MET_COLOR, marker="o", ms=3,
            label="residual $\\varepsilon_T$")
    ax.plot(t, curves["residual_fit"], color=MET_COLOR, ls="--", label="Eq. 5 fit")
    ax.axhline(0, ls="--", c="gray", zorder=0)
    ax.set(xlabel="minutes after perturbation", ylabel="relative temperature (°C)")
    ax.legend(fontsize=7)
    return ax


def plot_linear_model_r2(linear_fits: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Overlaid KDEs of per-participant R² for the 1/2/3-variable models (Fig. 3B).

    Drawn as lines with faint fills (not opaque fills) so no distribution is
    hidden behind another — adding predictors shifts the R² distribution right.
    """
    ax = ax or plt.gca()
    for size in (1, 2, 3):
        sns.kdeplot(linear_fits[f"r2_{size}var"], ax=ax, fill=True, alpha=0.18,
                    lw=2.2, color=LINEAR_COLORS[size], clip=(0, 1), label=f"{size} var")
    ax.set(xlabel="$R^2$", xlim=(-0.02, 1.02), title="linear model performance")
    ax.legend(title="predictors")
    return ax


def plot_model_comparison(
    linear_fits: pd.DataFrame, mechanistic_r2: pd.Series, ax: plt.Axes | None = None
) -> plt.Axes:
    """Linear (1/2/3-var) vs. mechanistic model R² (cf. Fig. 4G).

    The headline result: the state-dependent mechanistic model explains far more
    variance than the linear models, at the same parameter count.
    """
    ax = ax or plt.gca()
    for size in (1, 2, 3):
        sns.kdeplot(linear_fits[f"r2_{size}var"], ax=ax, fill=True, alpha=0.15,
                    lw=2, color=LINEAR_COLORS[size], clip=(0, 1), label=f"linear {size}-var")
    sns.kdeplot(mechanistic_r2, ax=ax, fill=True, alpha=0.25, lw=2.5,
                color=MECH_COLOR, clip=(0, 1), label="mechanistic")
    ax.set(xlabel="model $R^2$", xlim=(-0.02, 1.02), title="model performance")
    ax.legend()
    return ax


def plot_parameter_distributions(fits: pd.DataFrame, axes: np.ndarray | None = None) -> np.ndarray:
    """KDEs of the four dynamical biomarkers (cf. Fig. 4C–F)."""
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(11, 2.5))
    for ax, (param, label) in zip(np.ravel(axes), PARAM_LABELS.items()):
        lo, hi = fits[param].quantile([0.005, 0.995])
        sns.kdeplot(fits[param], ax=ax, fill=True, clip=(lo, hi),
                    color=TEMP_COLOR if param == "k1" else MET_COLOR)
        ax.set_xlabel(label)
    return axes


def plot_cohort_effect_sizes(
    effects: pd.DataFrame, *, age_group: str = "all", ax: plt.Axes | None = None
) -> plt.Axes:
    """Cohen's d per condition and biomarker with bootstrap whiskers (Fig. 5).

    A clean rewrite of the paper's dense effect-size panel: solid markers are
    significant (Bonferroni + CI excludes 0); hollow markers are not. Male and
    female cohorts are dodged side by side within each condition.
    """
    ax = ax or plt.gca()
    data = effects[effects["age_group"] == age_group]
    params = list(PARAM_LABELS)
    conditions = ["AF", "H", "D", "H+D", "CAD"]
    sex_dodge = {"male": -0.15, "female": 0.15}
    sex_color = {"male": MET_COLOR, "female": "#9467bd"}

    for pi, param in enumerate(params):
        for ci, cond in enumerate(conditions):
            x0 = pi * (len(conditions) + 1) + ci
            for sex, dx in sex_dodge.items():
                row = data[(data["param"] == param) & (data["condition"] == cond)
                           & (data["sex"] == sex)]
                if row.empty:
                    continue
                r = row.iloc[0]
                x = x0 + dx
                ax.plot([x, x], [r["ci_low"], r["ci_high"]], c="k", lw=1.5, zorder=1)
                ax.scatter(
                    x, r["cohen_d"], s=45, zorder=2, color=sex_color[sex],
                    edgecolor="k",
                    facecolor=sex_color[sex] if r["significant"] else "none",
                )
        # Anchor group labels just inside the top of the axes (x in data coords,
        # y in axes-fraction) so they never collide with the subplot title.
        ax.text(pi * (len(conditions) + 1) + 2, 0.96, PARAM_LABELS[param],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)

    for level in (0, 0.2, 0.5, 0.8, -0.2, -0.5, -0.8):
        ax.axhline(level, ls="--", c="gray", alpha=0.3 if level else 1, zorder=0)
    ticks = [pi * (len(conditions) + 1) + ci
             for pi in range(len(params)) for ci in range(len(conditions))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(conditions * len(params), fontsize=8)
    ax.set_ylabel("Cohen's $d$ (control − condition)")
    ax.set_title(f"cohort separability — {age_group} adults")
    return ax
