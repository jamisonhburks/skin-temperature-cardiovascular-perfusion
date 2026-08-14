"""Small, dependency-light statistics helpers shared across the pipeline.

Kept separate so the effect-size and multiple-comparison logic is unit-testable
in isolation rather than tangled inside plotting code (as it was in the original
exploratory notebooks).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cohen_d",
    "bootstrap_effect_ci",
    "bonferroni",
    "benjamini_hochberg",
]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between samples ``a`` and ``b`` (pooled SD).

    Positive values mean ``a`` has the larger mean. Follows the paper's
    convention of passing (control, condition), so d > 0 ⇒ controls are higher.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var == 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def bootstrap_effect_ci(
    control: np.ndarray,
    condition: np.ndarray,
    *,
    n_iter: int = 1000,
    ci: tuple[float, float] = (2.5, 97.5),
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence band for Cohen's d.

    The control cohort is resampled with replacement to the *condition* cohort's
    size ``n_iter`` times; Cohen's d is recomputed each time. This mirrors the
    paper's size-matched resampling used to draw the effect-size whiskers.

    Returns
    -------
    (median_d, ci_low, ci_high)
    """
    rng = np.random.default_rng() if rng is None else rng
    control = np.asarray(control, dtype=float)
    condition = np.asarray(condition, dtype=float)

    draws = np.empty(n_iter)
    for i in range(n_iter):
        resampled = rng.choice(control, size=len(condition), replace=True)
        draws[i] = cohen_d(resampled, condition)
    lo, hi = np.percentile(draws, ci)
    return float(np.median(draws)), float(lo), float(hi)


def bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses surviving Bonferroni correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    return pvalues < (alpha / len(pvalues))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses surviving the Benjamini–Hochberg FDR procedure.

    Provided alongside Bonferroni so results can be reported under both a
    family-wise-error and a false-discovery-rate criterion.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = pvalues[order] <= thresholds
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))  # largest rank that passes
        mask[order[: cutoff + 1]] = True
    return mask
