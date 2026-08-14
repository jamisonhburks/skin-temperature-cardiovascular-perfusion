"""Unit tests for the effect-size and multiple-comparison helpers."""

from __future__ import annotations

import numpy as np

from skin_temp_perfusion.stats_utils import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_effect_ci,
    cohen_d,
)


def test_cohen_d_sign_and_zero():
    a = np.array([2.0, 3.0, 2.0, 3.0, 2.5])
    b = np.array([1.0, 2.0, 1.0, 2.0, 1.5])
    assert cohen_d(a, b) > 0            # a has the larger mean
    assert cohen_d(b, a) < 0            # antisymmetric
    assert cohen_d(a, a) == 0.0         # identical samples → 0


def test_cohen_d_known_value():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 10000)
    b = rng.normal(0.0, 1.0, 10000)
    assert abs(cohen_d(a, b) - 1.0) < 0.1   # unit mean-shift, unit SD → d≈1


def test_bootstrap_ci_brackets_median():
    rng = np.random.default_rng(1)
    control = rng.normal(1.0, 1.0, 300)
    condition = rng.normal(0.0, 1.0, 120)
    med, lo, hi = bootstrap_effect_ci(control, condition, n_iter=500, rng=rng)
    assert lo < med < hi
    assert lo > 0                       # clearly separated → CI excludes 0


def test_bonferroni_threshold():
    p = np.array([0.001, 0.02, 0.5, 0.049])
    # alpha/4 = 0.0125 → only the first passes
    assert list(bonferroni(p, alpha=0.05)) == [True, False, False, False]


def test_benjamini_hochberg_more_permissive_than_bonferroni():
    p = np.array([0.001, 0.008, 0.02, 0.9])
    bh = benjamini_hochberg(p, alpha=0.05)
    bonf = bonferroni(p, alpha=0.05)
    assert bh.sum() >= bonf.sum()       # FDR control rejects at least as many
    assert bh[0]                        # the smallest p-value always survives
