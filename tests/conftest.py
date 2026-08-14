"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from skin_temp_perfusion.config import PipelineConfig
from skin_temp_perfusion.mechanistic import residual_effect


@pytest.fixture
def config() -> PipelineConfig:
    """A fast configuration for tests (few shuffles / bootstraps)."""
    return PipelineConfig(
        te_n_shuffles=10,
        bootstrap_iters=50,
        min_trajectories_per_participant=10,
        min_cohort_size=5,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def make_trajectory():
    """Factory fixture that builds a trajectory from the exact discrete recursion.

    Returned as a fixture (rather than an importable module function) so tests
    get it via dependency injection — no ``from tests.conftest import ...``, which
    is fragile under the bare ``pytest`` command CI uses. Used to check that
    :func:`mechanistic.fit_trajectory` inverts the forward model consistently.
    """

    def _make(
        k1: float, a: float, b: float, k2: float, met0: float,
        t0: float = 35.0, t_end: float = 35.4, length: int = 20,
        noise: float = 0.0, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        tau = np.arange(length, dtype=float)
        resid = residual_effect(tau, met0, a, b, k2)
        y = np.empty(length)
        y[0] = t0
        for i in range(1, length):
            y[i] = y[i - 1] + k1 * (t_end - y[i - 1]) - resid[i]
        if noise and rng is not None:
            y = y + rng.normal(0, noise, length)
        return y

    return _make
