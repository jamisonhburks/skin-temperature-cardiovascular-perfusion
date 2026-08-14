"""Tests for the state-dependent mechanistic model.

The two-stage estimator is a *consistent* (deterministic, monotone) inversion of
the forward model rather than an exact one, so tests assert those invariants —
good fit quality and monotone parameter recovery — not exact value recovery.
"""

from __future__ import annotations

import numpy as np

from skin_temp_perfusion.mechanistic import fit_trajectory, newton_recovery, residual_effect

# `make_trajectory` is a factory fixture defined in conftest.py (injected below).


def test_newton_recovery_endpoints():
    t = np.arange(50, dtype=float)
    y = newton_recovery(t, t0=35.0, t_end=36.0, k1=0.3)
    assert np.isclose(y[0], 35.0)              # starts at T₀
    assert np.isclose(y[-1], 36.0, atol=1e-3)  # asymptotes to T_end


def test_residual_effect_zero_at_origin_and_infinity():
    t = np.array([0.0, 2.0, 100.0])
    r = residual_effect(t, met0=0.4, a=6.0, b=6.0, k2=3.3)
    assert r[0] == 0.0                         # t^b = 0 at t=0
    assert r[1] > 0                            # positive transient
    assert np.isclose(r[2], 0.0, atol=1e-6)    # decays to 0


def test_fit_quality_on_clean_trajectory(make_trajectory):
    y = make_trajectory(k1=0.12, a=6.0, b=6.0, k2=3.3, met0=0.4)
    fit = fit_trajectory(y, 0.4)
    assert fit is not None
    assert fit.recovery_r2 > 0.9
    assert fit.residual_r2 > 0.9


def test_amplitude_recovery_is_monotone(make_trajectory):
    """Larger planted MET-gain a ⇒ larger fitted a (consistency, not exactness)."""
    fits = [fit_trajectory(make_trajectory(0.12, a, 6.0, 3.3, 0.4), 0.4).a
            for a in (2.0, 5.0, 10.0)]
    assert fits[0] < fits[1] < fits[2]


def test_recovery_gate_rejects_flat_noise():
    rng = np.random.default_rng(3)
    # Pure noise has no recovery structure → should fail the R² gate → None.
    from skin_temp_perfusion.config import PipelineConfig
    y = 35.0 + rng.normal(0, 0.5, 20)
    assert fit_trajectory(y, 0.4, config=PipelineConfig(recovery_r2_min=0.8)) is None
