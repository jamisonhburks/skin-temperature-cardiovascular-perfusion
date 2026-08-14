"""Tests for the transfer-entropy estimator."""

from __future__ import annotations

import numpy as np

from skin_temp_perfusion.transfer_entropy import transfer_entropy


def test_te_nonnegative_and_zero_for_independent():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 2000)
    y = rng.integers(0, 3, 2000)               # independent of x
    te = transfer_entropy(x, y, k=2)
    assert te >= -1e-9
    assert te < 0.02                           # ≈ 0 up to finite-sample bias


def test_te_detects_direction():
    """A drives B one step later, so TE(A→B) should exceed TE(B→A)."""
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.integers(0, 2, n)
    b = np.zeros(n, dtype=int)
    b[1:] = a[:-1]                             # B_t = A_{t-1}
    b ^= (rng.random(n) < 0.05).astype(int)    # light noise
    assert transfer_entropy(a, b, k=1) > transfer_entropy(b, a, k=1)


def test_te_deterministic():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, 500)
    y = rng.integers(0, 2, 500)
    assert transfer_entropy(x, y, k=2) == transfer_entropy(x, y, k=2)
