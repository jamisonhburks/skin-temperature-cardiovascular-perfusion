"""Transfer entropy between activity and temperature change.

Transfer entropy (TE) is an asymmetric, model-free measure of directed
information flow (Schreiber, 2000): TE(X→Y) quantifies how much knowing the
past of *source* X reduces uncertainty about the next value of *target* Y,
*beyond* what Y's own past already tells us. Because it is asymmetric and
nonlinear, TE(MET→ΔT) ≠ TE(ΔT→MET), which is exactly the directional evidence
the paper needs (Fig. 2F).

A self-contained plug-in estimator (in nats) is the default so the repository
has no hard dependency; installing the optional ``pyinform`` package enables an
exact-parity path via ``backend="pyinform"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COL_TEMP, DEFAULT_CONFIG, MET_BASELINE, MET_EVENT_TOL, PipelineConfig

__all__ = ["transfer_entropy", "TransferEntropyResult", "night_transfer_entropy"]


def _counts(keys: np.ndarray) -> dict[int, int]:
    """Map each distinct integer key to its count."""
    values, counts = np.unique(keys, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def _te_plugin(source: np.ndarray, target: np.ndarray, k: int) -> float:
    """Plug-in (maximum-likelihood) transfer entropy TE(source → target), nats.

    Uses target history length ``k`` and source lag 1 (the Schreiber form):

        TE = Σ p(yₜ₊₁, yₜᵏ, xₜ) · ln [ p(yₜ₊₁ | yₜᵏ, xₜ) / p(yₜ₊₁ | yₜᵏ) ]

    Vectorized: each length-``k`` target history is mixed-radix encoded to a
    single integer, joint states are counted with :func:`numpy.unique`, and the
    empirical TE is summed over the (few) occupied states.
    """
    source = np.asarray(source)
    target = np.asarray(target)
    n = target.size
    if n <= k:
        return 0.0

    A = int(target.max()) + 1           # target alphabet size
    S = int(source.max()) + 1           # source alphabet size
    H = A**k                            # number of distinct histories

    idx = np.arange(k, n)
    y_hist = np.zeros(idx.size, dtype=np.int64)
    for j in range(k):                  # encode history base-A (k is tiny, e.g. 2)
        y_hist = y_hist * A + target[idx - k + j]
    y_next = target[idx].astype(np.int64)
    x_now = source[idx - 1].astype(np.int64)
    total = idx.size

    c_full = _counts((y_next * H + y_hist) * S + x_now)  # (yₜ₊₁, yₜᵏ, xₜ)
    c_yx = _counts(y_hist * S + x_now)                   # (yₜᵏ, xₜ)
    c_ny = _counts(y_next * H + y_hist)                  # (yₜ₊₁, yₜᵏ)
    c_y = _counts(y_hist)                                # (yₜᵏ)

    te = 0.0
    for key, count in c_full.items():
        x_k = key % S
        rest = key // S                 # = y_next * H + y_hist
        yh = rest % H
        yn = rest // H
        cond_num = count / c_yx[yh * S + x_k]            # p(yₜ₊₁ | yₜᵏ, xₜ)
        cond_den = c_ny[yn * H + yh] / c_y[yh]           # p(yₜ₊₁ | yₜᵏ)
        te += (count / total) * np.log(cond_num / cond_den)
    return float(te)


def transfer_entropy(
    source: np.ndarray, target: np.ndarray, *, k: int = 2, backend: str = "plugin"
) -> float:
    """Transfer entropy TE(source → target) for discrete sequences, in nats.

    Parameters
    ----------
    source, target
        Integer-coded sequences of equal length.
    k
        History length for the target.
    backend
        ``"plugin"`` (default, no dependencies) or ``"pyinform"`` (exact parity
        with the paper; requires the optional ``pyinform`` package). pyinform
        returns bits, which we convert to nats.
    """
    if backend == "pyinform":
        from pyinform.transferentropy import transfer_entropy as _pyte

        bits = _pyte(np.asarray(source), np.asarray(target), k)
        return float(bits) * np.log(2.0)  # bits → nats
    return _te_plugin(source, target, k)


def _digitize_met(met: np.ndarray) -> np.ndarray:
    """Binarize MET: 1 during an activity event, 0 at baseline."""
    return (np.asarray(met) - MET_BASELINE > MET_EVENT_TOL).astype(int)


def _digitize_dtemp(dtemp: np.ndarray, bins: int) -> np.ndarray:
    """Quantile-bin ΔT into ``bins`` equal-mass levels (labels 0…bins-1)."""
    edges = np.nanquantile(dtemp, np.linspace(0, 1, bins + 1))
    return np.clip(np.digitize(dtemp, edges[1:-1]), 0, bins - 1)


@dataclass(frozen=True)
class TransferEntropyResult:
    """Directed TE and shuffle-based significance for one night."""

    te_met_to_dtemp: float
    te_dtemp_to_met: float
    p_met_to_dtemp: float
    p_dtemp_to_met: float
    n_events: int


def _shuffle_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided p-value of ``observed`` against a Gaussian fit to ``null``."""
    mu, sigma = np.nanmean(null), np.nanstd(null)
    if sigma == 0:
        return float(observed <= mu)
    from scipy.stats import norm

    return float(1.0 - norm.cdf((observed - mu) / sigma))


def night_transfer_entropy(
    night: pd.DataFrame,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    rng: np.random.Generator | None = None,
    backend: str = "plugin",
) -> TransferEntropyResult | None:
    """Compute directional TE between MET and ΔT for one night, with a null.

    MET is binarized (event / no event) and ΔT is quantile-binned. Significance
    is assessed by shuffling the source series ``config.te_n_shuffles`` times and
    z-scoring the observed TE against the resulting null distribution.

    Returns ``None`` when the night's activity-event count falls outside
    ``[te_min_events, te_max_events)`` (too few events to estimate, or dense
    enough that events are no longer isolated).
    """
    rng = np.random.default_rng(config.random_seed) if rng is None else rng

    met = _digitize_met(night["met"].to_numpy())
    n_events = int(met.sum())
    if not (config.te_min_events <= n_events < config.te_max_events):
        return None

    dtemp = np.gradient(night[COL_TEMP].to_numpy(dtype=float))
    dtemp_d = _digitize_dtemp(dtemp, config.te_dtemp_bins)
    k = config.te_history

    te_md = transfer_entropy(met, dtemp_d, k=k, backend=backend)
    te_dm = transfer_entropy(dtemp_d, met, k=k, backend=backend)

    null_md = np.empty(config.te_n_shuffles)
    null_dm = np.empty(config.te_n_shuffles)
    for i in range(config.te_n_shuffles):
        shuffled = rng.permutation(met)
        null_md[i] = transfer_entropy(shuffled, dtemp_d, k=k, backend=backend)
        null_dm[i] = transfer_entropy(dtemp_d, shuffled, k=k, backend=backend)

    return TransferEntropyResult(
        te_met_to_dtemp=te_md,
        te_dtemp_to_met=te_dm,
        p_met_to_dtemp=_shuffle_pvalue(te_md, null_md),
        p_dtemp_to_met=_shuffle_pvalue(te_dm, null_dm),
        n_events=n_events,
    )
