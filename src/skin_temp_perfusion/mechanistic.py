"""State-dependent mechanistic model of post-perturbation temperature dynamics.

The model assumes skin temperature obeys Newton's law of heating — its rate of
change is proportional to its distance from a steady state — and that an
activity burst adds a fast, transient cooling effect on top (Methods, Eqs. 4–6).
Each 20-minute trajectory is decomposed and summarized by four interpretable
"dynamical biomarkers":

    ΔTₜ = k₁·(T_end − Tₜ)                     (Eq. 4)  — state-dependent recovery
    ε_T(t) = a·M₀·t^b·e^(−k₂·t)               (Eq. 5)  — activity residual

    k₁ : recovery decay constant   (thermal inertia + resting vascular tone)
    a  : MET amplitude gain        (sensitivity of the vasoconstrictive response)
    b  : onset steepness           (vasoconstrictive elasticity)
    k₂ : offset steepness          (refractory / vasodilative recovery)

Fitting is two-stage: k₁ is estimated on the late "smooth recovery phase" where
the activity effect is negligible, then used to predict the whole trajectory;
the prediction error is the residual that Eq. 5 describes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .config import DEFAULT_CONFIG, PipelineConfig

__all__ = [
    "newton_recovery",
    "residual_effect",
    "TrajectoryFit",
    "fit_trajectory",
    "reconstruct",
]


def newton_recovery(t: np.ndarray, t0: float, t_end: float, k1: float) -> np.ndarray:
    """Closed-form Newton's-law recovery (Eq. 4), ``T_end − (T_end − T₀)·e^(−k₁t)``."""
    return t_end - (t_end - t0) * np.exp(-k1 * t)


def residual_effect(
    t: np.ndarray, met0: float, a: float, b: float, k2: float
) -> np.ndarray:
    """Time-dependent activity residual (Eq. 5), ``a·M₀·t^b·e^(−k₂t)``."""
    # np.errstate guards the ``0**b`` and overflow warnings at t=0 / large t.
    with np.errstate(invalid="ignore", over="ignore"):
        return a * met0 * np.power(t, b) * np.exp(-k2 * t)


@dataclass(frozen=True)
class TrajectoryFit:
    """Fitted parameters and diagnostics for one temperature trajectory."""

    pid: str
    night: int
    onset: int
    met0: float
    t0: float
    t_end: float
    k1: float
    a: float
    b: float
    k2: float
    recovery_r2: float
    residual_r2: float

    def to_record(self) -> dict:
        """Flatten to a row dict, using the canonical ``PID`` column name."""
        record = asdict(self)
        record["PID"] = record.pop("pid")
        return record


def _fit_recovery(y: np.ndarray, config: PipelineConfig) -> tuple[float, float]:
    """Estimate k₁ on the smooth-recovery tail; return ``(k1, recovery_r2)``.

    Only k₁ is free: T₀ and T_end are pinned to the tail's first and the
    trajectory's last observed samples, matching the paper's procedure of
    fitting recovery where the residual activity effect is negligible.
    """
    start = config.recovery_start_min
    tail = y[start:]
    tau = np.arange(tail.size, dtype=float)
    t_end = y[-1]

    def model(tau_: np.ndarray, k1: float) -> np.ndarray:
        return newton_recovery(tau_, tail[0], t_end, k1)

    popt, _ = curve_fit(model, tau, tail, p0=[0.25], maxfev=10000)
    k1 = float(popt[0])
    return k1, float(r2_score(tail, model(tau, k1)))


def _one_step_residual(y: np.ndarray, k1: float) -> np.ndarray:
    """Residual = state-dependent one-step prediction minus observation.

    Using the fitted k₁, predict each minute from the *observed* previous minute
    (``T̂ₜ = Tₜ₋₁ + k₁·(T_end − Tₜ₋₁)``). What Eq. 4 cannot explain — the
    perturbed-phase dip — remains in the residual for Eq. 5 to model.
    """
    t_end = y[-1]
    pred = y.copy()
    pred[1:] = y[:-1] + k1 * (t_end - y[:-1])
    residual = pred - y
    residual[0] = 0.0
    return residual


def fit_trajectory(
    y: np.ndarray,
    met0: float,
    *,
    pid: str = "",
    night: int = -1,
    onset: int = -1,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> TrajectoryFit | None:
    """Fit the two-stage mechanistic model to one absolute-temperature trajectory.

    Parameters
    ----------
    y
        Absolute temperature samples, minute resolution, ``t = 0…L-1``.
    met0
        MET above baseline at the perturbation (M₀ in Eq. 5).
    pid, night, onset
        Identifiers carried into the result for later pooling.
    config
        Reads ``recovery_start_min`` and ``recovery_r2_min``.

    Returns
    -------
    TrajectoryFit or None
        ``None`` when the recovery fit is too poor (``recovery_r2`` below
        ``recovery_r2_min``, Eq. 6) to trust k₁ — the paper's quality gate.
    """
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size, dtype=float)

    k1, recovery_r2 = _fit_recovery(y, config)
    if recovery_r2 < config.recovery_r2_min:
        return None

    residual = _one_step_residual(y, k1)
    try:
        (a, b, k2), _ = curve_fit(
            lambda t_, a_, b_, k2_: residual_effect(t_, met0, a_, b_, k2_),
            t,
            residual,
            p0=[1.0, 6.0, 3.0],
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        residual_r2 = float(r2_score(residual, residual_effect(t, met0, a, b, k2)))
    except (RuntimeError, ValueError):
        a = b = k2 = residual_r2 = np.nan

    return TrajectoryFit(
        pid=pid,
        night=night,
        onset=onset,
        met0=float(met0),
        t0=float(y[0]),
        t_end=float(y[-1]),
        k1=k1,
        a=float(a),
        b=float(b),
        k2=float(k2),
        recovery_r2=recovery_r2,
        residual_r2=residual_r2,
    )


def reconstruct(y: np.ndarray, fit: TrajectoryFit) -> dict[str, np.ndarray]:
    """Return the fitted curves for a trajectory (for plotting Figs. 4–5).

    Keys: ``observed``, ``recovery`` (one-step Eq. 4 prediction), ``residual``
    (observed error), and ``residual_fit`` (Eq. 5 curve).
    """
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size, dtype=float)
    residual = _one_step_residual(y, fit.k1)  # residual = prediction − observed
    return {
        "observed": y,
        "recovery": y + residual,  # the one-step Eq. 4 prediction, T̂
        "residual": residual,
        "residual_fit": residual_effect(t, fit.met0, fit.a, fit.b, fit.k2),
    }
