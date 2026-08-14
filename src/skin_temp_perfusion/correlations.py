"""Global (time-invariant) relationships between activity and temperature.

These analyses motivate the mechanistic model by showing its ceiling: a linear
correlation captures only weak, non-directional structure, whereas the coupling
is directional (MET leads ΔT) and state-dependent. Corresponds to Fig. 2 and
the Methods section "Calculation of global relationships".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import ccf

from .config import COL_MET, COL_TEMP

__all__ = ["CorrelationResult", "night_correlations", "cross_correlation_lag"]


@dataclass(frozen=True)
class CorrelationResult:
    """Pearson statistics for one night.

    ``r_temp`` / ``p_temp`` describe MET vs. absolute temperature T; ``r_dtemp``
    / ``p_dtemp`` describe MET vs. the minute-to-minute change ΔT. The paper
    reports that ΔT is the better covariate for ~88% of participants.
    """

    r_temp: float
    p_temp: float
    r_dtemp: float
    p_dtemp: float


def night_correlations(night: pd.DataFrame) -> CorrelationResult:
    """Correlate MET with T and with ΔT for a single night.

    ΔT is the discrete gradient of temperature, matching the manuscript's
    "new time series based on the minute-to-minute differences in temperature".
    """
    met = np.asarray(night[COL_MET], dtype=float)
    temp = np.asarray(night[COL_TEMP], dtype=float)
    dtemp = np.gradient(temp)

    r_t, p_t = stats.pearsonr(met, temp)
    r_dt, p_dt = stats.pearsonr(met, dtemp)
    return CorrelationResult(r_t, p_t, r_dt, p_dt)


def cross_correlation_lag(
    night: pd.DataFrame, *, max_lag: int = 5
) -> tuple[int, float]:
    """Lag (minutes) at which |cross-correlation| of MET and ΔT is greatest.

    A **negative** lag means MET *leads* ΔT — activity precedes the temperature
    change, the directional signature reported in Fig. 2E. The cross-correlation
    is evaluated symmetrically over ``[-max_lag, +max_lag]``.

    Returns
    -------
    (lag, corr)
        ``lag`` is the argmax-|corr| offset; ``corr`` is the signed correlation
        at that lag.
    """
    met = np.asarray(night[COL_MET], dtype=float)
    dtemp = np.gradient(np.asarray(night[COL_TEMP], dtype=float))

    # ccf(x, y)[k] correlates x[t] with y[t+k]; evaluate both directions so we
    # can express the result on a signed lag axis centred at zero.
    forward = ccf(dtemp, met, adjusted=False)[: max_lag + 1]   # MET leads (lag<=0)
    backward = ccf(met, dtemp, adjusted=False)[1 : max_lag + 1]  # MET lags (lag>0)
    lags = np.concatenate([np.arange(-max_lag, 1), np.arange(1, max_lag + 1)])
    corrs = np.concatenate([forward[::-1], backward])

    best = int(np.nanargmax(np.abs(corrs)))
    return int(lags[best]), float(corrs[best])
