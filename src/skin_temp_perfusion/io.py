"""Loading and schema-validation of per-participant wearable data.

The pipeline is agnostic to data *source*: it only requires per-participant
minute-resolution frames with the columns in :data:`REQUIRED_COLUMNS` and a
demographics table. Real Oura exports and the bundled synthetic generator both
satisfy this contract, so swapping one for the other is a one-line change.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .config import COL_AWAKE, COL_MET, COL_PID, COL_TEMP

REQUIRED_COLUMNS: tuple[str, ...] = (COL_TEMP, COL_MET, COL_AWAKE)

__all__ = [
    "REQUIRED_COLUMNS",
    "NON_PARTICIPANT_STEMS",
    "validate_timeseries",
    "load_participant",
    "iter_participants",
    "load_demographics",
]

#: Parquet files in a data directory that are not participant records and must
#: be skipped when iterating (e.g. the demographics table).
NON_PARTICIPANT_STEMS: frozenset[str] = frozenset({"demographics"})


def validate_timeseries(df: pd.DataFrame, *, name: str = "<frame>") -> pd.DataFrame:
    """Assert that ``df`` is a valid minute-resolution participant frame.

    Parameters
    ----------
    df
        Candidate time-series frame.
    name
        Label used in error messages (e.g. a participant id).

    Returns
    -------
    pandas.DataFrame
        The same frame, unchanged, once validated.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")
    return df


def load_participant(path: str | Path) -> pd.DataFrame:
    """Load one participant's parquet time series and validate its schema."""
    path = Path(path)
    df = pd.read_parquet(path)
    return validate_timeseries(df, name=path.stem)


def iter_participants(
    directory: str | Path, *, pattern: str = "*.parquet"
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(participant_id, frame)`` for every participant parquet in ``directory``.

    The participant id is the file stem, which lets the loader work identically
    for real exports (hashed ids) and synthetic files (``P0000`` …).
    Non-participant tables such as ``demographics.parquet`` are skipped (see
    :data:`NON_PARTICIPANT_STEMS`), so pointing this at a data directory "just
    works" without a custom pattern.
    """
    directory = Path(directory)
    for path in sorted(directory.glob(pattern)):
        if path.stem in NON_PARTICIPANT_STEMS:
            continue
        yield path.stem, load_participant(path)


def load_demographics(path: str | Path) -> pd.DataFrame:
    """Load the demographics/survey table indexed by participant id.

    Expected columns include ``age`` (years), ``sex`` (``"male"``/``"female"``)
    and the ``cd_*`` condition flags listed in
    :data:`~skin_temp_perfusion.config.CONDITION_COLUMNS`.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name != COL_PID and COL_PID in df.columns:
        df = df.set_index(COL_PID)
    return df
