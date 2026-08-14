"""Tests for sleep-window segmentation and perturbation selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skin_temp_perfusion.config import MET_BASELINE, PipelineConfig
from skin_temp_perfusion.perturbations import extract_trajectories, find_isolated_perturbations
from skin_temp_perfusion.sleep import extract_sleep_windows


def _record(asleep_lengths, awake_len=30):
    """Build a synthetic record with given asleep-run lengths separated by wake."""
    blocks = []
    for n in asleep_lengths:
        blocks.append(pd.DataFrame({"temp_skin": np.full(n, 35.0),
                                    "met": MET_BASELINE, "is_awake": False}))
        blocks.append(pd.DataFrame({"temp_skin": np.full(awake_len, 33.0),
                                    "met": 1.2, "is_awake": True}))
    return pd.concat(blocks, ignore_index=True)


def test_extract_sleep_windows_length_filter():
    cfg = PipelineConfig(min_sleep_samples=300)
    record = _record([320, 100, 500])          # only the 320 and 500 runs qualify
    windows = extract_sleep_windows(record, config=cfg)
    assert [len(w) for w in windows] == [320, 500]


def test_find_isolated_perturbations_respects_quiet_zones():
    cfg = PipelineConfig(pre_window=20, post_window=20, trajectory_len=20)
    met = np.full(200, MET_BASELINE)
    met[60] = MET_BASELINE + 0.3     # isolated: quiet 20 before & after, room ahead
    met[100] = MET_BASELINE + 0.3    # neighbour at 108 → not isolated
    met[108] = MET_BASELINE + 0.2
    met[10] = MET_BASELINE + 0.3     # too close to the start (no pre-window)
    idx = find_isolated_perturbations(met, config=cfg)
    assert list(idx) == [60]


def test_extract_trajectories_shape_and_fields():
    cfg = PipelineConfig(pre_window=20, post_window=20, trajectory_len=20)
    temp = np.linspace(35.0, 35.5, 200)
    met = np.full(200, MET_BASELINE)
    met[60] = MET_BASELINE + 0.4
    night = pd.DataFrame({"temp_skin": temp, "met": met})
    df = extract_trajectories([night], "P0001", config=cfg)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["PID"] == "P0001"
    assert np.isclose(row["met0"], 0.4)
    assert np.isclose(row["T0"], temp[60])            # t=0 is the perturbation
    assert np.isclose(row["t_prev"], temp[59])        # T₋₁ predictor
    assert np.isclose(row["dtemp1"], temp[61] - temp[60])
