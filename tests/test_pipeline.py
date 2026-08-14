"""End-to-end smoke test on a small synthetic dataset."""

from __future__ import annotations

import warnings

from skin_temp_perfusion import pipeline as pl
from skin_temp_perfusion.io import load_demographics
from skin_temp_perfusion.synthetic import generate_dataset


def test_full_pipeline_runs(tmp_path, config):
    warnings.simplefilter("ignore")  # scipy OptimizeWarning on flat tails
    data_dir = tmp_path / "synthetic"
    demo = generate_dataset(12, data_dir, n_nights=6, seed=0)
    assert len(demo) == 12
    assert (data_dir / "demographics.parquet").exists()

    trajectories = pl.extract_all_trajectories(data_dir, config=config, progress=False)
    assert not trajectories.empty
    assert {"PID", "met0", "dtemp1", "T0"} <= set(trajectories.columns)

    linear = pl.fit_all_linear_models(trajectories, config=config)
    # Adding predictors cannot reduce OLS R² (nested models).
    assert (linear["r2_3var"] >= linear["r2_1var"] - 1e-9).all()

    mech = pl.fit_all_mechanistic(trajectories, config=config, progress=False)
    assert {"k1", "a", "b", "k2", "PID"} <= set(mech.columns)

    demographics = load_demographics(data_dir / "demographics.parquet")
    effects, participants = pl.run_cohort_analysis(mech, demographics, config=config)
    assert set(effects.columns) >= {"condition", "param", "cohen_d", "significant"}
    assert len(participants) > 0
