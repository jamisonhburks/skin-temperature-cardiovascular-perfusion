#!/usr/bin/env python3
"""Stage 4 — fit the state-dependent mechanistic model to every trajectory.

Reads ``trajectories.parquet`` and writes per-trajectory (k₁, a, b, k₂) with fit
diagnostics to ``mechanistic_fits.parquet`` (Eqs. 4–6; Figs. 4–5).
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, config_from_args, ensure_dir
from skin_temp_perfusion.pipeline import fit_all_mechanistic


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = config_from_args(args)

    trajectories = pd.read_parquet(args.processed_dir / "trajectories.parquet")
    fits = fit_all_mechanistic(trajectories, config=config, progress=not args.no_progress)
    out = ensure_dir(args.processed_dir) / "mechanistic_fits.parquet"
    fits.to_parquet(out)

    kept = fits["residual_r2"] >= config.residual_r2_min
    print(f"Fit {len(fits)} trajectories -> {out}")
    print(f"Median recovery R²: {fits['recovery_r2'].median():.3f} | "
          f"median residual R²: {fits['residual_r2'].median():.3f} | "
          f"{kept.mean():.0%} pass the residual-R² gate")


if __name__ == "__main__":
    main()
