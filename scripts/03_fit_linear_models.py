#!/usr/bin/env python3
"""Stage 3 — fit the 1/2/3-variable OLS models per participant (Eqs. 1–3).

Reads ``trajectories.parquet`` and writes ``linear_fits.parquet`` (Fig. 3).
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, config_from_args, ensure_dir
from skin_temp_perfusion.pipeline import fit_all_linear_models


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = config_from_args(args)

    trajectories = pd.read_parquet(args.processed_dir / "trajectories.parquet")
    fits = fit_all_linear_models(trajectories, config=config)
    out = ensure_dir(args.processed_dir) / "linear_fits.parquet"
    fits.to_parquet(out)

    print(f"Fit {len(fits)} participants -> {out}")
    print("Median R² (paper: 0.04 / 0.19 / 0.35): "
          + " / ".join(f"{fits[f'r2_{k}var'].median():.3f}" for k in (1, 2, 3)))


if __name__ == "__main__":
    main()
