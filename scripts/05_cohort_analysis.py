#!/usr/bin/env python3
"""Stage 5 — cohort effect sizes and per-parameter multilinear models.

Aggregates mechanistic fits to one median row per participant, compares
condition cohorts to controls (Fig. 5), and fits Table 1's models. Writes
``cohort_effect_sizes.parquet`` and ``parameter_models.csv``.
"""

from __future__ import annotations

import pandas as pd

from _common import base_parser, config_from_args, ensure_dir
from skin_temp_perfusion.cohorts import parameter_multilinear_models
from skin_temp_perfusion.io import load_demographics
from skin_temp_perfusion.pipeline import run_cohort_analysis


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = config_from_args(args)

    fits = pd.read_parquet(args.processed_dir / "mechanistic_fits.parquet")
    demographics = load_demographics(args.data_dir / "demographics.parquet")

    effects, participants = run_cohort_analysis(fits, demographics, config=config)
    table1 = parameter_multilinear_models(participants, config=config)

    ensure_dir(args.processed_dir)
    effects.to_parquet(args.processed_dir / "cohort_effect_sizes.parquet")
    participants.to_parquet(args.processed_dir / "participant_parameters.parquet")
    table1.to_csv(args.processed_dir / "parameter_models.csv")

    n_sig = int(effects["significant"].sum())
    print(f"{len(participants)} participants | {len(effects)} comparisons | "
          f"{n_sig} significant (Bonferroni + CI) -> {args.processed_dir}")


if __name__ == "__main__":
    main()
