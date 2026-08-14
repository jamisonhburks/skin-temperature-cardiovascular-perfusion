#!/usr/bin/env python3
"""Stage 1 — global relationships: Pearson correlations, lead/lag, transfer entropy.

Reproduces the per-night statistics behind Fig. 2 and writes them to
``data/processed/global_relationships.parquet``.
"""

from __future__ import annotations

from _common import base_parser, config_from_args, ensure_dir
from skin_temp_perfusion.pipeline import run_global_relationships


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = config_from_args(args)

    result = run_global_relationships(
        args.data_dir, config=config, progress=not args.no_progress
    )
    out = ensure_dir(args.processed_dir) / "global_relationships.parquet"
    result.to_parquet(out)

    med = result.groupby("PID").median(numeric_only=True)
    print(f"Wrote {len(result)} night-level rows -> {out}")
    print(f"Median MET↔ΔT r: {med['r_dtemp'].median():.3f} "
          f"| median max-corr lag: {med['max_lag'].median():.0f} min "
          f"(negative ⇒ MET leads)")


if __name__ == "__main__":
    main()
