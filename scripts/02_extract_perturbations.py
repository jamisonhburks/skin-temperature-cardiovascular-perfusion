#!/usr/bin/env python3
"""Stage 2 — extract isolated-perturbation temperature trajectories.

Writes one row per trajectory to ``data/processed/trajectories.parquet`` (input
to both the linear and mechanistic models).
"""

from __future__ import annotations

from _common import base_parser, config_from_args, ensure_dir
from skin_temp_perfusion.pipeline import extract_all_trajectories


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = config_from_args(args)

    trajectories = extract_all_trajectories(
        args.data_dir, config=config, progress=not args.no_progress
    )
    out = ensure_dir(args.processed_dir) / "trajectories.parquet"
    trajectories.to_parquet(out)
    print(f"Wrote {len(trajectories)} trajectories from "
          f"{trajectories['PID'].nunique()} participants -> {out}")


if __name__ == "__main__":
    main()
