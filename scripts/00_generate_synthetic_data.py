#!/usr/bin/env python3
"""Stage 0 — generate a synthetic Oura-like dataset.

Writes ``<PID>.parquet`` per participant and ``demographics.parquet`` into the
data directory, so every downstream stage has input without any real data.

Example
-------
    python scripts/00_generate_synthetic_data.py --n-participants 200 --n-nights 20
"""

from __future__ import annotations

from _common import base_parser, ensure_dir
from skin_temp_perfusion.synthetic import generate_dataset


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-participants", type=int, default=600)
    parser.add_argument("--n-nights", type=int, default=14)
    args = parser.parse_args()

    ensure_dir(args.data_dir)
    demo = generate_dataset(
        args.n_participants, args.data_dir, n_nights=args.n_nights, seed=args.seed
    )
    print(f"Wrote {args.n_participants} participants to {args.data_dir}")
    print(f"Condition-free controls: {demo['cd_none'].mean():.0%}")


if __name__ == "__main__":
    main()
