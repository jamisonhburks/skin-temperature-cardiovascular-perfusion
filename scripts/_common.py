"""Shared argument parsing and path helpers for the stage scripts.

Every stage script is a thin, standalone entry point: it parses a couple of
paths, calls into :mod:`skin_temp_perfusion`, and writes a parquet artifact to
``data/processed``. Keeping the glue here keeps each script short.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skin_temp_perfusion.config import PROCESSED_DIR, SYNTHETIC_DIR, PipelineConfig


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by all stages (data dirs + a few config knobs)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path, default=SYNTHETIC_DIR,
                   help="directory of per-participant parquet files")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                   help="directory for intermediate/output artifacts")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return p


def config_from_args(args: argparse.Namespace, **overrides) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from CLI args plus explicit overrides."""
    return PipelineConfig(random_seed=args.seed, **overrides)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
