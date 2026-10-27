"""Canonical entrypoint for the LargeCommutingOD real-world experiment."""

from __future__ import annotations

from pathlib import Path

import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.core.cli import add_common_run_args
from opflow_experiments.pipelines.real_world_large_commuting_od import run_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_run_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run_from_args(parse_args())
