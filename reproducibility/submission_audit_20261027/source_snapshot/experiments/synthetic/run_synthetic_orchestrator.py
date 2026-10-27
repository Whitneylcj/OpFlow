"""Shard-aware synthetic runner for sequential or parallel execution."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.pipelines.synthetic_parallel import orchestrate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["debug", "full"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--questions", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/parallel_runs")
    parser.add_argument("--execution-mode", choices=["sequential", "parallel"], default="sequential")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--seed-chunk-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    raise SystemExit(orchestrate(parse_args()))


if __name__ == "__main__":
    main()
