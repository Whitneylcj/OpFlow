"""Thin real-world training entrypoint for OpFlow real-world suites."""

from __future__ import annotations

import argparse
from argparse import Namespace

from opflow_experiments.pipelines.real_world_large_commuting_od import run_from_args


DEFAULT_UPGRADE_CONFIG = "configs/real_world/realworld_opflow_upgrade.yaml"
DEFAULT_CONSTRAINED_SI_CONFIG = "configs/real_world/realworld_constrained_si.yaml"
DEFAULT_THREE_SEEDS = [2026, 2027, 2028]


def _resolve_seeds(values: list[int] | None) -> list[int] | None:
    if not values:
        return None
    if len(values) == 1 and values[0] == 3:
        return DEFAULT_THREE_SEEDS
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["upgrade", "constrained_si"], default="upgrade")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = DEFAULT_CONSTRAINED_SI_CONFIG if args.suite == "constrained_si" else DEFAULT_UPGRADE_CONFIG
    run_from_args(
        Namespace(
            config=config_path,
            run_id=args.run_id,
            run_name=args.run_name,
            output_dir=args.output_dir,
            debug=args.mode == "debug",
            seed=args.seed,
            seeds=_resolve_seeds(args.seeds),
            override=args.override,
        )
    )


if __name__ == "__main__":
    main()
