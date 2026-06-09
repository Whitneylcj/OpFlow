"""Rebuild upgrade evaluation exports from an existing real-world run."""

from __future__ import annotations

import argparse
from pathlib import Path

from opflow_experiments.utils.config import load_resolved_config
from opflow_experiments.pipelines.realworld_upgrade import write_realworld_upgrade_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    run_dir = run if (run / "resolved_config.yaml").exists() else Path("outputs/runs") / run.name
    config_path = Path(args.config) if args.config else run_dir / "resolved_config.yaml"
    config = load_resolved_config(str(config_path), [])
    out_dir = write_realworld_upgrade_outputs(config, run_dir)
    print(f"Real-world upgrade evaluation directory: {out_dir}")


if __name__ == "__main__":
    main()
