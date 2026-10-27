"""Merge seed-sharded synthetic run outputs into isolated final artifacts."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.pipelines.synthetic_parallel import finalize_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = finalize_shards(
        run_dirs=args.run_dirs,
        config_path=args.config,
        output_root=args.output_root,
        overrides=args.override,
    )
    for key, path in sorted(written.items()):
        print(f"{key}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
