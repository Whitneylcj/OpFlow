"""Shared CLI helpers for canonical experiment entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--override", action="append", default=[])


def bootstrap_repo_path(file_path: str, levels_up: int = 2) -> None:
    """Make repo-root imports work for nested scripts run by file path."""
    root = Path(file_path).resolve()
    for _ in range(levels_up):
        root = root.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
