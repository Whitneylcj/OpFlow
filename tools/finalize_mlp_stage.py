"""Thin CLI wrapper for final MLP-stage cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.pipelines.finalize_mlp_stage import finalize


if __name__ == "__main__":
    finalize()
