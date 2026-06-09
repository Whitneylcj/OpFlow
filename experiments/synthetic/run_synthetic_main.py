"""Canonical synthetic entrypoint for the final DGP."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.pipelines.run_dgp_main import main


if __name__ == "__main__":
    main()
