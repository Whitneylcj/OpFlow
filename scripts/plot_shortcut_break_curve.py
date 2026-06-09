"""Thin CLI wrapper for the shortcut-break severity curve."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_experiments.pipelines.plot_shortcut_break_curve import main


if __name__ == "__main__":
    main()
