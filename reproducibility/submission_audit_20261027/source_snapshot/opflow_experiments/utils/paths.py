"""Path helpers for reproducible experiment outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from opflow_experiments.utils.io import ensure_dir


def make_run_id(experiment_name: str, timestamp: datetime | None = None) -> str:
    """Build a stable, sortable run id."""
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    clean_name = experiment_name.strip().lower().replace(" ", "_").replace("/", "_")
    return f"{stamp}_{clean_name}"


def create_run_dir(
    experiment_name: str,
    run_id: str | None = None,
    output_root: str | Path = "outputs/runs",
    run_name: str | None = None,
) -> Path:
    """Create and return a run-level output directory."""
    resolved_run_id = run_name or run_id or make_run_id(experiment_name)
    run_dir = ensure_dir(Path(output_root) / resolved_run_id)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "artifacts")
    return run_dir
