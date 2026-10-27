"""Seed-sharded orchestration helpers for synthetic DGP runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from opflow_experiments.evaluation.standard_outputs import metrics_frame_to_standard, write_standard_outputs
from opflow_experiments.pipelines.run_dgp_main import _write_four_question_outputs, load_exposure_config
from opflow_experiments.utils.io import ensure_dir


THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

DUPLICATE_KEY_COLUMNS = ["seed", "question", "scenario", "method", "split", "regime"]


@dataclass(frozen=True)
class ShardSpec:
    index: int
    shard_id: str
    child_run_id: str
    seeds: tuple[int, ...]


def chunk_seeds(seeds: Iterable[int], chunk_size: int) -> list[tuple[int, ...]]:
    if chunk_size <= 0:
        raise ValueError("--seed-chunk-size must be positive.")
    values = [int(seed) for seed in seeds]
    return [tuple(values[idx : idx + chunk_size]) for idx in range(0, len(values), chunk_size)]


def build_shards(run_id: str, seeds: Iterable[int], chunk_size: int) -> list[ShardSpec]:
    shards: list[ShardSpec] = []
    for index, chunk in enumerate(chunk_seeds(seeds, chunk_size)):
        seed_part = f"seed_{chunk[0]}" if len(chunk) == 1 else f"seeds_{chunk[0]}_{chunk[-1]}"
        shard_id = f"shard_{index:03d}_{seed_part}"
        shards.append(ShardSpec(index=index, shard_id=shard_id, child_run_id=f"{run_id}__{shard_id}", seeds=chunk))
    return shards


def free_space_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024.0**3)


def disk_preflight(path: str | Path, min_free_gb: float) -> dict[str, Any]:
    free_gb = free_space_gb(path)
    payload = {
        "path": str(Path(path).resolve()),
        "free_gb": free_gb,
        "min_free_gb": float(min_free_gb),
        "ok": free_gb >= float(min_free_gb),
    }
    if not payload["ok"]:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GiB available, need at least {min_free_gb:.2f} GiB.")
    return payload


def estimate_output_gb(num_shards: int) -> float:
    return max(0.1, 0.25 * float(max(num_shards, 1)))


def default_seeds(mode: str | None, explicit_seeds: list[int] | None, seed: int) -> list[int]:
    if explicit_seeds:
        return [int(value) for value in explicit_seeds]
    if mode == "full":
        return list(range(30))
    if mode == "debug":
        return [0]
    return [int(seed)]


def shard_run_dir(root: Path, shard: ShardSpec) -> Path:
    return root / "shards" / shard.child_run_id


def shard_completion_path(root: Path, shard: ShardSpec) -> Path:
    return shard_run_dir(root, shard) / "completion.json"


def shard_raw_metrics_path(root: Path, shard: ShardSpec) -> Path:
    return shard_run_dir(root, shard) / "artifacts" / "DGP" / "raw_metrics.csv"


def is_shard_complete(root: Path, shard: ShardSpec) -> bool:
    manifest = shard_completion_path(root, shard)
    raw_metrics = shard_raw_metrics_path(root, shard)
    if not manifest.exists() or not raw_metrics.exists() or raw_metrics.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("completed")) and int(payload.get("exit_code", -1)) == 0


def build_child_command(args: argparse.Namespace, root: Path, shard: ShardSpec) -> list[str]:
    child_dir = shard_run_dir(root, shard)
    cmd = [
        sys.executable,
        "experiments/synthetic/run_synthetic_main.py",
        "--config",
        str(args.config),
        "--run-id",
        shard.child_run_id,
        "--output-dir",
        str(root / "shards"),
        "--results-dir",
        str(child_dir / "results"),
        "--figures-dir",
        str(child_dir / "figures"),
        "--tables-dir",
        str(child_dir / "tables"),
        "--reports-dir",
        str(child_dir / "reports"),
        "--skip-paper-outputs",
        "--seeds",
        *[str(seed) for seed in shard.seeds],
    ]
    if args.mode:
        cmd.extend(["--mode", str(args.mode)])
    if args.debug:
        cmd.append("--debug")
    if args.questions:
        cmd.extend(["--questions", *[str(value) for value in args.questions]])
    if args.methods:
        cmd.extend(["--methods", *[str(value) for value in args.methods]])
    for override in args.override or []:
        cmd.extend(["--override", str(override)])
    return cmd


def build_child_env(root: Path, shard: ShardSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_LIMIT_ENV)
    mpl_dir = ensure_dir(root / "cache" / "mpl" / shard.shard_id)
    xdg_dir = ensure_dir(root / "cache" / "xdg" / shard.shard_id)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(xdg_dir)
    return env


def write_shard_manifest(root: Path, shard: ShardSpec, *, command: list[str], exit_code: int, elapsed_sec: float) -> Path:
    run_dir = ensure_dir(shard_run_dir(root, shard))
    raw_metrics = shard_raw_metrics_path(root, shard)
    payload = {
        "completed": int(exit_code) == 0 and raw_metrics.exists() and raw_metrics.stat().st_size > 0,
        "exit_code": int(exit_code),
        "elapsed_sec": float(elapsed_sec),
        "shard_id": shard.shard_id,
        "child_run_id": shard.child_run_id,
        "seeds": list(shard.seeds),
        "command": command,
        "raw_metrics": str(raw_metrics),
    }
    path = run_dir / "completion.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_shard(args: argparse.Namespace, root: Path, shard: ShardSpec) -> dict[str, Any]:
    command = build_child_command(args, root, shard)
    log_dir = ensure_dir(root / "logs")
    log_path = log_dir / f"{shard.shard_id}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(command, cwd=Path.cwd(), env=build_child_env(root, shard), stdout=handle, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    manifest = write_shard_manifest(root, shard, command=command, exit_code=completed.returncode, elapsed_sec=elapsed)
    return {
        "shard_id": shard.shard_id,
        "child_run_id": shard.child_run_id,
        "seeds": list(shard.seeds),
        "exit_code": int(completed.returncode),
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "completion_manifest": str(manifest),
    }


def run_shards(args: argparse.Namespace, root: Path, shards: list[ShardSpec]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending = [shard for shard in shards if not (args.resume and is_shard_complete(root, shard))]
    for shard in shards:
        if args.resume and is_shard_complete(root, shard):
            results.append(
                {
                    "shard_id": shard.shard_id,
                    "child_run_id": shard.child_run_id,
                    "seeds": list(shard.seeds),
                    "exit_code": 0,
                    "elapsed_sec": 0.0,
                    "skipped_by_resume": True,
                    "log_path": str(root / "logs" / f"{shard.shard_id}.log"),
                    "completion_manifest": str(shard_completion_path(root, shard)),
                }
            )
    if args.execution_mode == "sequential":
        for shard in pending:
            print(f"[orchestrator] running {shard.shard_id} seeds={list(shard.seeds)}")
            results.append(run_shard(args, root, shard))
        return results

    import concurrent.futures

    workers = max(1, int(args.parallel_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_shard = {executor.submit(run_shard, args, root, shard): shard for shard in pending}
        for future in concurrent.futures.as_completed(future_to_shard):
            shard = future_to_shard[future]
            print(f"[orchestrator] finished {shard.shard_id}")
            results.append(future.result())
    results.sort(key=lambda row: str(row["shard_id"]))
    return results


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing required shard CSV: {path}")
    return pd.read_csv(path)


def _normalize_metrics_for_merge(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "method" not in out and "model" in out:
        out["method"] = out["model"]
    missing = [column for column in DUPLICATE_KEY_COLUMNS if column not in out]
    if missing:
        raise ValueError(f"Cannot merge metrics; missing duplicate-key columns: {missing}")
    return out


def validate_no_duplicate_metric_rows(metrics: pd.DataFrame) -> None:
    frame = _normalize_metrics_for_merge(metrics)
    dupes = frame.duplicated(DUPLICATE_KEY_COLUMNS, keep=False)
    if dupes.any():
        sample = frame.loc[dupes, DUPLICATE_KEY_COLUMNS].head(10)
        raise ValueError("Duplicate metric rows detected during finalize:\n" + sample.to_string(index=False))


def collect_shard_run_dirs(run_dirs: Iterable[str | Path]) -> list[Path]:
    values = [Path(path) for path in run_dirs]
    if not values:
        raise ValueError("No shard run dirs were provided.")
    return values


def finalize_shards(
    *,
    run_dirs: Iterable[str | Path],
    config_path: str | Path,
    output_root: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Path]:
    shard_dirs = collect_shard_run_dirs(run_dirs)
    output_root = ensure_dir(output_root)
    artifact_dir = ensure_dir(output_root / "artifacts" / "DGP")
    metrics_frames = []
    diag_frames = []
    for run_dir in shard_dirs:
        dgp_dir = Path(run_dir) / "artifacts" / "DGP"
        metrics_frames.append(_read_required_csv(dgp_dir / "raw_metrics.csv"))
        diag_path = dgp_dir / "diagnostics.csv"
        if diag_path.exists() and diag_path.stat().st_size > 0:
            try:
                diag_frames.append(pd.read_csv(diag_path))
            except pd.errors.EmptyDataError:
                pass
    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    diagnostics = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    validate_no_duplicate_metric_rows(metrics)
    metrics.to_csv(artifact_dir / "raw_metrics.csv", index=False)
    diagnostics.to_csv(artifact_dir / "diagnostics.csv", index=False)

    config = load_exposure_config(str(config_path), overrides or [])
    written = _write_four_question_outputs(
        metrics,
        diagnostics,
        config,
        output_root,
        output_root / "results",
        figures_dir=output_root / "figures",
        tables_dir=output_root / "tables",
        reports_dir=output_root / "reports",
        skip_paper_outputs=False,
    )
    standard = metrics_frame_to_standard(
        metrics,
        experiment="DGP",
        regime="DGP",
        config=config,
        gate_diagnostics=pd.DataFrame(),
        notes="merged synthetic shard outputs",
    )
    write_standard_outputs(
        output_root,
        experiment="DGP",
        config_path=str(config_path),
        standard_metrics=standard,
        diagnostics={
            "merged_shards": [str(path) for path in shard_dirs],
            "diagnostic_rows": int(len(diagnostics)),
            "outputs": {key: str(path) for key, path in written.items()},
        },
    )
    manifest_path = output_root / "finalize_manifest.json"
    manifest = {
        "run_dirs": [str(path) for path in shard_dirs],
        "raw_metrics": str(artifact_dir / "raw_metrics.csv"),
        "diagnostics": str(artifact_dir / "diagnostics.csv"),
        "written": {key: str(path) for key, path in written.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "raw_metrics": artifact_dir / "raw_metrics.csv",
        "diagnostics": artifact_dir / "diagnostics.csv",
        "manifest": manifest_path,
        **written,
    }


def orchestrate(args: argparse.Namespace) -> int:
    root = Path(args.output_root) / args.run_id
    if root.exists() and not args.resume:
        raise FileExistsError(f"Run root already exists: {root}. Use --resume or choose a new --run-id.")
    seeds = default_seeds(args.mode, args.seeds, args.seed)
    shards = build_shards(args.run_id, seeds, args.seed_chunk_size)
    preflight = disk_preflight(Path.cwd(), args.min_free_gb)
    preflight["num_shards"] = len(shards)
    preflight["estimated_output_gb"] = estimate_output_gb(len(shards))
    print(
        "[orchestrator] disk free: "
        f"{preflight['free_gb']:.2f} GiB "
        f"(minimum {preflight['min_free_gb']:.2f} GiB, "
        f"estimated output {preflight['estimated_output_gb']:.2f} GiB)"
    )
    ensure_dir(root)
    (root / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "shards.json").write_text(
        json.dumps(
            [
                {"index": shard.index, "shard_id": shard.shard_id, "child_run_id": shard.child_run_id, "seeds": list(shard.seeds)}
                for shard in shards
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    results = run_shards(args, root, shards)
    (root / "shard_status.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [row for row in results if int(row.get("exit_code", 1)) != 0]
    if failures:
        print(f"[orchestrator] {len(failures)} shard(s) failed; preserving artifacts under {root}")
        return 1
    completed_dirs = [shard_run_dir(root, shard) for shard in shards]
    finalize_shards(run_dirs=completed_dirs, config_path=args.config, output_root=root / "final", overrides=args.override)
    print(f"[orchestrator] final outputs: {root / 'final'}")
    return 0
