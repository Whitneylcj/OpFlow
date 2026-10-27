"""Pipeline for the LargeCommutingOD real-world benchmark."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for progress display.
    tqdm = None

from opflow_benchmark.baselines import build_model
from opflow_benchmark.baselines.common import aggregate_area_metrics, set_seed, write_json
from opflow_benchmark.metrics import ood_gap
from opflow_benchmark.real_spatial_splits import expand_split_protocols, generate_spatial_splits, load_split_keys
from opflow_experiments.core.outputs import STANDARD_METRIC_COLUMNS, write_standard_outputs
from opflow_experiments.data.real_world_registry import get_real_world_dataset
from opflow_experiments.pipelines.realworld_upgrade import (
    apply_realworld_upgrade_debug,
    is_realworld_upgrade,
    write_realworld_upgrade_outputs,
)
from opflow_experiments.pipelines.realworld_constrained_si import (
    apply_constrained_si_debug,
    is_constrained_si,
    write_constrained_si_outputs,
)
from opflow_experiments.utils.config import apply_debug_overrides, deep_merge, load_resolved_config, save_resolved_config
from opflow_experiments.utils.paths import create_run_dir


SCORE_METRICS = {"global_cpc", "row_cpc", "rank_corr", "outflow_corr"}
DEFAULT_MODELS = [
    "distance_only",
    "gravity",
    "intervening_opportunity",
    "pair_mlp",
    "scale_aware_mlp",
    "opflow_real",
    "official_gm_p",
    "official_gm_e",
    "official_svr",
    "official_rf",
    "official_gbrt",
    "official_dgm",
    "official_gmel",
    "official_netgan",
    "official_diffodgen",
    "official_wedan",
]
DEFAULT_SPLITS = ["RandSP", "GeoSP", "OppSP"]
OFFICIAL_CPU_FRIENDLY_DEFAULTS = {
    "device": "cpu",
    "official_max_train_pairs": 5000,
    "official_pair_row_chunk_size": "auto",
    "official_pair_chunk_target_mb": 64,
    "official_pair_chunk_max_rows": 128,
}


def _model_override_keys(model_name: str) -> list[str]:
    name = str(model_name)
    keys = ["*"]
    if name == "opflow_real" or name == "opflow_lite" or name.startswith("opflow_"):
        keys.append("opflow")
    if name.startswith("official_"):
        keys.append("official")
    if name in {"pair_mlp", "pair_mlp_sampled", "raw_count_mlp", "scale_aware_mlp"}:
        keys.append("neural_baseline")
    keys.append(name)
    return keys


def training_config_for_model(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Return effective training config after optional model-specific overrides."""
    training = dict(config.get("training", {}))
    overrides = config.get("model_overrides", {}) or {}
    if not isinstance(overrides, dict):
        return training
    applied = []
    for key in _model_override_keys(model_name):
        override = overrides.get(key)
        if not isinstance(override, dict):
            continue
        training = deep_merge(training, override)
        applied.append(key)
    if applied:
        training["_model_override_keys"] = applied
    return training
OFFICIAL_CUDA_DEFAULTS = {
    "device": "cuda",
    "official_pair_row_chunk_size": 32,
    "official_pair_chunk_target_mb": 128,
    "official_pair_chunk_max_rows": 256,
}


def _configure_streaming_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except (OSError, ValueError):
            pass


def _uses_official_models(config: dict[str, Any]) -> bool:
    return any(str(model).startswith("official_") for model in config.get("models", DEFAULT_MODELS))


def apply_official_runtime_auto(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve official baseline runtime defaults from local hardware.

    This is opt-in via training.official_runtime=auto. It keeps the ordinary
    OpFlow real-world config stable while allowing official baseline checks to
    choose a CUDA profile when available and the Mac-safe profile otherwise.
    """
    if not _uses_official_models(config):
        return config
    training = dict(config.get("training", {}))
    if str(training.get("official_runtime", "")).lower() != "auto":
        return config

    has_cuda = bool(torch.cuda.is_available())
    profile = "cuda" if has_cuda else "cpu_mac"
    defaults = OFFICIAL_CUDA_DEFAULTS if has_cuda else OFFICIAL_CPU_FRIENDLY_DEFAULTS
    for key, value in defaults.items():
        if key == "device":
            if training.get("device") in (None, "auto"):
                training["device"] = value
            continue
        training.setdefault(key, value)
    training["official_runtime_resolved"] = profile
    training["official_cuda_available"] = has_cuda
    training["official_cuda_device_count"] = int(torch.cuda.device_count()) if has_cuda else 0

    resolved = dict(config)
    resolved["training"] = training
    return resolved


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _progress_message(
    *,
    status: str,
    completed: int,
    total: int,
    task: dict[str, Any],
    elapsed: float,
    task_seconds: float | None = None,
) -> str:
    percent = 100.0 * completed / max(total, 1)
    avg_seconds = elapsed / completed if completed > 0 else None
    eta = avg_seconds * (total - completed) if avg_seconds is not None else None
    fields = [
        f"[real-world][{status}]",
        f"{completed}/{total}",
        f"{percent:.1f}%",
        f"elapsed={_format_duration(elapsed)}",
        f"eta={_format_duration(eta)}",
        f"avg/task={_format_duration(avg_seconds)}",
    ]
    if task_seconds is not None:
        fields.append(f"last={_format_duration(task_seconds)}")
    fields.extend(
        [
            f"task={task['idx']}",
            f"split={task['split']}",
            f"model={task['model']}",
            f"seed={task['seed']}",
        ]
    )
    return " ".join(fields)


def _emit_progress(progress_path: Path, message: str) -> None:
    print(message, flush=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run(cmd: list[str], cwd: Path | None = None, stdout_path: Path | None = None) -> None:
    if stdout_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)


def _areas(dataset) -> list[dict[str, Any]]:
    return [dataset[idx] for idx in range(len(dataset))]


def resolve_split_name(split_name: str) -> str:
    return str(split_name)


def _ensure_requested_splits(config: dict[str, Any], run_dir: Path | None = None) -> None:
    dataset_cfg = config.get("dataset", {})
    data_root = Path(dataset_cfg.get("data_root", "data/opflow_large_commuting_od"))
    split_path = data_root / dataset_cfg.get("splits_file", "splits.json")
    requested = [str(split) for split in config.get("splits", DEFAULT_SPLITS)]
    existing = load_split_keys(data_root, split_path)
    expanded = expand_split_protocols(requested, existing)
    if split_path.exists() and expanded and set(expanded) <= set(existing) and (data_root / "splits" / "split_summary.csv").exists():
        summary = pd.read_csv(data_root / "splits" / "split_summary.csv")
    else:
        summary = generate_spatial_splits(data_root)
    if run_dir is not None:
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        summary.to_csv(run_dir / "tables" / "split_summary.csv", index=False)


def _expanded_splits(config: dict[str, Any]) -> list[str]:
    dataset_cfg = config.get("dataset", {})
    data_root = Path(dataset_cfg.get("data_root", "data/opflow_large_commuting_od"))
    split_path = data_root / dataset_cfg.get("splits_file", "splits.json")
    available = load_split_keys(data_root, split_path)
    return expand_split_protocols([str(split) for split in config.get("splits", DEFAULT_SPLITS)], available)


def _load_split_metadata(config: dict[str, Any], split_name: str) -> dict[str, dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    data_root = Path(dataset_cfg.get("data_root", "data/opflow_large_commuting_od"))
    path = data_root / "splits" / f"{split_name}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"area_id": str, "STATEFP": str, "COUNTYFP": str})
    return {str(row["area_id"]): row.dropna().to_dict() for _, row in frame.iterrows()}


def _attach_split_metadata(area: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    area["split_metadata"] = dict(metadata)
    for key in ("STATEFP", "COUNTYFP", "census_division", "opp_group", "fold_id", "role"):
        if key in metadata:
            area[key] = metadata[key]
    if "census_division" in metadata:
        area["leo_group"] = metadata["census_division"]


def _metric_mean(metrics: dict[str, Any], name: str, default: Any = "NA") -> Any:
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("mean", default)
    return default if value is None else value


def extract_zip(zip_path: Path, raw_parent: Path) -> Path:
    target = raw_parent / "CommutingODGen-Dataset-main"
    if (target / "data").exists():
        return target
    raw_parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(raw_parent)
    return target


def run_prepare(config: dict[str, Any]) -> None:
    prepare_cfg = config.get("prepare", {})
    dataset_cfg = config.get("dataset", {})
    raw_root = extract_zip(Path(prepare_cfg["zip_path"]), Path(prepare_cfg.get("raw_parent", "data/raw")))
    data_root = dataset_cfg.get("data_root", "data/opflow_large_commuting_od")
    _run(
        [
            sys.executable,
            "scripts/inspect_large_commuting_od.py",
            "--data_root",
            str(raw_root),
            "--out_json",
            "logs/data_inspection.json",
            "--out_summary",
            "logs/data_inspection_summary.txt",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/build_opflow_dataset.py",
            "--raw_root",
            str(raw_root),
            "--out_root",
            data_root,
            "--log_dir",
            "logs",
        ]
    )
    _run([sys.executable, "scripts/make_real_spatial_splits.py", "--data_root", data_root])


def _official_rf_smoke(raw_root: Path, log_path: Path) -> None:
    from sklearn.ensemble import RandomForestRegressor

    def cpc(left: np.ndarray, right: np.ndarray) -> float:
        left = np.maximum(left, 0.0)
        right = np.maximum(right, 0.0)
        denom = float(left.sum() + right.sum())
        return float(2.0 * np.minimum(left, right).sum() / denom) if denom > 0 else float("nan")

    data_root = raw_root / "data"
    areas = sorted(path.name for path in data_root.iterdir() if (path / "od.npy").exists())
    areas = sorted(areas, key=lambda area: np.load(data_root / area / "od.npy", mmap_mode="r").shape[0])[:24]
    train, test = areas[:20], areas[20:24]

    def make_pairs(area_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
        xs, ys, shapes = [], [], []
        for area in area_ids:
            demos = np.load(data_root / area / "demos.npy")
            pois = np.load(data_root / area / "pois.npy")
            feat = np.concatenate([demos, pois], axis=1)
            dis = np.load(data_root / area / "dis.npy")
            n = feat.shape[0]
            pair_x = np.concatenate(
                [
                    np.broadcast_to(feat[:, None, :], (n, n, feat.shape[1])),
                    np.broadcast_to(feat[None, :, :], (n, n, feat.shape[1])),
                    dis[:, :, None],
                ],
                axis=2,
            ).reshape(n * n, -1)
            xs.append(pair_x)
            ys.append(np.load(data_root / area / "od.npy").reshape(-1))
            shapes.append((area, n))
        return np.concatenate(xs), np.concatenate(ys), shapes

    x_train, y_train, _ = make_pairs(train)
    model = RandomForestRegressor(n_estimators=5, min_samples_leaf=2, n_jobs=1, random_state=2026)
    start = time.time()
    model.fit(x_train, y_train)
    lines = ["Official RF-style smoke on smallest 24 areas", f"train_areas={len(train)} test_areas={len(test)}"]
    for area in test:
        x_test, y_test, shapes = make_pairs([area])
        pred = model.predict(x_test)
        n = shapes[0][1]
        y = y_test.reshape(n, n)
        y_hat = pred.reshape(n, n)
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        lines.append(f"{area}: n={n} rmse={rmse:.6f} cpc={cpc(y_hat, y):.6f}")
    lines.append(f"runtime_seconds={time.time() - start:.3f}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_official(config: dict[str, Any]) -> None:
    prepare_cfg = config.get("prepare", {})
    raw_root = extract_zip(Path(prepare_cfg["zip_path"]), Path(prepare_cfg.get("raw_parent", "data/raw")))
    _run([sys.executable, "models/WeDAN/main.py"], cwd=raw_root, stdout_path=Path("logs/official_wedan_run.log"))
    _official_rf_smoke(raw_root, Path("logs/official_rf_smoke.log"))


def _load_split_areas(config: dict[str, Any], split_name: str) -> dict[str, list[dict[str, Any]]]:
    spec = get_real_world_dataset(config)
    limits = config.get("area_limits", {})
    storage_split = resolve_split_name(split_name)
    metadata = _load_split_metadata(config, storage_split)
    output = {}
    for part in ("train", "val", "test"):
        dataset = spec.dataset_cls(
            spec.root,
            split_json=spec.splits_file,
            split_name=storage_split,
            split_part=part,
            max_regions=limits.get("max_regions"),
            max_areas=limits.get(f"max_{part}_areas"),
        )
        output[part] = _areas(dataset)
        for area in output[part]:
            _attach_split_metadata(area, metadata.get(str(area.get("area_id")), {}))
    return output


def _aggregate(area_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        variant: {split: aggregate_area_metrics(rows) for split, rows in split_rows.items()}
        for variant, split_rows in area_metrics.items()
    }


def _gaps(metrics: dict[str, Any]) -> dict[str, Any]:
    gaps: dict[str, Any] = {}
    for variant, split_rows in metrics.items():
        train = split_rows.get("train", {})
        test = split_rows.get("test", {})
        variant_gaps = {}
        for metric, test_summary in test.items():
            train_summary = train.get(metric)
            if not isinstance(train_summary, dict) or not isinstance(test_summary, dict):
                continue
            higher = metric in SCORE_METRICS or metric.startswith("topk") and "coverage" not in metric
            variant_gaps[metric] = ood_gap(train_summary.get("mean"), test_summary.get("mean"), higher_is_better=higher)
        gaps[variant] = variant_gaps
    return gaps


def train_large_commuting_model(
    config: dict[str, Any],
    *,
    split_name: str,
    model_name: str,
    seed: int,
    out_dir: str | Path,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    set_seed(seed)
    train_cfg = training_config_for_model(config, model_name)
    start_load = time.perf_counter()
    split_areas = _load_split_areas(config, split_name)
    load_seconds = time.perf_counter() - start_load
    if not split_areas["train"]:
        raise ValueError("No train areas after filtering; increase area limits")
    if not split_areas["test"]:
        raise ValueError("No test areas after filtering; increase area limits")

    model = build_model(
        model_name,
        lr=train_cfg.get("lr", 1e-3),
        epochs=train_cfg.get("epochs", 50),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        lambda_count=train_cfg.get("lambda_count", 0.1),
        lambda_scale=train_cfg.get("lambda_scale", 1.0),
        lambda_gate=train_cfg.get("lambda_gate", 0.0),
        lambda_inv=train_cfg.get("lambda_inv", 0.1),
        lambda_min=train_cfg.get("lambda_min", 1e-3),
        lambda_entropy=train_cfg.get("lambda_entropy", 1e-3),
        lambda_split=train_cfg.get("lambda_split", 0.05),
        lambda_vrex=train_cfg.get("lambda_vrex", 0.1),
        lambda_op=train_cfg.get("lambda_op", 0.01),
        use_groupdro=train_cfg.get("use_groupdro", True),
        use_rank_exposure=train_cfg.get("use_rank_exposure", True),
        use_competition=train_cfg.get("use_competition", True),
        use_opportunity_field=train_cfg.get("use_opportunity_field", True),
        use_row_center=train_cfg.get("use_row_center", train_cfg.get("use_row_centering", True)),
        detach_count_allocation=train_cfg.get("detach_count_allocation", True),
        use_sit_operator_potential=train_cfg.get("use_sit_operator_potential", True),
        use_monotone_penalty=train_cfg.get("use_monotone_penalty", True),
        use_vrex=train_cfg.get("use_vrex", True),
        alpha=train_cfg.get("alpha", 1.0),
        tau=train_cfg.get("tau", 1.0),
        omega_grid=train_cfg.get("omega_grid", [10000.0, 25000.0, 50000.0]),
        omega_selection=train_cfg.get("omega_selection", "fixed"),
        local_exposure_mode=train_cfg.get("local_exposure_mode", "kernel_sum"),
        scale_residual_weight=train_cfg.get("scale_residual_weight", 0.1),
        lambda_anchor=train_cfg.get("lambda_anchor", 0.005),
        lambda_smooth=train_cfg.get("lambda_smooth", 0.0005),
        lambda_selection_scale=train_cfg.get("lambda_selection_scale", 0.1),
        selection_protocol=train_cfg.get("selection_protocol", "validation"),
        leo_selection_metric=train_cfg.get("leo_selection_metric", "joint_selection_score"),
        leo_max_epochs=train_cfg.get("leo_max_epochs", train_cfg.get("epochs", 20)),
        leo_patience=train_cfg.get("leo_patience", train_cfg.get("patience", 20)),
        leo_max_folds=train_cfg.get("leo_max_folds", 0),
        leo_group_key=train_cfg.get("leo_group_key", "auto"),
        leo_fold_rank_prefix_mode=train_cfg.get("leo_fold_rank_prefix_mode"),
        leo_min_folds_per_epoch=train_cfg.get("leo_min_folds_per_epoch"),
        leo_min_fold_fraction=train_cfg.get("leo_min_fold_fraction", 0.5),
        rank_prefix_mode=train_cfg.get("rank_prefix_mode", "hard"),
        rank_prefix_temperature=train_cfg.get("rank_prefix_temperature", 0.05),
        use_learned_opportunity_reconstruction=train_cfg.get("use_learned_opportunity_reconstruction", True),
        use_origin_gates=train_cfg.get("use_origin_gates", True),
        use_residual_interaction=train_cfg.get("use_residual_interaction", True),
        use_crowding_shift_gate=train_cfg.get("use_crowding_shift_gate", False),
        opp_recon_delta_scale=train_cfg.get("opp_recon_delta_scale", 0.25),
        origin_gate_rho=train_cfg.get("origin_gate_rho", 0.5),
        eta_residual=train_cfg.get("eta_residual", 0.2),
        use_pair_residual_interaction=train_cfg.get("use_pair_residual_interaction", True),
        pair_residual_weight=train_cfg.get("pair_residual_weight", 0.2),
        pair_residual_hidden_dim=train_cfg.get("pair_residual_hidden_dim"),
        checkpoint_metric=train_cfg.get("checkpoint_metric", "alloc_loss"),
        benefit_skip=train_cfg.get("benefit_skip", 1.15),
        deterrence_skip=train_cfg.get("deterrence_skip", -0.35),
        exposure_skip=train_cfg.get("exposure_skip", -0.65),
        competition_skip=train_cfg.get("competition_skip", 0.45),
        crowding_gate_threshold=train_cfg.get("crowding_gate_threshold", 1.5),
        crowding_gate_temperature=train_cfg.get("crowding_gate_temperature", 0.5),
        crowding_fallback_weight=train_cfg.get("crowding_fallback_weight", 1.0),
        io_cache_max_mb=train_cfg.get("io_cache_max_mb", 256.0),
        rank_prefix_row_chunk_size=train_cfg.get("rank_prefix_row_chunk_size"),
        progress_path=str(progress_path) if progress_path is not None else None,
        progress_prefix=f"task split={split_name} model={model_name} seed={seed}",
        hidden_dim=train_cfg.get("official_gmel_hidden_dim", 64) if model_name == "official_gmel" else train_cfg.get("hidden_dim", 64),
        max_pair_mlp_train_areas=train_cfg.get("max_pair_mlp_train_areas"),
        max_raw_count_train_areas=train_cfg.get("max_raw_count_train_areas"),
        max_train_pairs=train_cfg.get("official_max_train_pairs", 300_000),
        pair_row_chunk_size=train_cfg.get("official_pair_row_chunk_size", train_cfg.get("row_chunk_size", 16)),
        pair_chunk_target_mb=train_cfg.get("official_pair_chunk_target_mb", 64.0),
        pair_chunk_max_rows=train_cfg.get("official_pair_chunk_max_rows", 128),
        dgm_hidden_dim=train_cfg.get("official_dgm_hidden_dim", 256),
        dgm_layers=train_cfg.get("official_dgm_layers", 15),
        residual_hidden_dim=train_cfg.get("official_residual_hidden_dim") if model_name.startswith("official_") else train_cfg.get("residual_hidden_dim", 16),
        residual_layers=train_cfg.get("official_residual_layers"),
        netgan_hidden_dim=train_cfg.get("official_netgan_hidden_dim"),
        netgan_layers=train_cfg.get("official_netgan_layers"),
        diffodgen_hidden_dim=train_cfg.get("official_diffodgen_hidden_dim"),
        diffodgen_layers=train_cfg.get("official_diffodgen_layers"),
        wedan_hidden_dim=train_cfg.get("official_wedan_hidden_dim"),
        wedan_layers=train_cfg.get("official_wedan_layers"),
        area_batch_size=train_cfg.get("area_batch_size", 4),
        cache_area_tensors=train_cfg.get("cache_area_tensors", True),
        cache_area_max_mb=train_cfg.get("cache_area_max_mb", 64.0),
        row_chunk_size=train_cfg.get("row_chunk_size", 16),
        device=train_cfg.get("device", "auto"),
        patience=train_cfg.get("patience", 20),
        grad_clip=train_cfg.get("grad_clip", 5.0),
    )
    start_fit = time.perf_counter()
    history = model.fit(split_areas["train"], split_areas["val"])
    fit_seconds = time.perf_counter() - start_fit
    fit_profile = dict(getattr(model, "profile", {}) or {})
    start_eval = time.perf_counter()
    area_metrics = model.evaluate(split_areas, topk=train_cfg.get("topk", 10))
    eval_seconds = time.perf_counter() - start_eval
    profile = dict(getattr(model, "profile", {}) or {})
    for key, value in fit_profile.items():
        if key.startswith("area_cache_"):
            profile[f"fit_{key}"] = value
    start_aggregate = time.perf_counter()
    metrics = _aggregate(area_metrics)
    gaps = _gaps(metrics)
    aggregate_seconds = time.perf_counter() - start_aggregate
    start_checkpoint = time.perf_counter()
    checkpoint = model.checkpoint()
    checkpoint_seconds = time.perf_counter() - start_checkpoint
    train_config = {
        "dataset": config.get("dataset", {}),
        "area_limits": config.get("area_limits", {}),
        "training": train_cfg,
        "split_name": split_name,
        "model": model_name,
        "seed": seed,
        "area_counts": {part: len(areas) for part, areas in split_areas.items()},
        "max_n_regions_used": {
            part: max((int(area["y_raw"].shape[0]) for area in areas), default=0) for part, areas in split_areas.items()
        },
    }
    timings = {
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "eval_seconds": eval_seconds,
        "aggregate_seconds": aggregate_seconds,
        "checkpoint_seconds": checkpoint_seconds,
    }
    profile.update(
        {
            "load_seconds": load_seconds,
            "fit_seconds": fit_seconds,
            "eval_seconds": eval_seconds,
            "aggregate_seconds": aggregate_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "areas_loaded": float(sum(len(areas) for areas in split_areas.values())),
            "train_areas": float(len(split_areas["train"])),
            "val_areas": float(len(split_areas["val"])),
            "test_areas": float(len(split_areas["test"])),
            "max_n_regions": float(max((int(area["y_raw"].shape[0]) for areas in split_areas.values() for area in areas), default=0)),
        }
    )
    if isinstance(checkpoint, dict) and "stats" in checkpoint:
        train_config["normalizer"] = checkpoint["stats"]
    if isinstance(checkpoint, dict) and checkpoint.get("selection_summary") is not None:
        train_config["selection_summary"] = checkpoint.get("selection_summary")
    payload = {
        "config": train_config,
        "history": history,
        "selection_summary": checkpoint.get("selection_summary", {}) if isinstance(checkpoint, dict) else {},
        "metrics": metrics,
        "ood_gap": gaps,
        "area_metrics": area_metrics,
        "timings": timings,
        "profile": profile,
        "primary_variant": f"{model_name}_learned_scale",
    }
    start_write = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    write_json(out_path / "config.json", train_config)
    write_json(out_path / "metrics.json", _json_safe(payload))
    area_risks = pd.DataFrame(_area_risk_rows(area_metrics, split_name=split_name, model_name=model_name, seed=seed))
    area_risks.to_csv(out_path / "area_risks.csv", index=False)
    operator_cols = [
        "split_name",
        "model",
        "variant",
        "seed",
        "part",
        "area_id",
        "alloc_risk_max",
        "alloc_risk_mean",
        "vrex_alloc_var",
        "scale_risk_max",
        "scale_risk_mean",
        "op_monotone_violation_D",
        "op_monotone_violation_S",
        "op_monotone_violation_B",
        "operator_mean_o_tilde",
        "operator_mean_c_tilde",
        "psi_bar_row_mean_abs",
        "opportunity_anchor_loss",
        "opportunity_smooth_loss",
        "crowding_shift_gate",
        "residual_interaction_abs_mean",
        "uses_learned_opportunity_reconstruction",
        "uses_origin_gates",
        "uses_residual_interaction",
        "uses_crowding_shift_gate",
        "rank_prefix_soft",
    ]
    area_risks[[col for col in operator_cols if col in area_risks]].to_csv(out_path / "operator_diagnostics.csv", index=False)
    write_json(
        out_path / "predictions_summary.json",
        _json_safe(
            {
                "model": model_name,
                "split_name": split_name,
                "seed": seed,
                "area_counts": train_config["area_counts"],
                "primary_variant": payload["primary_variant"],
                "metrics": metrics,
            }
        ),
    )
    profile_row = {
        "split_name": split_name,
        "model": model_name,
        "seed": seed,
        **profile,
    }
    pd.DataFrame([profile_row]).to_csv(out_path / "profile_summary.csv", index=False)
    checkpoint["train_config"] = train_config
    checkpoint["timings"] = timings
    checkpoint["profile"] = profile
    torch.save(checkpoint, out_path / "checkpoint.pt")
    timings["write_seconds"] = time.perf_counter() - start_write
    timings["total_seconds"] = time.perf_counter() - start_total
    profile["write_seconds"] = timings["write_seconds"]
    profile["total_seconds"] = timings["total_seconds"]
    pd.DataFrame([{**profile_row, "write_seconds": timings["write_seconds"], "total_seconds": timings["total_seconds"]}]).to_csv(
        out_path / "profile_summary.csv",
        index=False,
    )
    write_json(out_path / "eval.json", _json_safe({"metrics": metrics, "ood_gap": gaps, "area_metrics": area_metrics}))
    (out_path / "train.log").write_text(
        "\n".join(
            [
                f"model={model_name}",
                f"split_name={split_name}",
                f"seed={seed}",
                f"epochs_requested={train_cfg.get('epochs', 50)}",
                f"epochs_completed={len(history.get('train_loss', []))}",
                f"area_counts={train_config['area_counts']}",
                f"load_seconds={timings['load_seconds']:.3f}",
                f"fit_seconds={timings['fit_seconds']:.3f}",
                f"eval_seconds={timings['eval_seconds']:.3f}",
                f"aggregate_seconds={timings['aggregate_seconds']:.3f}",
                f"checkpoint_seconds={timings['checkpoint_seconds']:.3f}",
                f"predict_seconds={profile.get('predict_seconds', 0.0):.3f}",
                f"metric_seconds={profile.get('metric_seconds', 0.0):.3f}",
                f"metric_rows={int(profile.get('metric_rows', 0))}",
                f"write_seconds={timings['write_seconds']:.3f}",
                f"total_seconds={timings['total_seconds']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _payload_to_rows(payload: dict[str, Any], *, split_name: str, model_name: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    metrics = payload["metrics"]
    for variant, split_rows in metrics.items():
        train_rmse = _metric_mean(split_rows.get("train", {}), "rmse")
        test_rmse = _metric_mean(split_rows.get("test", {}), "rmse")
        gaps = payload.get("ood_gap", {}).get(variant, {})
        for split, values in split_rows.items():
            topk = _metric_mean(values, "topk_overlap@5")
            if topk == "NA":
                topk = _metric_mean(values, "topk@5")
            rows.append(
                {
                    "experiment": "real_world_large_commuting_od",
                    "regime": split_name,
                    "method": variant,
                    "seed": seed,
                    "split": split,
                    "n_units": payload["config"]["area_counts"].get(split, "NA"),
                    "noise_level": "NA",
                    "shift_strength": "NA",
                    "shortcut_strength": "NA",
                    "graph_noise": "NA",
                    "ood_rmse": test_rmse,
                    "id_rmse": train_rmse,
                    "mae": _metric_mean(values, "mae"),
                    "log_rmse": _metric_mean(values, "log_rmse"),
                    "global_cpc": _metric_mean(values, "global_cpc"),
                    "row_kl": _metric_mean(values, "row_kl"),
                    "row_cpc": _metric_mean(values, "row_cpc"),
                    "rank_corr": _metric_mean(values, "rank_corr"),
                    "topk_overlap": topk,
                    "ndcg@5": _metric_mean(values, "ndcg@5"),
                    "alloc_risk_max": _metric_mean(values, "alloc_risk_max"),
                    "alloc_risk_mean": _metric_mean(values, "alloc_risk_mean"),
                    "vrex_alloc_var": _metric_mean(values, "vrex_alloc_var"),
                    "scale_risk_max": _metric_mean(values, "scale_risk_max"),
                    "scale_risk_mean": _metric_mean(values, "scale_risk_mean"),
                    "op_monotone_violation_D": _metric_mean(values, "op_monotone_violation_D"),
                    "op_monotone_violation_S": _metric_mean(values, "op_monotone_violation_S"),
                    "op_monotone_violation_B": _metric_mean(values, "op_monotone_violation_B"),
                    "operator_mean_o_tilde": _metric_mean(values, "operator_mean_o_tilde"),
                    "operator_mean_c_tilde": _metric_mean(values, "operator_mean_c_tilde"),
                    "psi_bar_row_mean_abs": _metric_mean(values, "psi_bar_row_mean_abs"),
                    "shortcut_sensitivity": gaps.get("rank_corr", "NA") if split == "test" else "NA",
                    "distance_sensitivity": "NA",
                    "gate_opp_alignment": _metric_mean(values, "gate_opportunity_corr"),
                    "gate_distance_corr": _metric_mean(values, "gate_distance_corr"),
                    "gate_source_corr": _metric_mean(values, "gate_source_corr", _metric_mean(values, "outflow_corr")),
                    "origin_scale_error": _metric_mean(values, "outflow_rmse"),
                    "row_share_sum_error": "NA",
                    "graph_gain": "NA",
                    "notes": f"{model_name};alloc_risk_max={_metric_mean(values, 'alloc_risk_max')}",
                }
            )
    return rows


def _area_risk_rows(area_metrics: dict[str, Any], *, split_name: str, model_name: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for variant, split_rows in area_metrics.items():
        for split, values in split_rows.items():
            for row in values:
                rows.append(
                    {
                        "split_name": split_name,
                        "model": model_name,
                        "variant": variant,
                        "seed": seed,
                        "part": split,
                        "area_id": row.get("area_id"),
                        "n_regions": row.get("n_regions"),
                        "row_kl": row.get("row_kl"),
                        "row_cpc": row.get("row_cpc"),
                        "rank_corr": row.get("rank_corr"),
                        "topk@5": row.get("topk@5", row.get("topk_overlap@5")),
                        "global_cpc": row.get("global_cpc"),
                        "log_rmse": row.get("log_rmse"),
                        "alloc_risk_max": row.get("alloc_risk_max"),
                        "alloc_risk_mean": row.get("alloc_risk_mean"),
                        "vrex_alloc_var": row.get("vrex_alloc_var"),
                        "scale_risk_max": row.get("scale_risk_max"),
                        "scale_risk_mean": row.get("scale_risk_mean"),
                        "op_monotone_violation_D": row.get("op_monotone_violation_D"),
                        "op_monotone_violation_S": row.get("op_monotone_violation_S"),
                        "op_monotone_violation_B": row.get("op_monotone_violation_B"),
                        "operator_mean_o_tilde": row.get("operator_mean_o_tilde"),
                        "operator_mean_c_tilde": row.get("operator_mean_c_tilde"),
                        "psi_bar_row_mean_abs": row.get("psi_bar_row_mean_abs"),
                        "opportunity_anchor_loss": row.get("opportunity_anchor_loss"),
                        "opportunity_smooth_loss": row.get("opportunity_smooth_loss"),
                        "crowding_shift_gate": row.get("crowding_shift_gate"),
                        "residual_interaction_abs_mean": row.get("residual_interaction_abs_mean"),
                        "uses_learned_opportunity_reconstruction": row.get("uses_learned_opportunity_reconstruction"),
                        "uses_origin_gates": row.get("uses_origin_gates"),
                        "uses_residual_interaction": row.get("uses_residual_interaction"),
                        "uses_crowding_shift_gate": row.get("uses_crowding_shift_gate"),
                        "rank_prefix_soft": row.get("rank_prefix_soft"),
                    }
                )
    return rows


def _set_worker_threads(num_threads: Any) -> None:
    if num_threads is None:
        return
    threads = int(num_threads)
    if threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass


SEED_INVARIANT_MODELS = {
    "distance_only",
    "gravity",
    "intervening_opportunity",
    "official_gm_p",
    "official_gm_e",
    "official_svr",
}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _available_memory_bytes() -> int | None:
    try:  # pragma: no cover - psutil is optional.
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(float(out) * 0.60)
    except Exception:
        return None


def _area_size_stats(config: dict[str, Any], splits: list[str]) -> dict[str, float]:
    dataset_cfg = config.get("dataset", {})
    data_root = Path(dataset_cfg.get("data_root", "data/opflow_large_commuting_od"))
    split_path = data_root / dataset_cfg.get("splits_file", "splits.json")
    if not split_path.exists():
        return {"max_n": 0.0, "sum_n2": 0.0, "areas": 0.0}
    try:
        split_spec = json.loads(split_path.read_text(encoding="utf-8"))
    except Exception:
        return {"max_n": 0.0, "sum_n2": 0.0, "areas": 0.0}
    limits = config.get("area_limits", {})
    max_regions = limits.get("max_regions")
    max_regions = None if max_regions in (None, "") else int(max_regions)
    max_n = 0
    sum_n2 = 0
    area_count = 0
    seen: set[str] = set()
    for split in splits:
        storage_split = resolve_split_name(str(split))
        parts = split_spec.get(storage_split, {})
        for part in ("train", "val", "test"):
            max_areas = limits.get(f"max_{part}_areas")
            max_areas = None if max_areas in (None, "") else int(max_areas)
            area_ids = list(parts.get(part, []))
            if max_areas is not None:
                area_ids = area_ids[:max_areas]
            for area_id in area_ids:
                key = str(area_id)
                if key in seen:
                    continue
                path = data_root / key / "y_raw.npy"
                if not path.exists():
                    continue
                n = int(np.load(path, mmap_mode="r").shape[0])
                if max_regions is not None and n > max_regions:
                    continue
                seen.add(key)
                max_n = max(max_n, n)
                sum_n2 += n * n
                area_count += 1
    return {"max_n": float(max_n), "sum_n2": float(sum_n2), "areas": float(area_count)}


def _estimate_worker_memory_bytes(config: dict[str, Any], models: list[str], splits: list[str]) -> tuple[int, dict[str, float]]:
    stats = _area_size_stats(config, splits)
    train_cfg = config.get("training", {})
    max_n = max(int(stats.get("max_n", 0)), int(train_cfg.get("memory_estimate_min_n", 512)))
    dense_bytes = max_n * max_n * np.dtype(np.float32).itemsize
    model_multiplier = 28.0
    if any(model in {"opflow_real", "pair_mlp", "scale_aware_mlp"} or str(model).startswith("opflow_") for model in models):
        model_multiplier = max(model_multiplier, 56.0)
    if any(str(model).startswith("official_") for model in models):
        model_multiplier = max(model_multiplier, 36.0)
    floor_gb = float(train_cfg.get("process_worker_memory_floor_gb", 1.25))
    estimated = int(max(floor_gb * 1024**3, dense_bytes * model_multiplier + 384 * 1024**2))
    stats.update({"max_dense_matrix_mb": dense_bytes / 1024**2, "worker_memory_gb": estimated / 1024**3})
    return estimated, stats


def _resolve_process_workers(config: dict[str, Any], tasks: list[dict[str, Any]], progress_path: Path, splits: list[str] | None = None) -> int:
    train_cfg = config.get("training", {})
    raw = train_cfg.get("process_workers", 1)
    auto_max = max(int(train_cfg.get("process_workers_auto_max", 4)), 1)
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        requested = min(os.cpu_count() or 1, auto_max)
    else:
        requested = max(int(raw), 1)
    requested = min(requested, len(tasks))
    per_worker_bytes, stats = _estimate_worker_memory_bytes(
        config,
        list(config.get("models", DEFAULT_MODELS)),
        list(splits if splits is not None else config.get("splits", DEFAULT_SPLITS)),
    )
    available = _available_memory_bytes()
    reserved_gb = float(train_cfg.get("process_workers_reserved_memory_gb", 2.0))
    fraction = float(train_cfg.get("process_workers_memory_fraction", 0.65))
    memory_limited = requested
    if available is not None:
        usable = max(int(available * fraction - reserved_gb * 1024**3), per_worker_bytes)
        memory_limited = max(1, int(usable // max(per_worker_bytes, 1)))
    workers = max(1, min(requested, memory_limited))
    _emit_progress(
        progress_path,
        "[real-world][memory] "
        f"requested_workers={requested} resolved_workers={workers} "
        f"available_gb={(available / 1024**3) if available else 'NA'} "
        f"per_worker_gb={per_worker_bytes / 1024**3:.2f} "
        f"max_n={int(stats.get('max_n', 0))} max_dense_matrix_mb={stats.get('max_dense_matrix_mb', 0.0):.1f}",
    )
    return workers


def _dedup_enabled(config: dict[str, Any]) -> bool:
    return _truthy(config.get("training", {}).get("deduplicate_seed_invariant_models", True), default=True)


def _dedup_model_set(config: dict[str, Any]) -> set[str]:
    configured = config.get("training", {}).get("seed_invariant_models")
    if configured is None:
        return set(SEED_INVARIANT_MODELS)
    return {str(model) for model in configured}


def _train_matrix_task(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    _set_worker_threads(task.get("torch_num_threads"))
    payload = train_large_commuting_model(
        task["config"],
        split_name=task["split"],
        model_name=task["model"],
        seed=int(task["seed"]),
        out_dir=task["out_dir"],
        progress_path=task.get("progress_path"),
    )
    rows = _payload_to_rows(payload, split_name=task["split"], model_name=task["model"], seed=int(task["seed"]))
    profile = {
        "idx": int(task["idx"]),
        "split_name": task["split"],
        "model": task["model"],
        "seed": int(task["seed"]),
        **payload.get("profile", {}),
    }
    return int(task["idx"]), rows, profile


def _run_train_matrix_sequential(tasks: list[dict[str, Any]], total: int, progress_path: Path) -> list[tuple[int, list[dict[str, Any]], dict[str, Any]]]:
    indexed_rows = []
    run_start = time.perf_counter()
    iterator = (
        tqdm(tasks, total=total, desc="real-world models", unit="task", file=sys.stdout, dynamic_ncols=True, mininterval=1.0)
        if tqdm is not None
        else tasks
    )
    for task in iterator:
        if tqdm is not None:
            iterator.set_postfix(split=task["split"], model=task["model"], seed=task["seed"])
        _emit_progress(
            progress_path,
            _progress_message(status="start", completed=len(indexed_rows), total=total, task=task, elapsed=time.perf_counter() - run_start),
        )
        task_start = time.perf_counter()
        indexed_rows.append(_train_matrix_task(task))
        task_seconds = time.perf_counter() - task_start
        _emit_progress(
            progress_path,
            _progress_message(
                status="done",
                completed=len(indexed_rows),
                total=total,
                task=task,
                elapsed=time.perf_counter() - run_start,
                task_seconds=task_seconds,
            ),
        )
    return indexed_rows


def run_train_matrix(config: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    _configure_streaming_output()
    _ensure_requested_splits(config, run_dir)
    splits = _expanded_splits(config)
    models = config.get("models", DEFAULT_MODELS)
    seeds = config.get("seeds", [config.get("seed", 2026)])
    train_cfg = config.get("training", {})
    out_root = Path(run_dir) / "artifacts" / "real_world_models"
    progress_path = Path(run_dir) / "logs" / "real_world_progress.log"
    if progress_path.exists():
        progress_path.unlink()
    tasks = []
    skipped_tasks = []
    logical_total = max(len(splits) * len(models) * len(seeds), 1)
    dedup = _dedup_enabled(config)
    dedup_models = _dedup_model_set(config)
    primary_seed = int(seeds[0]) if seeds else int(config.get("seed", 2026))
    idx = 0
    for split in splits:
        for model in models:
            for seed in seeds:
                idx += 1
                out_dir = out_root / split / model / f"seed_{seed}"
                task = {
                    "idx": idx,
                    "total": logical_total,
                    "config": config,
                    "split": split,
                    "model": model,
                    "seed": int(seed),
                    "out_dir": str(out_dir),
                    "progress_path": str(progress_path),
                    "torch_num_threads": train_cfg.get("torch_num_threads_per_worker", 1),
                }
                if dedup and str(model) in dedup_models and int(seed) != primary_seed:
                    task["deduplicated_from_seed"] = primary_seed
                    skipped_tasks.append(task)
                else:
                    tasks.append(task)
    execution_total = max(len(tasks), 1)
    workers = _resolve_process_workers(config, tasks, progress_path, splits=splits)
    device = str(train_cfg.get("device", "auto"))
    if workers > 1 and (device == "cuda" or device == "auto" and torch.cuda.is_available()):
        if not bool(train_cfg.get("allow_device_parallel", False)):
            print("[real-world] GPU detected; forcing process_workers=1 unless training.allow_device_parallel=true", flush=True)
            workers = 1
    _emit_progress(
        progress_path,
        f"[real-world][plan] logical_tasks={logical_total} execution_tasks={len(tasks)} dedup_skipped={len(skipped_tasks)} "
        f"workers={workers} splits={len(splits)} models={len(models)} seeds={len(seeds)} progress_log={progress_path}",
    )
    if workers <= 1:
        indexed_rows = _run_train_matrix_sequential(tasks, execution_total, progress_path)
    else:
        indexed_rows = []
        _emit_progress(progress_path, f"[real-world][parallel] process_workers={workers} torch_num_threads_per_worker={tasks[0]['torch_num_threads']}")
        try:
            run_start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for task in tasks:
                    _emit_progress(
                        progress_path,
                        _progress_message(status="submit", completed=0, total=execution_total, task=task, elapsed=time.perf_counter() - run_start),
                    )
                    futures[executor.submit(_train_matrix_task, task)] = task
                completed = 0
                iterator = as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=execution_total, desc="real-world models", unit="task", file=sys.stdout, dynamic_ncols=True, mininterval=1.0)
                for future in iterator:
                    task = futures[future]
                    indexed_rows.append(future.result())
                    completed += 1
                    if tqdm is not None:
                        iterator.set_postfix(split=task["split"], model=task["model"], seed=task["seed"])
                    _emit_progress(
                        progress_path,
                        _progress_message(
                            status="done",
                            completed=completed,
                            total=execution_total,
                            task=task,
                            elapsed=time.perf_counter() - run_start,
                        ),
                    )
        except PermissionError as exc:
            _emit_progress(progress_path, f"[real-world][fallback] process pool unavailable ({exc}); falling back to sequential execution")
            indexed_rows = _run_train_matrix_sequential(tasks, execution_total, progress_path)
    rows = []
    profile_rows = []
    executed_by_idx = {idx: (task_rows, profile) for idx, task_rows, profile in indexed_rows}
    source_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for _, task_rows, profile in indexed_rows:
        source_rows[(str(profile["split_name"]), str(profile["model"]), int(profile["seed"]))] = task_rows
    skipped_by_idx = {int(task["idx"]): task for task in skipped_tasks}
    for logical_idx in range(1, logical_total + 1):
        if logical_idx in executed_by_idx:
            task_rows, profile = executed_by_idx[logical_idx]
            rows.extend(task_rows)
            profile_rows.append(profile)
            continue
        task = skipped_by_idx.get(logical_idx)
        if task is None:
            continue
        source_key = (str(task["split"]), str(task["model"]), int(task["deduplicated_from_seed"]))
        source = source_rows.get(source_key)
        if source is None:
            raise RuntimeError(f"Missing dedup source rows for {source_key}")
        copied = []
        for row in source:
            clone = dict(row)
            clone["seed"] = int(task["seed"])
            copied.append(clone)
        rows.extend(copied)
        profile = {
            "idx": int(task["idx"]),
            "split_name": task["split"],
            "model": task["model"],
            "seed": int(task["seed"]),
            "deduplicated_from_seed": int(task["deduplicated_from_seed"]),
            "deduplicated": 1.0,
            "total_seconds": 0.0,
        }
        profile_rows.append(profile)
        out_dir = Path(task["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "dedup.json", profile)
        pd.DataFrame([profile]).to_csv(out_dir / "profile_summary.csv", index=False)
        (out_dir / "train.log").write_text(
            "\n".join(
                [
                    f"model={task['model']}",
                    f"split_name={task['split']}",
                    f"seed={task['seed']}",
                    f"deduplicated_from_seed={task['deduplicated_from_seed']}",
                    "total_seconds=0.000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    if profile_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(profile_rows).sort_values("idx").to_csv(out_root / "profile_summary.csv", index=False)
    return pd.DataFrame(rows, columns=STANDARD_METRIC_COLUMNS)


def run_analyze(config: dict[str, Any], run_dir: Path) -> None:
    dataset_cfg = config.get("dataset", {})
    out_root = Path(run_dir) / "artifacts" / "real_world_models"
    data_root = dataset_cfg.get("data_root", "data/opflow_large_commuting_od")
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    _run([sys.executable, "scripts/make_tables.py", "--data_root", data_root, "--results_root", str(out_root), "--tables_dir", str(tables_dir)])
    _run([sys.executable, "scripts/make_figures.py", "--data_root", data_root, "--results_root", str(out_root), "--figures_dir", str(figures_dir)])
    _run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--data_root",
            data_root,
            "--results_root",
            str(out_root),
            "--tables_dir",
            str(tables_dir),
            "--reports_dir",
            str(reports_dir),
            "--logs_dir",
            str(logs_dir),
        ]
    )


def _summary_mean_std(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for metric in metric_cols:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if values.empty:
                row[metric] = "NA"
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[metric] = f"{mean:.6g} +/- {std:.6g}"
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def _oracle_scale_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    test = results.loc[results["part"] == "test"].copy()
    learned = test.loc[test["variant"].str.endswith("learned_scale", na=False)].copy()
    oracle = test.loc[test["variant"].str.endswith("oracle_scale", na=False)].copy()
    oracle = oracle.rename(
        columns={
            "global_cpc": "oracle_global_cpc",
            "log_rmse": "oracle_log_rmse",
            "row_kl": "oracle_row_kl",
            "row_cpc": "oracle_row_cpc",
        }
    )
    merged = learned.merge(
        oracle[["split_name", "model", "seed", "oracle_global_cpc", "oracle_log_rmse", "oracle_row_kl", "oracle_row_cpc"]],
        on=["split_name", "model", "seed"],
        how="left",
    )
    return merged.rename(columns={"global_cpc": "learned_global_cpc", "log_rmse": "learned_log_rmse"})[
        [
            "split_name",
            "model",
            "seed",
            "total_flow_ratio",
            "outflow_log_rmse",
            "outflow_corr",
            "learned_global_cpc",
            "oracle_global_cpc",
            "learned_log_rmse",
            "oracle_log_rmse",
            "oracle_row_kl",
            "oracle_row_cpc",
        ]
    ]


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "NA"
    display = frame.fillna("NA").copy()
    headers = [str(col) for col in display.columns]
    rows = []
    for row in display.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def _write_real_world_report(out_dir: Path, main: pd.DataFrame, summary: pd.DataFrame, balance: pd.DataFrame, config: dict[str, Any]) -> None:
    lines = [
        "# Real-World Exposure-Potential OpFlow Report",
        "",
        "## Data and Splits",
        "",
        f"- Dataset root: `{config.get('dataset', {}).get('data_root', 'data/opflow_large_commuting_od')}`.",
        "- Node features are treated as exogenous geographic/opportunity proxies; OD labels are not used as model inputs.",
        "- `RandSP` is a random county-level spatial/environment split; `GeoSP` holds out complete Census divisions; `OppSP` holds out opportunity-concentration tails.",
        "",
        "## Model Settings",
        "",
        "- `opflow_real` uses learned opportunity and impedance operators, rank-prefix exposure, destination local exposure, row-centered additive potentials, and a separate origin-scale branch.",
        "- The local-exposure bandwidth `omega` is fixed or selected without OD labels according to the configuration.",
        "- Model selection uses the configured validation/LEO allocation objective; count reconstruction is scale-isolated with detached allocation by default.",
        "",
        "## Main Results",
        "",
    ]
    if main.empty:
        lines.append("No main result rows were produced.")
    else:
        metric_cols = ["row_kl", "row_cpc", "rank_corr", "topk@5", "log_rmse", "global_cpc"]
        display = main[["split_name", "model", "seed", *[col for col in metric_cols if col in main]]].copy()
        lines.append(_frame_to_markdown(display))
    lines.extend(["", "## Mean +/- Std", ""])
    if summary.empty:
        lines.append("No summary rows were produced.")
    else:
        keep = ["split_name", "model", "row_kl", "row_cpc", "rank_corr", "topk@5", "log_rmse", "global_cpc"]
        lines.append(_frame_to_markdown(summary[[col for col in keep if col in summary]]))
    lines.extend(["", "## Split Balance", ""])
    if balance.empty:
        lines.append("No split balance table was available.")
    else:
        cols = [
            "split_name",
            "part",
            "n_areas",
            "fold_id",
            "train_counties",
            "val_counties",
            "test_counties",
            "train_tract_nodes",
            "val_tract_nodes",
            "test_tract_nodes",
            "stratification",
        ]
        lines.append(_frame_to_markdown(balance[[col for col in cols if col in balance]]))
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This experiment evaluates whether exposure-potential allocation is useful under realistic cross-area shifts. It does not claim semantic opportunity identification or full causal recovery.",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_world_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_real_world_figure(out_dir: Path, main: pd.DataFrame) -> None:
    mpl_cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-opflow"))
    xdg_cache = Path(os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/opflow-cache"))
    mpl_cache.mkdir(parents=True, exist_ok=True)
    (xdg_cache / "fontconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    panels = [
        ("RandSP_seed0", "row_kl", "RandSP Row-KL"),
        ("RandSP_seed0", "rank_corr", "RandSP Rank"),
        ("GeoSP_NewEngland", "row_kl", "GeoSP Row-KL"),
        ("GeoSP_NewEngland", "topk@5", "GeoSP Top-5"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (split_name, metric, title) in zip(axes.reshape(-1), panels):
        if "split_name" not in main:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.set_title(title)
            continue
        frame = main.loc[main["split_name"] == split_name].copy()
        if frame.empty or metric not in frame:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.set_title(title)
            continue
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        grouped = frame.groupby("model", as_index=False)[metric].mean(numeric_only=True).sort_values(metric)
        ax.bar(grouped["model"], grouped[metric], color="#4C78A8")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_real_world_summary.png", dpi=180)
    plt.close(fig)


def write_exposure_potential_outputs(config: dict[str, Any], run_dir: Path) -> None:
    from scripts.make_tables import collect_results

    reporting_cfg = config.get("reporting", {})
    out_dir = Path(reporting_cfg.get("exposure_output_dir", "results/real_opflow_exposure_potential"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config.get("dataset", {}).get("data_root", "data/opflow_large_commuting_od"))
    summary_path = data_root / "splits" / "split_summary.csv"
    balance = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not balance.empty:
        balance.to_csv(out_dir / "split_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "split_summary.csv", index=False)

    results_root = run_dir / "artifacts" / "real_world_models"
    results = collect_results(results_root)
    if results.empty:
        pd.DataFrame().to_csv(out_dir / "main_results.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "main_results_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "oracle_scale_diagnostics.csv", index=False)
        _write_real_world_report(out_dir, pd.DataFrame(), pd.DataFrame(), balance, config)
        _write_real_world_figure(out_dir, pd.DataFrame())
        return

    main = results.loc[(results["part"] == "test") & (results["is_primary"])].copy()
    main.to_csv(out_dir / "main_results.csv", index=False)
    metric_cols = ["row_kl", "row_cpc", "rank_corr", "topk@5", "log_rmse", "global_cpc", "outflow_log_rmse"]
    summary = _summary_mean_std(main, ["split_name", "model"], [col for col in metric_cols if col in main])
    summary.to_csv(out_dir / "main_results_summary.csv", index=False)
    oracle = _oracle_scale_diagnostics(results)
    oracle.to_csv(out_dir / "oracle_scale_diagnostics.csv", index=False)
    _write_real_world_report(out_dir, main, summary, balance, config)
    _write_real_world_figure(out_dir, main)


def apply_real_world_debug(config: dict[str, Any]) -> dict[str, Any]:
    def _limit(value: Any, default: int, cap: int) -> int:
        return min(int(default if value is None else value), cap)

    resolved = dict(config)
    requested = list(resolved.get("splits", DEFAULT_SPLITS))
    resolved["splits"] = ["RandSP_seed0"] if "RandSP" in requested else requested[:1]
    models = list(resolved.get("models", DEFAULT_MODELS))
    resolved["models"] = ["opflow_real"] if "opflow_real" in models else models[:1]
    resolved["seeds"] = list(resolved.get("seeds", [resolved.get("seed", 2026)]))[:1]
    training = dict(resolved.get("training", {}))
    training["epochs"] = min(int(training.get("epochs", 2)), 2)
    training["hidden_dim"] = min(int(training.get("hidden_dim", 8)), 8)
    training["patience"] = min(int(training.get("patience", 2)), 2)
    resolved["training"] = training
    overrides = dict(resolved.get("model_overrides", {}) or {})
    for key, value in list(overrides.items()):
        if not isinstance(value, dict):
            continue
        model_training = dict(value)
        if "epochs" in model_training:
            model_training["epochs"] = min(int(model_training.get("epochs", 2)), 2)
        if "hidden_dim" in model_training:
            model_training["hidden_dim"] = min(int(model_training.get("hidden_dim", 8)), 8)
        if "patience" in model_training:
            model_training["patience"] = min(int(model_training.get("patience", 2)), 2)
        if "leo_max_epochs" in model_training:
            model_training["leo_max_epochs"] = min(int(model_training.get("leo_max_epochs", 2)), 2)
        if "leo_patience" in model_training:
            model_training["leo_patience"] = min(int(model_training.get("leo_patience", 1)), 1)
        overrides[key] = model_training
    resolved["model_overrides"] = overrides
    limits = dict(resolved.get("area_limits", {}))
    limits["max_regions"] = _limit(limits.get("max_regions"), 30, 30)
    limits["max_train_areas"] = _limit(limits.get("max_train_areas"), 4, 4)
    limits["max_val_areas"] = _limit(limits.get("max_val_areas"), 2, 2)
    limits["max_test_areas"] = _limit(limits.get("max_test_areas"), 2, 2)
    resolved["area_limits"] = limits
    if resolved.get("stage", "train") == "all":
        resolved["stage"] = "train"
    return resolved


def run_from_args(args) -> Path:
    config = load_resolved_config(args.config, args.override)
    if args.seed is not None:
        config["seeds"] = [args.seed]
    if args.seeds:
        config["seeds"] = args.seeds
    if args.debug:
        if is_constrained_si(config):
            config = apply_constrained_si_debug(config)
        elif is_realworld_upgrade(config):
            config = apply_realworld_upgrade_debug(config)
        else:
            config = apply_real_world_debug(config)
    config = apply_official_runtime_auto(config)

    run_dir = create_run_dir("real_world_large_commuting_od", run_id=args.run_id, run_name=args.run_name, output_root=args.output_dir)
    save_resolved_config(config, run_dir)

    stage = config.get("stage", "train")
    warnings = []
    standard_metrics = pd.DataFrame(columns=STANDARD_METRIC_COLUMNS)
    if stage in {"all", "prepare"}:
        run_prepare(config)
    if stage in {"all", "official"}:
        run_official(config)
    if stage in {"all", "train"}:
        standard_metrics = run_train_matrix(config, run_dir)
        raw_dir = run_dir / "artifacts" / "real_world_large_commuting_od"
        raw_dir.mkdir(parents=True, exist_ok=True)
        standard_metrics.to_csv(raw_dir / "standard_metrics_raw.csv", index=False)
    if stage in {"all", "analyze"}:
        run_analyze(config, run_dir)
    if standard_metrics.empty and stage not in {"train", "all"}:
        warnings.append(f"Stage `{stage}` did not produce metrics.csv rows.")

    write_standard_outputs(
        run_dir,
        experiment="real_world_large_commuting_od",
        config_path=args.config,
        standard_metrics=standard_metrics,
        diagnostics={
            "dataset": config.get("dataset", {}).get("name", "large_commuting_od"),
            "splits": _expanded_splits(config) if (Path(config.get("dataset", {}).get("data_root", "data/opflow_large_commuting_od")) / config.get("dataset", {}).get("splits_file", "splits.json")).exists() else config.get("splits", DEFAULT_SPLITS),
            "models": config.get("models", DEFAULT_MODELS),
            "stage": stage,
        },
        warnings=warnings,
    )
    if stage in {"all", "train"}:
        write_exposure_potential_outputs(config, run_dir)
        if is_constrained_si(config):
            constrained_dir = write_constrained_si_outputs(config, run_dir)
            print(f"Real-world constrained-SI directory: {constrained_dir}")
        if is_realworld_upgrade(config):
            upgrade_dir = write_realworld_upgrade_outputs(config, run_dir)
            print(f"Real-world upgrade directory: {upgrade_dir}")
    print(f"Run directory: {run_dir}")
    return run_dir
