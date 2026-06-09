"""Configuration helpers for reproducible experiment runs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from opflow_experiments.utils.io import ensure_dir


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path_obj}")
    return config


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of config with explicit top-level overrides applied."""
    resolved = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value
    return resolved


def parse_override_value(raw: str) -> Any:
    """Parse a CLI override value using YAML scalar/list syntax."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def parse_cli_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse key=value CLI overrides into a nested mapping."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty: {item}")
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            if not part:
                raise ValueError(f"Invalid override key: {key}")
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path conflicts with scalar value: {key}")
        cursor[parts[-1]] = parse_override_value(raw_value)
    return overrides


def deep_merge(base: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    resolved = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key] = deep_merge(resolved[key], value)
        else:
            resolved[key] = copy.deepcopy(value)
    return resolved


def load_resolved_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply dotted key=value CLI overrides."""
    return deep_merge(load_config(path), parse_cli_overrides(overrides))


def apply_debug_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a small, fast-running config for smoke/debug experiments."""
    resolved = copy.deepcopy(config)
    resolved["N_o"] = min(int(resolved.get("N_o", 20)), 20)
    resolved["N_d"] = min(int(resolved.get("N_d", 25)), 25)
    resolved["K_candidates"] = min(int(resolved.get("K_candidates", 8)), 8, resolved["N_d"])
    resolved["n_train_envs"] = 1
    resolved["n_val_envs"] = 1
    resolved["n_test_envs"] = 1

    if "baseline" in resolved:
        baseline = resolved["baseline"]
        baseline["hidden_dim"] = min(int(baseline.get("hidden_dim", 8)), 8)
        baseline["epochs"] = min(int(baseline.get("epochs", 2)), 2)
        baseline["batch_size"] = min(int(baseline.get("batch_size", 64)), 64)
        baseline["multi_seed_default"] = [0]

    if "gnn" in resolved:
        gnn = resolved["gnn"]
        gnn["hidden_dim"] = min(int(gnn.get("hidden_dim", 8)), 8)
        gnn["epochs"] = min(int(gnn.get("epochs", 2)), 2)
        gnn["patience"] = min(int(gnn.get("patience", 2)), 2)
        gnn["message_passing_layers"] = min(int(gnn.get("message_passing_layers", 1)), 1)

    if "gamma_sweep" in resolved:
        resolved["gamma_sweep"]["seeds"] = [0]
        resolved["gamma_sweep"]["gamma_share_values"] = [0.0]

    return resolved


def validate_config(config: dict[str, Any], required_sections: tuple[str, ...] = ()) -> None:
    """Validate that required top-level sections are present."""
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {', '.join(missing)}")


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist the resolved config next to experiment artifacts."""
    output_path = ensure_dir(output_dir) / "resolved_config.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return output_path
