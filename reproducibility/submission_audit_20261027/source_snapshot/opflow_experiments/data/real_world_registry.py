"""Registry for real-world OD datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opflow_benchmark.dataset import AreaODDataset


@dataclass(frozen=True)
class RealWorldDatasetSpec:
    name: str
    root: Path
    splits_file: Path
    dataset_cls: type


def get_real_world_dataset(config: dict[str, Any]) -> RealWorldDatasetSpec:
    dataset_cfg = config.get("dataset", {})
    name = dataset_cfg.get("name", "large_commuting_od")
    if name != "large_commuting_od":
        raise ValueError(f"Unsupported real-world dataset: {name}")
    root = Path(dataset_cfg.get("data_root", config.get("data_root", "data/opflow_large_commuting_od")))
    return RealWorldDatasetSpec(
        name=name,
        root=root,
        splits_file=root / dataset_cfg.get("splits_file", "splits.json"),
        dataset_cls=AreaODDataset,
    )
