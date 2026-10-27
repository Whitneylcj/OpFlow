"""Area-level dataset for the OpFlow LargeCommutingOD benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class AreaODDataset(Dataset):
    """Load one variable-size OD area per item."""

    def __init__(
        self,
        root: str | Path,
        split_json: str | Path | dict[str, Any] | None = None,
        split_name: str | None = None,
        split_part: str | None = None,
        area_ids: list[str] | None = None,
        max_regions: int | None = None,
        max_areas: int | None = None,
        require_features: bool = True,
    ) -> None:
        self.root = Path(root)
        self.require_features = require_features
        if area_ids is None:
            if split_json is None or split_name is None or split_part is None:
                area_ids = self._discover_area_ids()
            else:
                splits = self._load_splits(split_json)
                area_ids = list(splits[split_name][split_part])
        if max_regions is not None:
            area_ids = [area_id for area_id in area_ids if self._n_regions(area_id) <= max_regions]
        if max_areas is not None:
            area_ids = self._limit_area_ids(area_ids, max_areas)
        self.area_ids = area_ids
        self.diagnostics = self._load_diagnostics()

    def _discover_area_ids(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if (path / "y_raw.npy").exists())

    def _load_splits(self, split_json: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(split_json, dict):
            return split_json
        with Path(split_json).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_diagnostics(self) -> dict[str, dict[str, Any]]:
        path = self.root / "split_records.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        return {row["area_id"]: row for row in records}

    def _n_regions(self, area_id: str) -> int:
        path = self.root / area_id / "y_raw.npy"
        return int(np.load(path, mmap_mode="r").shape[0])

    def _limit_area_ids(self, area_ids: list[str], max_areas: int) -> list[str]:
        if max_areas < 0:
            raise ValueError("max_areas must be non-negative")
        if max_areas == 0:
            return []
        if len(area_ids) <= max_areas:
            return area_ids
        indices = np.linspace(0, len(area_ids) - 1, num=max_areas, dtype=int)
        return [area_ids[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.area_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        area_id = self.area_ids[index]
        area_dir = self.root / area_id
        item: dict[str, Any] = {
            "area_id": area_id,
            "y_raw": torch.as_tensor(np.load(area_dir / "y_raw.npy"), dtype=torch.float32),
            "p_share": torch.as_tensor(np.load(area_dir / "p_share.npy"), dtype=torch.float32),
            "outflow": torch.as_tensor(np.load(area_dir / "outflow.npy"), dtype=torch.float32),
            "valid_origin": torch.as_tensor(np.load(area_dir / "valid_origin.npy"), dtype=torch.bool),
            "distance": None,
            "adjacency": None,
            "node_feat": None,
        }
        for key, filename, dtype in (
            ("distance", "distance.npy", torch.float32),
            ("adjacency", "adjacency.npy", torch.float32),
            ("node_feat", "node_feat.npy", torch.float32),
        ):
            path = area_dir / filename
            if path.exists():
                item[key] = torch.as_tensor(np.load(path), dtype=dtype)
        if self.require_features:
            missing = [key for key in ("distance", "adjacency", "node_feat") if item[key] is None]
            if missing:
                raise ValueError(f"Area {area_id} is missing required arrays: {missing}")
        item["diagnostics"] = self.diagnostics.get(area_id, {})
        return item


def collate_area_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one variable-size area."""
    if len(batch) != 1:
        raise ValueError("The first benchmark implementation supports batch size 1 only")
    return batch[0]


def load_meta(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "meta.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "areas" in payload:
        return list(payload["areas"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported metadata format: {path}")
