"""Inspect a LargeCommutingOD / CommutingODGen dataset snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


OD_NAMES = {"od.npy", "flow.npy", "flows.npy", "commuting.npy", "matrix.npy"}
DIST_NAMES = {"dis.npy", "dist.npy", "distance.npy"}
ADJ_NAMES = {"adj.npy", "adjacency.npy"}
NODE_NAMES = {"feature.npy", "features.npy", "x.npy", "node_feat.npy", "demos.npy", "pois.npy", "demo.npy", "poi.npy"}
EXTS = {".npy", ".csv", ".pkl", ".json"}


def _load_array_stats(path: Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    stats: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if np.issubdtype(arr.dtype, np.number):
        stats.update(
            {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }
        )
    return stats


def _find_area_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("*.npy") if path.name in OD_NAMES})


def _candidate(path: Path, names: set[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def inspect(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    ext_counts = {ext: sum(1 for path in files if path.suffix == ext) for ext in sorted(EXTS)}
    area_dirs = _find_area_dirs(root)
    areas = []
    n_regions = []
    total_flows = []
    for area_dir in area_dirs:
        od_path = _candidate(area_dir, OD_NAMES)
        dist_path = _candidate(area_dir, DIST_NAMES)
        adj_path = _candidate(area_dir, ADJ_NAMES)
        node_paths = [area_dir / name for name in sorted(NODE_NAMES) if (area_dir / name).exists()]
        area: dict[str, Any] = {
            "area_id": area_dir.name,
            "source_dir": str(area_dir),
            "od": None,
            "distance": None,
            "adjacency": None,
            "node_features": [],
            "is_square_od": False,
            "distance_matches_od": False,
            "adjacency_matches_od": False,
            "node_feature_rows_match": False,
        }
        if od_path is not None:
            area["od"] = {"path": str(od_path), **_load_array_stats(od_path)}
            shape = area["od"]["shape"]
            area["is_square_od"] = len(shape) == 2 and shape[0] == shape[1]
            if area["is_square_od"]:
                n_regions.append(int(shape[0]))
                od = np.load(od_path, mmap_mode="r")
                total_flows.append(float(np.nansum(od)))
        if dist_path is not None:
            area["distance"] = {"path": str(dist_path), **_load_array_stats(dist_path)}
            area["distance_matches_od"] = area["od"] is not None and area["distance"]["shape"] == area["od"]["shape"]
        if adj_path is not None:
            area["adjacency"] = {"path": str(adj_path), **_load_array_stats(adj_path)}
            area["adjacency_matches_od"] = area["od"] is not None and area["adjacency"]["shape"] == area["od"]["shape"]
        for node_path in node_paths:
            node = {"path": str(node_path), **_load_array_stats(node_path)}
            area["node_features"].append(node)
        if area["od"] is not None and area["node_features"]:
            area["node_feature_rows_match"] = all(len(node["shape"]) >= 1 and node["shape"][0] == area["od"]["shape"][0] for node in area["node_features"])
        areas.append(area)

    complete = [
        area
        for area in areas
        if area["is_square_od"] and area["distance_matches_od"] and area["adjacency_matches_od"] and area["node_feature_rows_match"]
    ]
    summary = {
        "total_areas": len(areas),
        "areas_with_od": sum(area["od"] is not None for area in areas),
        "areas_with_od_distance": sum(area["od"] is not None and area["distance_matches_od"] for area in areas),
        "areas_with_od_distance_node_features": sum(
            area["od"] is not None and area["distance_matches_od"] and area["node_feature_rows_match"] for area in areas
        ),
        "areas_complete": len(complete),
        "n_regions_min": int(min(n_regions)) if n_regions else None,
        "n_regions_median": float(median(n_regions)) if n_regions else None,
        "n_regions_max": int(max(n_regions)) if n_regions else None,
        "total_flow_min": float(min(total_flows)) if total_flows else None,
        "total_flow_median": float(median(total_flows)) if total_flows else None,
        "total_flow_max": float(max(total_flows)) if total_flows else None,
        "county_subset_exists": (root / "data").exists() and bool(list((root / "data").glob("*/od.npy"))) if (root / "data").exists() else False,
        "metropolitan_subset_exists": any("metro" in path.name.lower() for path in root.iterdir() if path.is_dir()),
        "metropolitan_od_matrices_found": any("metro" in str(path).lower() for path in root.rglob("od.npy")),
    }
    return {"root": str(root), "file_counts": ext_counts, "summary": summary, "areas": areas}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        f"total areas: {s['total_areas']}",
        f"areas with OD: {s['areas_with_od']}",
        f"areas with OD + distance: {s['areas_with_od_distance']}",
        f"areas with OD + distance + node features: {s['areas_with_od_distance_node_features']}",
        f"areas with complete OD/distance/adjacency/features: {s['areas_complete']}",
        f"min/median/max number of regions: {s['n_regions_min']} / {s['n_regions_median']} / {s['n_regions_max']}",
        f"min/median/max total OD flow: {s['total_flow_min']} / {s['total_flow_median']} / {s['total_flow_max']}",
        f"county subset exists: {s['county_subset_exists']}",
        f"metropolitan subset exists: {s['metropolitan_subset_exists']}",
        f"metropolitan OD matrices found: {s['metropolitan_od_matrices_found']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", "--out_json", dest="out", default="logs/data_inspection.json")
    parser.add_argument("--summary_out", "--out_summary", dest="summary_out", default="logs/data_inspection_summary.txt")
    args = parser.parse_args()
    payload = inspect(Path(args.data_root))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(Path(args.summary_out), payload)


if __name__ == "__main__":
    main()
