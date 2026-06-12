"""Write feature-name metadata for the LargeCommutingOD conversion.

The official archive stores demographic and POI feature names only in README.md.
This script extracts those tables, validates them against the converted arrays,
and writes machine-readable schema files next to the converted data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ZIP = Path("/Users/liuchangjian/Downloads/CommutingODGen-Dataset-main.zip")
DEFAULT_OUT_ROOT = Path("data/opflow_large_commuting_od")
README_NAME = "CommutingODGen-Dataset-main/README.md"


def _slug(text: str) -> str:
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _read_readme(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(README_NAME).decode("utf-8", errors="replace")


def _extract_one_column_table(readme: str, heading: str) -> list[str]:
    lines = readme.splitlines()
    start = next(idx for idx, line in enumerate(lines) if heading in line)
    names: list[str] = []
    collecting = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            collecting = True
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) != 1:
                continue
            value = cells[0]
            if not value or value.startswith(":---") or value.startswith("Characteristics"):
                continue
            names.append(value)
            continue
        if collecting and not stripped:
            break
    return names


def _find_area_dirs(out_root: Path) -> list[Path]:
    return sorted(path for path in out_root.iterdir() if path.is_dir() and (path / "node_feat.npy").exists())


def _unique_dims(area_dirs: list[Path], filename: str) -> list[int]:
    dims = set()
    for area_dir in area_dirs:
        path = area_dir / filename
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            if arr.ndim >= 2:
                dims.add(int(arr.shape[1]))
            elif arr.ndim == 1:
                dims.add(1)
    return sorted(dims)


def _region_meta_columns(out_root: Path) -> list[str]:
    for area_dir in _find_area_dirs(out_root):
        path = area_dir / "region_meta.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                return next(reader)
    return []


def _write_feature_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_schema(zip_path: Path, out_root: Path) -> dict[str, Any]:
    readme = _read_readme(zip_path)
    demo_names = _extract_one_column_table(readme, "Characteristics of Demographic Part")
    poi_names = _extract_one_column_table(readme, "Characteristics of Point-of-interests Part")
    area_dirs = _find_area_dirs(out_root)

    demo_rows = [
        {
            "source_array": "demos.npy",
            "source_index": idx,
            "node_feat_index": idx,
            "feature_id": f"demo_{idx:03d}_{_slug(name)}",
            "feature_name": name,
            "source_document": README_NAME,
        }
        for idx, name in enumerate(demo_names)
    ]
    poi_rows = [
        {
            "source_array": "pois.npy",
            "source_index": idx,
            "node_feat_index": len(demo_names) + idx,
            "feature_id": f"poi_{idx:03d}_{_slug(name)}",
            "feature_name": name,
            "source_document": README_NAME,
        }
        for idx, name in enumerate(poi_names)
    ]
    node_rows = demo_rows + poi_rows

    observed = {
        "areas_checked": len(area_dirs),
        "demos_feature_dims": _unique_dims(area_dirs, "demos.npy"),
        "pois_feature_dims": _unique_dims(area_dirs, "pois.npy"),
        "node_feat_feature_dims": _unique_dims(area_dirs, "node_feat.npy"),
    }
    expected_node_dim = len(demo_names) + len(poi_names)
    validation = {
        "demo_name_count_matches_arrays": observed["demos_feature_dims"] == [len(demo_names)],
        "poi_name_count_matches_arrays": observed["pois_feature_dims"] == [len(poi_names)],
        "node_feature_name_count_matches_arrays": observed["node_feat_feature_dims"] == [expected_node_dim],
    }

    return {
        "source_zip": str(zip_path),
        "source_readme": README_NAME,
        "notes": [
            "Feature names are extracted from the official README table in the original archive.",
            "node_feat.npy is the column-wise concatenation of demos.npy followed by pois.npy.",
            "The README prose says POIs are divided into 36 categories, but the README table and observed pois.npy arrays contain 34 columns.",
            "region_meta.csv columns come from the shapefile DBF fields plus computed geometry summaries.",
        ],
        "readme_claims": {
            "demographic_dimensions": 97,
            "poi_dimensions_in_prose": 36,
            "poi_names_listed": len(poi_names),
        },
        "observed_dimensions": observed,
        "validation": validation,
        "arrays": {
            "demos.npy": {
                "axis_0": "node_index aligned with region_meta.csv and shapefile record order",
                "axis_1": "demographic feature",
                "n_features": len(demo_names),
                "feature_names": demo_names,
            },
            "pois.npy": {
                "axis_0": "node_index aligned with region_meta.csv and shapefile record order",
                "axis_1": "POI category feature",
                "n_features": len(poi_names),
                "feature_names": poi_names,
            },
            "node_feat.npy": {
                "axis_0": "node_index aligned with region_meta.csv and shapefile record order",
                "axis_1": "node feature",
                "n_features": expected_node_dim,
                "composition": [
                    {"source_array": "demos.npy", "node_feat_index_start": 0, "node_feat_index_end_inclusive": len(demo_names) - 1},
                    {
                        "source_array": "pois.npy",
                        "node_feat_index_start": len(demo_names),
                        "node_feat_index_end_inclusive": expected_node_dim - 1,
                    },
                ],
                "feature_names": [row["feature_name"] for row in node_rows],
            },
            "od.npy": {"shape": "[N, N]", "meaning": "raw commuting OD counts; row is origin node_index, column is destination node_index"},
            "y_raw.npy": {"shape": "[N, N]", "meaning": "converted alias of od.npy used by OpFlow loaders"},
            "dis.npy": {"shape": "[N, N]", "meaning": "official pairwise centroid Euclidean distance matrix"},
            "distance.npy": {"shape": "[N, N]", "meaning": "converted alias of dis.npy used by OpFlow loaders"},
            "adj.npy": {"shape": "[N, N]", "meaning": "official pairwise adjacency matrix"},
            "adjacency.npy": {"shape": "[N, N]", "meaning": "converted alias of adj.npy used by OpFlow loaders"},
            "p_share.npy": {"shape": "[N, N]", "meaning": "row-normalized destination allocation from y_raw.npy"},
            "outflow.npy": {"shape": "[N]", "meaning": "origin production scale, equal to row sum of y_raw.npy"},
            "valid_origin.npy": {"shape": "[N]", "meaning": "boolean mask for origins with positive outflow"},
            "centroids.npy": {"shape": "[N, 2]", "meaning": "computed centroid coordinates as [longitude, latitude]"},
            "region_meta.csv": {
                "shape": "[N, columns]",
                "columns": _region_meta_columns(out_root),
            },
            "shapefile/*.shp": {"meaning": "official region geometries with DBF attributes; use as source of truth for GIS plotting"},
            "geometry.geojson": {"meaning": "convenience GeoJSON derived from the shapefile for quick visualization"},
        },
        "feature_tables": {
            "demos": demo_rows,
            "pois": poi_rows,
            "node_feat": node_rows,
        },
    }


def write_schema(schema: dict[str, Any], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    demo_rows = schema["feature_tables"]["demos"]
    poi_rows = schema["feature_tables"]["pois"]
    node_rows = schema["feature_tables"]["node_feat"]
    fieldnames = ["source_array", "source_index", "node_feat_index", "feature_id", "feature_name", "source_document"]
    _write_feature_csv(out_root / "demos_feature_names.csv", demo_rows, fieldnames)
    _write_feature_csv(out_root / "pois_feature_names.csv", poi_rows, fieldnames)
    _write_feature_csv(out_root / "node_feature_names.csv", node_rows, fieldnames)
    (out_root / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    schema = build_schema(args.zip_path, args.out_root)
    write_schema(schema, args.out_root)
    validation = schema["validation"]
    if not all(validation.values()):
        raise SystemExit(f"feature schema validation failed: {validation}")
    print(
        "wrote feature schema: "
        f"demos={schema['arrays']['demos.npy']['n_features']} "
        f"pois={schema['arrays']['pois.npy']['n_features']} "
        f"node_feat={schema['arrays']['node_feat.npy']['n_features']}"
    )


if __name__ == "__main__":
    main()
