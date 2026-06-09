"""Convert official LargeCommutingOD arrays to OpFlow allocation format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_benchmark.metrics import matrix_to_share


EPS = 1e-8


def _find_area_dirs(raw_root: Path) -> list[Path]:
    return sorted(path.parent for path in raw_root.rglob("od.npy"))


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _node_features(area_dir: Path) -> tuple[np.ndarray | None, list[str]]:
    parts = []
    names = []
    for filename in ("demos.npy", "pois.npy"):
        path = area_dir / filename
        if path.exists():
            parts.append(np.load(path))
            names.append(filename)
    if parts:
        return np.concatenate(parts, axis=1).astype(np.float32), names
    for filename in ("node_feat.npy", "features.npy", "feature.npy", "x.npy"):
        path = area_dir / filename
        if path.exists():
            return np.load(path).astype(np.float32), [filename]
    return None, []


def convert(raw_root: Path, out_root: Path, zero_diagonal: bool = False, report_path: Path | None = None) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    validation = []
    for area_dir in _find_area_dirs(raw_root):
        area_id = area_dir.name
        y = np.load(area_dir / "od.npy").astype(np.float32)
        if y.ndim != 2 or y.shape[0] != y.shape[1]:
            validation.append({"area_id": area_id, "status": "skipped", "reason": "od_not_square", "shape": list(y.shape)})
            continue
        diagonal_flow = float(np.trace(y))
        if zero_diagonal:
            np.fill_diagonal(y, 0.0)
        y = np.maximum(y, 0.0)
        p_share, outflow, valid_origin = matrix_to_share(y, eps=EPS)
        distance = _load_optional(area_dir / "dis.npy")
        adjacency = _load_optional(area_dir / "adj.npy")
        node_feat, node_sources = _node_features(area_dir)

        area_out = out_root / area_id
        area_out.mkdir(parents=True, exist_ok=True)
        np.save(area_out / "y_raw.npy", y)
        np.save(area_out / "p_share.npy", p_share.astype(np.float32))
        np.save(area_out / "outflow.npy", outflow.astype(np.float32))
        np.save(area_out / "valid_origin.npy", valid_origin.astype(bool))
        if distance is not None:
            np.save(area_out / "distance.npy", distance.astype(np.float32))
        if adjacency is not None:
            np.save(area_out / "adjacency.npy", adjacency.astype(np.float32))
        if node_feat is not None:
            np.save(area_out / "node_feat.npy", node_feat)

        row_sum_error = float(np.max(np.abs(p_share[valid_origin].sum(axis=1) - 1.0))) if np.any(valid_origin) else 0.0
        outflow_error = float(np.max(np.abs(outflow - y.sum(axis=1)))) if y.size else 0.0
        checks = {
            "area_id": area_id,
            "status": "ok",
            "y_square": y.shape[0] == y.shape[1],
            "p_shape_matches": p_share.shape == y.shape,
            "valid_row_sum_max_error": row_sum_error,
            "outflow_max_error": outflow_error,
            "distance_shape_matches": distance is None or distance.shape == y.shape,
            "adjacency_shape_matches": adjacency is None or adjacency.shape == y.shape,
            "node_feat_rows_match": node_feat is None or node_feat.shape[0] == y.shape[0],
            "all_saved_arrays_finite": bool(
                np.isfinite(y).all()
                and np.isfinite(p_share).all()
                and np.isfinite(outflow).all()
                and (distance is None or np.isfinite(distance).all())
                and (adjacency is None or np.isfinite(adjacency).all())
                and (node_feat is None or np.isfinite(node_feat).all())
            ),
            "nonnegative_od": bool((y >= 0).all()),
            "distance_diagonal_finite": bool(distance is None or np.isfinite(np.diag(distance)).all()),
            "distance_diagonal_max_abs": float(np.max(np.abs(np.diag(distance)))) if distance is not None else None,
        }
        validation.append(checks)
        records.append(
            {
                "area_id": area_id,
                "subset": "county",
                "n_regions": int(y.shape[0]),
                "total_flow": float(y.sum()),
                "n_valid_origins": int(valid_origin.sum()),
                "has_distance": distance is not None,
                "has_adjacency": adjacency is not None,
                "has_node_feat": node_feat is not None,
                "node_feature_sources": node_sources,
                "source_dir": str(area_dir),
                "diagonal_flow": diagonal_flow,
                "diagonal_flow_after_conversion": float(np.trace(y)),
                "diagonal_flow_share": diagonal_flow / float(np.load(area_dir / "od.npy").sum()) if float(np.load(area_dir / "od.npy").sum()) > EPS else 0.0,
                "zero_diagonal": zero_diagonal,
            }
        )

    meta = {"source_root": str(raw_root), "zero_diagonal": zero_diagonal, "areas": records}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "source_root": str(raw_root),
        "out_root": str(out_root),
        "n_converted": len(records),
        "n_skipped": sum(row["status"] != "ok" for row in validation),
        "validation": validation,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", default="data/opflow_large_commuting_od")
    parser.add_argument("--report", default="logs/opflow_conversion_report.json")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--zero_diagonal", action="store_true")
    args = parser.parse_args()
    report_path = Path(args.report)
    if args.log_dir is not None:
        report_path = Path(args.log_dir) / "opflow_conversion_report.json"
    report = convert(Path(args.raw_root), Path(args.out_root), zero_diagonal=args.zero_diagonal, report_path=report_path)
    summary_path = (Path(args.log_dir) if args.log_dir is not None else report_path.parent) / "opflow_conversion_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                f"source root: {report['source_root']}",
                f"out root: {report['out_root']}",
                f"converted areas: {report['n_converted']}",
                f"skipped areas: {report['n_skipped']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
