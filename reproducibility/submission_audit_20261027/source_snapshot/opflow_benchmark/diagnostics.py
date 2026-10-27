"""Area-level diagnostics for LargeCommutingOD split construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-8


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_corr(left: np.ndarray, right: np.ndarray, *, spearman: bool = True) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return float("nan")
    if spearman:
        left = average_ranks(left)
        right = average_ranks(right)
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float(np.sum(left**2) * np.sum(right**2)))
    if denom <= EPS:
        return float("nan")
    return float(np.sum(left * right) / denom)


def entropy_from_mass(mass: np.ndarray) -> float:
    mass = np.maximum(np.asarray(mass, dtype=float), 0.0)
    total = float(mass.sum())
    if total <= EPS:
        return float("nan")
    p = mass / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + EPS)))


def hhi_from_mass(mass: np.ndarray) -> float:
    mass = np.maximum(np.asarray(mass, dtype=float), 0.0)
    total = float(mass.sum())
    if total <= EPS:
        return float("nan")
    p = mass / total
    return float(np.sum(p**2))


def gini_from_mass(mass: np.ndarray) -> float:
    mass = np.maximum(np.asarray(mass, dtype=float), 0.0)
    if mass.size == 0:
        return float("nan")
    total = float(mass.sum())
    if total <= EPS:
        return float("nan")
    sorted_mass = np.sort(mass)
    n = sorted_mass.size
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(weights * sorted_mass) / (n * total)) - ((n + 1.0) / n))


def area_diagnostics(root: str | Path, area_meta: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    area_id = area_meta["area_id"]
    area_dir = root / area_id
    y = np.load(area_dir / "y_raw.npy")
    p = np.load(area_dir / "p_share.npy")
    outflow = np.load(area_dir / "outflow.npy")
    valid = np.load(area_dir / "valid_origin.npy").astype(bool)
    distance = np.load(area_dir / "distance.npy") if (area_dir / "distance.npy").exists() else None
    node_feat = np.load(area_dir / "node_feat.npy") if (area_dir / "node_feat.npy").exists() else None
    n = int(y.shape[0])
    total_flow = float(y.sum())
    positive = y > 0
    if distance is not None:
        finite_dist = np.isfinite(distance)
        pos_mask = positive & finite_dist
        all_mask = finite_dist
        distance_flow_corr = safe_corr(-distance[pos_mask], np.log1p(y[pos_mask]), spearman=True) if np.any(pos_mask) else float("nan")
        distance_flow_corr_all = safe_corr(-distance[all_mask], y[all_mask], spearman=True) if np.any(all_mask) else float("nan")
        mean_distance = float(np.mean(distance[all_mask])) if np.any(all_mask) else float("nan")
    else:
        distance_flow_corr = float("nan")
        distance_flow_corr_all = float("nan")
        mean_distance = float("nan")
    dest_inflow = y.sum(axis=0)
    dest_entropy = entropy_from_mass(dest_inflow)
    mean_row_entropy = float(np.mean((-p[valid] * np.log(p[valid] + EPS)).sum(axis=1))) if np.any(valid) else float("nan")
    logn = math.log(max(n, 2))
    feature_opportunity_corr = float("nan")
    feature_opportunity_hhi = float("nan")
    feature_opportunity_gini = float("nan")
    feature_opportunity_entropy = float("nan")
    feature_opportunity_entropy_norm = float("nan")
    feature_norm_opportunity_corr = float("nan")
    if node_feat is not None:
        feat_sum = np.maximum(node_feat, 0.0).sum(axis=1)
        feat_norm = np.linalg.norm(node_feat, axis=1)
        feature_opportunity_corr = safe_corr(dest_inflow, feat_sum, spearman=True)
        feature_norm_opportunity_corr = safe_corr(dest_inflow, feat_norm, spearman=True)
        feature_opportunity_hhi = hhi_from_mass(feat_sum)
        feature_opportunity_gini = gini_from_mass(feat_sum)
        feature_opportunity_entropy = entropy_from_mass(feat_sum)
        feature_opportunity_entropy_norm = feature_opportunity_entropy / logn if np.isfinite(feature_opportunity_entropy) else float("nan")
    return {
        "area_id": area_id,
        "subset": area_meta.get("subset", "county"),
        "n_regions": n,
        "log_n_regions": math.log1p(n),
        "total_flow": total_flow,
        "log_total_flow": math.log1p(total_flow),
        "mean_outflow": float(np.mean(outflow)) if outflow.size else float("nan"),
        "median_outflow": float(np.median(outflow)) if outflow.size else float("nan"),
        "zero_origin_rate": float(np.mean(~valid)) if valid.size else float("nan"),
        "flow_density": float(np.mean(positive)) if y.size else float("nan"),
        "mean_distance": mean_distance,
        "distance_flow_corr": distance_flow_corr,
        "distance_flow_corr_all": distance_flow_corr_all,
        "dest_entropy": dest_entropy,
        "dest_entropy_norm": dest_entropy / logn if np.isfinite(dest_entropy) else float("nan"),
        "dest_hhi": hhi_from_mass(dest_inflow),
        "dest_gini": gini_from_mass(dest_inflow),
        "mean_row_entropy": mean_row_entropy,
        "mean_row_entropy_norm": mean_row_entropy / logn if np.isfinite(mean_row_entropy) else float("nan"),
        "feature_opportunity_corr": feature_opportunity_corr,
        "feature_norm_opportunity_corr": feature_norm_opportunity_corr,
        "feature_opportunity_hhi": feature_opportunity_hhi,
        "feature_opportunity_gini": feature_opportunity_gini,
        "feature_opportunity_entropy": feature_opportunity_entropy,
        "feature_opportunity_entropy_norm": feature_opportunity_entropy_norm,
    }


def compute_area_diagnostics(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    records = [area_diagnostics(root, area) for area in meta["areas"]]
    return pd.DataFrame(records)


def save_area_diagnostics(root: str | Path, table_path: str | Path = "tables/area_diagnostics.csv") -> pd.DataFrame:
    root = Path(root)
    frame = compute_area_diagnostics(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "split_records.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)
    return frame
