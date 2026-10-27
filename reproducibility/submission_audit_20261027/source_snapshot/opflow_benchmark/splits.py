"""Matched split construction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def quantile_bins(series: pd.Series, q: int) -> pd.Series:
    ranked = series.rank(method="first")
    bins = pd.qcut(ranked, q=min(q, ranked.nunique()), labels=False, duplicates="drop")
    return bins.astype("Int64")


def random_split(area_ids: list[str], seed: int = 2026) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(area_ids)
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    return {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}


def matched_extreme_split(
    records: pd.DataFrame,
    *,
    target_col: str,
    control_cols: tuple[str, str] = ("log_n_regions", "log_total_flow"),
    n_bins: int = 5,
    tail_frac: float = 0.3,
    mid_frac: float = 0.2,
    high_to_low: bool = True,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    frame = records.dropna(subset=[target_col, *control_cols]).copy()
    for col in control_cols:
        frame[f"{col}_bin"] = quantile_bins(frame[col], n_bins)
    train_ids: list[str] = []
    test_ids: list[str] = []
    val_ids: list[str] = []
    bin_rows = []
    group_cols = [f"{col}_bin" for col in control_cols]
    for bin_key, group in frame.groupby(group_cols, dropna=True):
        group = group.sort_values(target_col)
        n = len(group)
        n_tail = int(np.floor(n * tail_frac))
        n_mid = int(np.floor(n * mid_frac))
        if n_tail < 2 or n_mid < 1:
            continue
        low = group.head(n_tail)
        high = group.tail(n_tail)
        mid_start = max((n - n_mid) // 2, 0)
        mid = group.iloc[mid_start : mid_start + n_mid]
        if high_to_low:
            train_ids.extend(high["area_id"].tolist())
            test_ids.extend(low["area_id"].tolist())
        else:
            train_ids.extend(low["area_id"].tolist())
            test_ids.extend(high["area_id"].tolist())
        val_ids.extend(mid["area_id"].tolist())
        bin_rows.append(
            {
                "bin": str(bin_key),
                "n": n,
                "n_tail": n_tail,
                "n_mid": n_mid,
                "low_mean": float(low[target_col].mean()),
                "high_mean": float(high[target_col].mean()),
            }
        )
    return {"train": train_ids, "val": val_ids, "test": test_ids}, pd.DataFrame(bin_rows)


def split_balance(records: pd.DataFrame, splits: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    by_area = records.set_index("area_id")
    rows = []
    for split_name, parts in splits.items():
        for part, area_ids in parts.items():
            sub = by_area.loc[[area_id for area_id in area_ids if area_id in by_area.index]].copy()
            rows.append(
                {
                    "split_name": split_name,
                    "part": part,
                    "n_areas": int(len(sub)),
                    "mean_n_regions": float(sub["n_regions"].mean()) if len(sub) else float("nan"),
                    "std_n_regions": float(sub["n_regions"].std()) if len(sub) > 1 else 0.0,
                    "median_n_regions": float(sub["n_regions"].median()) if len(sub) else float("nan"),
                    "mean_total_flow": float(sub["total_flow"].mean()) if len(sub) else float("nan"),
                    "median_total_flow": float(sub["total_flow"].median()) if len(sub) else float("nan"),
                    "mean_log_total_flow": float(sub["log_total_flow"].mean()) if len(sub) else float("nan"),
                    "mean_distance_flow_corr": float(sub["distance_flow_corr"].mean()) if len(sub) else float("nan"),
                    "mean_dest_entropy_norm": float(sub["dest_entropy_norm"].mean()) if len(sub) else float("nan"),
                    "mean_dest_hhi": float(sub["dest_hhi"].mean()) if len(sub) else float("nan"),
                    "mean_dest_gini": float(sub["dest_gini"].mean()) if "dest_gini" in sub and len(sub) else float("nan"),
                    "mean_feature_opportunity_entropy_norm": float(sub["feature_opportunity_entropy_norm"].mean())
                    if "feature_opportunity_entropy_norm" in sub and len(sub)
                    else float("nan"),
                    "mean_feature_opportunity_gini": float(sub["feature_opportunity_gini"].mean())
                    if "feature_opportunity_gini" in sub and len(sub)
                    else float("nan"),
                    "mean_flow_density": float(sub["flow_density"].mean()) if len(sub) else float("nan"),
                    "mean_distance": float(sub["mean_distance"].mean()) if "mean_distance" in sub and len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def balance_acceptance(balance: pd.DataFrame, split_name: str) -> dict[str, Any]:
    parts = balance.loc[balance["split_name"] == split_name].set_index("part")
    if not {"train", "test"} <= set(parts.index):
        return {"accepted": False, "reason": "missing train/test"}
    train = parts.loc["train"]
    test = parts.loc["test"]
    n_ratio = max(train["mean_n_regions"], test["mean_n_regions"]) / max(min(train["mean_n_regions"], test["mean_n_regions"]), 1e-8)
    flow_ratio = max(train["mean_total_flow"], test["mean_total_flow"]) / max(min(train["mean_total_flow"], test["mean_total_flow"]), 1e-8)
    return {
        "accepted": bool(n_ratio < 2.0 and flow_ratio < 2.0 and train["n_areas"] >= 20 and test["n_areas"] >= 20),
        "n_regions_ratio": float(n_ratio),
        "total_flow_ratio": float(flow_ratio),
        "train_n": int(train["n_areas"]),
        "test_n": int(test["n_areas"]),
    }


def write_splits(root: str | Path, splits: dict[str, dict[str, list[str]]], balance: pd.DataFrame) -> None:
    root = Path(root)
    (root / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    (root / "split_balance.json").write_text(balance.to_json(orient="records", indent=2), encoding="utf-8")
