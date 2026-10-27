"""RandSP, GeoSP, and OppSP county-level splits for LargeCommutingOD."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


CENSUS_DIVISION = {
    "01": "East South Central",
    "02": "Pacific",
    "04": "Mountain",
    "05": "West South Central",
    "06": "Pacific",
    "08": "Mountain",
    "09": "New England",
    "10": "South Atlantic",
    "11": "South Atlantic",
    "12": "South Atlantic",
    "13": "South Atlantic",
    "15": "Pacific",
    "16": "Mountain",
    "17": "East North Central",
    "18": "East North Central",
    "19": "West North Central",
    "20": "West North Central",
    "21": "East South Central",
    "22": "West South Central",
    "23": "New England",
    "24": "South Atlantic",
    "25": "New England",
    "26": "East North Central",
    "27": "West North Central",
    "28": "East South Central",
    "29": "West North Central",
    "30": "Mountain",
    "31": "West North Central",
    "32": "Mountain",
    "33": "New England",
    "34": "Middle Atlantic",
    "35": "Mountain",
    "36": "Middle Atlantic",
    "37": "South Atlantic",
    "38": "West North Central",
    "39": "East North Central",
    "40": "West South Central",
    "41": "Pacific",
    "42": "Middle Atlantic",
    "44": "New England",
    "45": "South Atlantic",
    "46": "West North Central",
    "47": "East South Central",
    "48": "West South Central",
    "49": "Mountain",
    "50": "New England",
    "51": "South Atlantic",
    "53": "Pacific",
    "54": "South Atlantic",
    "55": "East North Central",
    "56": "Mountain",
}

RANDSP_SEEDS = [0, 1, 2, 3, 4]
RANDSP_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
GEOSP_VAL_RATIO = 0.10
GEOSP_VAL_SEED = 0
OPPSP_SEED = 0
OPPSP_VAL_RATIO = 0.10
OPPSP_TAIL_QUANTILE = 0.10
OPPSP_EPS = 1e-8
REQUIRED_FILES = ("od.npy", "node_feat.npy", "distance.npy", "region_meta.csv")
OPPSP_REQUIRED_FILES = ("node_feat.npy", "region_meta.csv")
SPLIT_COLUMNS = ["split_name", "seed", "fold_id", "area_id", "role", "STATEFP", "COUNTYFP", "census_division", "n_nodes"]
OPPSP_COLUMNS = [
    "split_name",
    "seed",
    "fold_id",
    "area_id",
    "role",
    "STATEFP",
    "COUNTYFP",
    "census_division",
    "n_nodes",
    "opp_concentration_hhi",
    "opp_group",
]
OPPSP_SCORE_COLUMNS = [
    "area_id",
    "STATEFP",
    "COUNTYFP",
    "census_division",
    "n_nodes",
    "poi_total",
    "poi_nonzero_nodes",
    "opp_concentration_hhi",
    "opp_group",
]


def division_slug(name: str) -> str:
    return "".join(ch for ch in str(name) if ch.isalnum())


def randsp_key(seed: int) -> str:
    return f"RandSP_seed{int(seed)}"


def geosp_key(division: str) -> str:
    return f"GeoSP_{division_slug(division)}"


def _read_region_codes(area_dir: Path, area_id: str) -> tuple[str, str]:
    meta = pd.read_csv(area_dir / "region_meta.csv", dtype=str)
    canonical_state = area_id[:2]
    canonical_county = area_id[2:5]
    if "STATEFP" in meta and not meta["STATEFP"].dropna().empty:
        state_values = sorted({str(value).zfill(2) for value in meta["STATEFP"].dropna().unique()})
        if state_values and state_values != [canonical_state]:
            warnings.warn(
                f"STATEFP in {area_dir / 'region_meta.csv'} is {state_values}, expected {canonical_state}; using area_id prefix.",
                RuntimeWarning,
                stacklevel=2,
            )
    if "COUNTYFP" in meta and not meta["COUNTYFP"].dropna().empty:
        county_values = sorted({str(value).zfill(3) for value in meta["COUNTYFP"].dropna().unique()})
        if county_values and county_values != [canonical_county]:
            warnings.warn(
                f"COUNTYFP in {area_dir / 'region_meta.csv'} is {county_values}, expected {canonical_county}; using area_id suffix.",
                RuntimeWarning,
                stacklevel=2,
            )
    return canonical_state, canonical_county


def oppsp_key() -> str:
    return "OppSP"


def discover_counties(data_root: str | Path, *, required_files: Sequence[str] = REQUIRED_FILES) -> pd.DataFrame:
    root = Path(data_root)
    rows: list[dict[str, Any]] = []
    for area_dir in sorted(root.iterdir()):
        area_id = area_dir.name
        if not area_dir.is_dir() or len(area_id) != 5 or not area_id.isdigit():
            continue
        if any(not (area_dir / filename).exists() for filename in required_files):
            continue
        statefp, countyfp = _read_region_codes(area_dir, area_id)
        n_nodes = int(np.load(area_dir / "node_feat.npy", mmap_mode="r").shape[0])
        rows.append(
            {
                "area_id": area_id,
                "STATEFP": statefp,
                "COUNTYFP": countyfp,
                "census_division": CENSUS_DIVISION.get(statefp, "Unknown"),
                "n_nodes": n_nodes,
            }
        )
    frame = pd.DataFrame(rows, columns=["area_id", "STATEFP", "COUNTYFP", "census_division", "n_nodes"])
    if frame.empty:
        raise ValueError(f"No valid county directories found under {root}")
    frame["area_id"] = frame["area_id"].astype(str)
    frame["STATEFP"] = frame["STATEFP"].astype(str).str.zfill(2)
    frame["COUNTYFP"] = frame["COUNTYFP"].astype(str).str.zfill(3)
    return frame.sort_values("area_id").reset_index(drop=True)


def _with_node_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    unique = int(out["n_nodes"].nunique())
    if unique <= 1:
        out["n_nodes_bin"] = "bin0"
        return out
    q = min(4, unique)
    try:
        bins = pd.qcut(out["n_nodes"], q=q, labels=False, duplicates="drop")
        out["n_nodes_bin"] = bins.astype("Int64").astype(str).replace("<NA>", "0").radd("bin")
    except ValueError:
        out["n_nodes_bin"] = "bin0"
    return out


def _sklearn_split(
    area_ids: list[str],
    *,
    test_size: float,
    seed: int,
    labels: list[str] | None,
) -> tuple[list[str], list[str]]:
    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        labels = None
    if labels is not None:
        try:
            left, right = train_test_split(area_ids, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)
            return sorted(left), sorted(right)
        except ValueError:
            pass
    rng = np.random.default_rng(seed)
    shuffled = list(area_ids)
    rng.shuffle(shuffled)
    n_right = int(round(len(shuffled) * float(test_size)))
    if len(shuffled) > 1:
        n_right = min(max(n_right, 1), len(shuffled) - 1)
    else:
        n_right = len(shuffled)
    right = sorted(shuffled[:n_right])
    left = sorted(shuffled[n_right:])
    return left, right


def _strata(frame: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    return frame.loc[:, list(cols)].astype(str).agg("|".join, axis=1).tolist()


def _split_with_fallback(
    frame: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    preferred: tuple[str, ...],
    fallback: tuple[str, ...],
) -> tuple[list[str], list[str], str]:
    for label, cols in (("division_node_bin", preferred), ("division", fallback)):
        labels = _strata(frame, cols)
        if min(pd.Series(labels).value_counts()) < 2:
            continue
        left, right = _sklearn_split(frame["area_id"].tolist(), test_size=test_size, seed=seed, labels=labels)
        if left and right:
            return left, right, label
    left, right = _sklearn_split(frame["area_id"].tolist(), test_size=test_size, seed=seed, labels=None)
    return left, right, "unstratified"


def _randsp_roles(frame: pd.DataFrame, seed: int) -> tuple[dict[str, str], str]:
    binned = _with_node_bins(frame)
    train_val, test, first = _split_with_fallback(
        binned,
        test_size=RANDSP_RATIOS["test"],
        seed=seed,
        preferred=("census_division", "n_nodes_bin"),
        fallback=("census_division",),
    )
    train_val_frame = binned.loc[binned["area_id"].isin(train_val)].copy()
    val_size = RANDSP_RATIOS["val"] / (RANDSP_RATIOS["train"] + RANDSP_RATIOS["val"])
    train, val, second = _split_with_fallback(
        train_val_frame,
        test_size=val_size,
        seed=seed + 7919,
        preferred=("census_division", "n_nodes_bin"),
        fallback=("census_division",),
    )
    roles = {area_id: "train" for area_id in train}
    roles.update({area_id: "val" for area_id in val})
    roles.update({area_id: "test" for area_id in test})
    return roles, f"{first}/{second}"


def _geosp_roles(frame: pd.DataFrame, division: str) -> tuple[dict[str, str], str]:
    binned = _with_node_bins(frame)
    test = sorted(binned.loc[binned["census_division"].eq(division), "area_id"].tolist())
    non_test = binned.loc[~binned["area_id"].isin(test)].copy()
    train, val, method = _split_with_fallback(
        non_test,
        test_size=GEOSP_VAL_RATIO,
        seed=GEOSP_VAL_SEED,
        preferred=("census_division", "n_nodes_bin"),
        fallback=("census_division",),
    )
    roles = {area_id: "train" for area_id in train}
    roles.update({area_id: "val" for area_id in val})
    roles.update({area_id: "test" for area_id in test})
    return roles, method


def _parse_poi_cols(raw: str | Sequence[int] | None) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
        if not values:
            return None
        return sorted({int(value) for value in values})
    return sorted({int(value) for value in raw})


def _poi_cols_from_csv(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "node_feat_index" not in frame:
        return None
    if "source_array" in frame:
        mask = frame["source_array"].astype(str).str.lower().eq("pois.npy")
        if bool(mask.any()):
            return sorted(frame.loc[mask, "node_feat_index"].astype(int).tolist())
    if path.name == "pois_feature_names.csv":
        return sorted(frame["node_feat_index"].astype(int).tolist())
    if "feature_id" in frame:
        mask = frame["feature_id"].astype(str).str.lower().str.startswith("poi_")
        if bool(mask.any()):
            return sorted(frame.loc[mask, "node_feat_index"].astype(int).tolist())
    return None


def _poi_cols_from_schema(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    arrays = payload.get("arrays", {})
    node_feat = arrays.get("node_feat.npy", {})
    for block in node_feat.get("composition", []):
        if str(block.get("source_array", "")).lower() == "pois.npy":
            start = int(block["node_feat_index_start"])
            end = int(block["node_feat_index_end_inclusive"])
            return list(range(start, end + 1))
    tables = payload.get("feature_tables", {})
    pois = tables.get("pois")
    if pois:
        return sorted(int(row["node_feat_index"]) for row in pois if "node_feat_index" in row)
    return None


def resolve_poi_columns(data_root: str | Path, poi_cols: str | Sequence[int] | None = None) -> list[int]:
    explicit = _parse_poi_cols(poi_cols)
    if explicit is not None:
        if not explicit:
            raise ValueError("--poi-cols was provided but no columns were parsed")
        return explicit
    root = Path(data_root)
    candidates = [
        _poi_cols_from_csv(root / "pois_feature_names.csv"),
        _poi_cols_from_csv(root / "node_feature_names.csv"),
        _poi_cols_from_csv(root / "feature_names.csv"),
        _poi_cols_from_csv(root / "node_feat_columns.csv"),
        _poi_cols_from_schema(root / "feature_schema.json"),
        _poi_cols_from_schema(root / "node_feat_columns.json"),
        _poi_cols_from_schema(root / "feature_names.json"),
    ]
    for cols in candidates:
        if cols:
            return sorted({int(col) for col in cols})
    raise ValueError(
        "Could not identify POI columns from feature metadata. "
        "Provide comma-separated node_feat indices with --poi-cols; OppSP will not silently use all node features."
    )


def _oppsp_scores(data_root: str | Path, *, poi_cols: str | Sequence[int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_root)
    poi_indices = resolve_poi_columns(root, poi_cols)
    inventory = discover_counties(root, required_files=OPPSP_REQUIRED_FILES)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for area in inventory.itertuples(index=False):
        area_id = str(area.area_id)
        node_feat = np.asarray(np.load(root / area_id / "node_feat.npy", mmap_mode="r"), dtype=float)
        if node_feat.ndim != 2:
            raise ValueError(f"{area_id}/node_feat.npy must be 2D, got shape {node_feat.shape}")
        if max(poi_indices) >= node_feat.shape[1] or min(poi_indices) < 0:
            raise ValueError(f"POI column indices {poi_indices[:5]}... are out of bounds for {area_id}/node_feat.npy with {node_feat.shape[1]} columns")
        n_nodes = int(node_feat.shape[0])
        if n_nodes < 2:
            skipped.append({"area_id": area_id, "reason": "n_nodes < 2", "n_nodes": n_nodes})
            continue
        poi_values = node_feat[:, poi_indices]
        if not np.isfinite(poi_values).all():
            raise ValueError(f"Non-finite POI values found in {area_id}/node_feat.npy")
        min_value = float(poi_values.min()) if poi_values.size else 0.0
        if min_value < -OPPSP_EPS:
            raise ValueError(
                f"Negative POI values found in {area_id}/node_feat.npy. "
                "OppSP requires raw nonnegative POI count columns; standardized features are not acceptable."
            )
        opportunity = np.maximum(poi_values, 0.0).sum(axis=1)
        smoothed = opportunity + OPPSP_EPS
        p = smoothed / smoothed.sum()
        raw_hhi = float(np.square(p).sum())
        denom = 1.0 - 1.0 / float(n_nodes)
        concentration = 0.0 if denom <= OPPSP_EPS else (raw_hhi - 1.0 / float(n_nodes)) / denom
        rows.append(
            {
                "area_id": area_id,
                "STATEFP": str(area.STATEFP).zfill(2),
                "COUNTYFP": str(area.COUNTYFP).zfill(3),
                "census_division": area.census_division,
                "n_nodes": n_nodes,
                "poi_total": float(opportunity.sum()),
                "poi_nonzero_nodes": int((opportunity > OPPSP_EPS).sum()),
                "opp_concentration_hhi": float(np.clip(concentration, 0.0, 1.0)),
            }
        )
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError(f"No valid counties with n_nodes >= 2 found under {root}")
    q_low = float(scores["opp_concentration_hhi"].quantile(OPPSP_TAIL_QUANTILE))
    q_high = float(scores["opp_concentration_hhi"].quantile(1.0 - OPPSP_TAIL_QUANTILE))
    scores["opp_group"] = "middle"
    scores.loc[scores["opp_concentration_hhi"] <= q_low, "opp_group"] = "low_tail"
    scores.loc[scores["opp_concentration_hhi"] >= q_high, "opp_group"] = "high_tail"
    scores = scores.loc[:, OPPSP_SCORE_COLUMNS].sort_values("area_id").reset_index(drop=True)
    metadata = {
        "poi_columns": poi_indices,
        "q_low": q_low,
        "q_high": q_high,
        "tail_quantile": OPPSP_TAIL_QUANTILE,
        "skipped_counties": skipped,
        "uses_od_labels": False,
    }
    return scores, metadata


def _oppsp_roles(scores: pd.DataFrame) -> tuple[dict[str, str], str]:
    test = sorted(scores.loc[scores["opp_group"].isin(["low_tail", "high_tail"]), "area_id"].astype(str).tolist())
    source = scores.loc[~scores["area_id"].isin(test)].copy()
    val_count = int(round(len(scores) * OPPSP_VAL_RATIO))
    if len(source) > 1:
        val_count = min(max(val_count, 1), len(source) - 1)
    else:
        val_count = len(source)
    if val_count <= 0:
        train, val, method = source["area_id"].astype(str).tolist(), [], "none"
    else:
        binned = _with_node_bins(source)
        test_size = val_count / max(len(binned), 1)
        train, val, method = _split_with_fallback(
            binned,
            test_size=test_size,
            seed=OPPSP_SEED,
            preferred=("census_division", "n_nodes_bin"),
            fallback=("census_division",),
        )
    roles = {area_id: "train" for area_id in train}
    roles.update({area_id: "val" for area_id in val})
    roles.update({area_id: "test" for area_id in test})
    return roles, method


def _oppsp_rows(scores: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    rows = scores.copy()
    rows["split_name"] = "OppSP"
    rows["seed"] = OPPSP_SEED
    rows["fold_id"] = "OppSP"
    rows["role"] = rows["area_id"].astype(str).map(roles)
    if rows["role"].isna().any():
        missing = rows.loc[rows["role"].isna(), "area_id"].tolist()
        raise ValueError(f"Missing OppSP split role for counties: {missing[:5]}")
    return rows.loc[:, OPPSP_COLUMNS].sort_values(["role", "area_id"]).reset_index(drop=True)


def _split_rows(frame: pd.DataFrame, *, split_name: str, seed: int, fold_id: str | int, roles: dict[str, str]) -> pd.DataFrame:
    rows = frame.copy()
    rows["split_name"] = split_name
    rows["seed"] = int(seed)
    rows["fold_id"] = str(fold_id)
    rows["role"] = rows["area_id"].map(roles)
    if rows["role"].isna().any():
        missing = rows.loc[rows["role"].isna(), "area_id"].tolist()
        raise ValueError(f"Missing split role for counties: {missing[:5]}")
    return rows.loc[:, SPLIT_COLUMNS].sort_values(["role", "area_id"]).reset_index(drop=True)


def _role_stats(rows: pd.DataFrame, role: str) -> dict[str, Any]:
    sub = rows.loc[rows["role"].eq(role)]
    dist = sub["census_division"].value_counts().sort_index().to_dict()
    return {
        f"{role}_counties": int(len(sub)),
        f"{role}_tract_nodes": int(sub["n_nodes"].sum()),
        f"{role}_division_distribution": json.dumps(dist, sort_keys=True),
        f"{role}_n_nodes_min": int(sub["n_nodes"].min()) if not sub.empty else 0,
        f"{role}_n_nodes_median": float(sub["n_nodes"].median()) if not sub.empty else 0.0,
        f"{role}_n_nodes_max": int(sub["n_nodes"].max()) if not sub.empty else 0,
    }


def _summary_row(rows: pd.DataFrame, *, split_key: str, stratification: str) -> dict[str, Any]:
    first = rows.iloc[0]
    row = {
        "split_key": split_key,
        "split_name": first["split_name"],
        "seed": int(first["seed"]),
        "fold_id": first["fold_id"],
        "stratification": stratification,
    }
    for role in ("train", "val", "test"):
        row.update(_role_stats(rows, role))
    return row


def _role_distribution_json(frame: pd.DataFrame, column: str) -> str:
    return json.dumps(frame[column].value_counts().sort_index().to_dict(), sort_keys=True)


def _oppsp_role_summary(rows: pd.DataFrame, *, checks: dict[str, bool]) -> pd.DataFrame:
    records = []
    for role in ("train", "val", "test"):
        sub = rows.loc[rows["role"].eq(role)].copy()
        records.append(
            {
                "split_name": "OppSP",
                "seed": OPPSP_SEED,
                "fold_id": "OppSP",
                "role": role,
                "n_counties": int(len(sub)),
                "tract_nodes": int(sub["n_nodes"].sum()),
                "n_nodes_min": int(sub["n_nodes"].min()) if not sub.empty else 0,
                "n_nodes_median": float(sub["n_nodes"].median()) if not sub.empty else 0.0,
                "n_nodes_max": int(sub["n_nodes"].max()) if not sub.empty else 0,
                "opp_concentration_hhi_min": float(sub["opp_concentration_hhi"].min()) if not sub.empty else 0.0,
                "opp_concentration_hhi_median": float(sub["opp_concentration_hhi"].median()) if not sub.empty else 0.0,
                "opp_concentration_hhi_max": float(sub["opp_concentration_hhi"].max()) if not sub.empty else 0.0,
                "poi_total_sum": float(sub["poi_total"].sum()) if "poi_total" in sub else 0.0,
                "poi_total_min": float(sub["poi_total"].min()) if "poi_total" in sub and not sub.empty else 0.0,
                "poi_total_median": float(sub["poi_total"].median()) if "poi_total" in sub and not sub.empty else 0.0,
                "poi_total_max": float(sub["poi_total"].max()) if "poi_total" in sub and not sub.empty else 0.0,
                "poi_nonzero_nodes_sum": int(sub["poi_nonzero_nodes"].sum()) if "poi_nonzero_nodes" in sub else 0,
                "poi_nonzero_nodes_min": int(sub["poi_nonzero_nodes"].min()) if "poi_nonzero_nodes" in sub and not sub.empty else 0,
                "poi_nonzero_nodes_median": float(sub["poi_nonzero_nodes"].median()) if "poi_nonzero_nodes" in sub and not sub.empty else 0.0,
                "poi_nonzero_nodes_max": int(sub["poi_nonzero_nodes"].max()) if "poi_nonzero_nodes" in sub and not sub.empty else 0,
                "census_division_distribution": _role_distribution_json(sub, "census_division") if not sub.empty else "{}",
                "opp_group_distribution": _role_distribution_json(sub, "opp_group") if not sub.empty else "{}",
                **{f"check_{name}": bool(value) for name, value in checks.items()},
            }
        )
    return pd.DataFrame(records)


def _oppsp_summary_payload(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    stratification: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    role_by_area = rows.set_index("area_id")["role"]
    checks = {
        "every_valid_county_once": int(scores["area_id"].nunique()) == int(len(rows)) == int(rows["area_id"].nunique()),
        "no_county_multiple_roles": not rows.duplicated("area_id").any(),
        "no_od_label_loaded_or_used": True,
        "test_has_low_tail": bool(((rows["role"] == "test") & (rows["opp_group"] == "low_tail")).any()),
        "test_has_high_tail": bool(((rows["role"] == "test") & (rows["opp_group"] == "high_tail")).any()),
        "validation_only_middle": bool(rows.loc[rows["role"].eq("val"), "opp_group"].eq("middle").all()),
        "train_only_middle": bool(rows.loc[rows["role"].eq("train"), "opp_group"].eq("middle").all()),
        "all_scores_assigned": bool(scores["area_id"].astype(str).isin(role_by_area.index.astype(str)).all()),
    }
    summary_csv = _oppsp_role_summary(rows, checks=checks)
    roles: dict[str, Any] = {}
    for role in ("train", "val", "test"):
        sub = rows.loc[rows["role"].eq(role)].copy()
        roles[role] = {
            "n_counties": int(len(sub)),
            "tract_nodes": int(sub["n_nodes"].sum()),
            "n_nodes": {
                "min": int(sub["n_nodes"].min()) if not sub.empty else 0,
                "median": float(sub["n_nodes"].median()) if not sub.empty else 0.0,
                "max": int(sub["n_nodes"].max()) if not sub.empty else 0,
            },
            "opp_concentration_hhi": {
                "min": float(sub["opp_concentration_hhi"].min()) if not sub.empty else 0.0,
                "median": float(sub["opp_concentration_hhi"].median()) if not sub.empty else 0.0,
                "max": float(sub["opp_concentration_hhi"].max()) if not sub.empty else 0.0,
            },
            "census_division_distribution": sub["census_division"].value_counts().sort_index().to_dict(),
            "poi_total": {
                "sum": float(sub["poi_total"].sum()) if "poi_total" in sub else 0.0,
                "min": float(sub["poi_total"].min()) if "poi_total" in sub and not sub.empty else 0.0,
                "median": float(sub["poi_total"].median()) if "poi_total" in sub and not sub.empty else 0.0,
                "max": float(sub["poi_total"].max()) if "poi_total" in sub and not sub.empty else 0.0,
            },
            "poi_nonzero_nodes": {
                "sum": int(sub["poi_nonzero_nodes"].sum()) if "poi_nonzero_nodes" in sub else 0,
                "min": int(sub["poi_nonzero_nodes"].min()) if "poi_nonzero_nodes" in sub and not sub.empty else 0,
                "median": float(sub["poi_nonzero_nodes"].median()) if "poi_nonzero_nodes" in sub and not sub.empty else 0.0,
                "max": int(sub["poi_nonzero_nodes"].max()) if "poi_nonzero_nodes" in sub and not sub.empty else 0,
            },
            "opp_group_distribution": sub["opp_group"].value_counts().sort_index().to_dict(),
        }
    payload = {
        "split_name": "OppSP",
        "seed": OPPSP_SEED,
        "fold_id": "OppSP",
        "stratification": stratification,
        "valid_counties": int(len(scores)),
        "quantiles": {
            "q_low": float(metadata["q_low"]),
            "q_high": float(metadata["q_high"]),
            "tail_quantile": float(metadata["tail_quantile"]),
        },
        "poi_columns": [int(col) for col in metadata["poi_columns"]],
        "checks": checks,
        "roles": roles,
        "warnings": {
            "skipped_counties": metadata.get("skipped_counties", []),
        },
        "strict_leakage_rule": {
            "uses_od_labels": False,
            "loaded_files": ["node_feat.npy", "region_meta.csv", "feature metadata"],
            "not_loaded_or_used": [
                "od.npy",
                "row sums",
                "column sums",
                "total OD flow",
                "allocation entropy from OD",
                "distance-flow correlation",
                "evaluation metrics",
                "model outputs",
            ],
        },
    }
    return summary_csv, payload


def generate_spatial_splits(
    data_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    poi_cols: str | Sequence[int] | None = None,
) -> pd.DataFrame:
    root = Path(data_root)
    split_dir = Path(output_dir) if output_dir is not None else root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_counties(root)
    split_json: dict[str, dict[str, list[str]]] = {}
    summaries: list[dict[str, Any]] = []
    geo_frames: list[pd.DataFrame] = []

    inventory.to_csv(split_dir / "county_inventory.csv", index=False)
    for seed in RANDSP_SEEDS:
        key = randsp_key(seed)
        roles, method = _randsp_roles(inventory, seed)
        rows = _split_rows(inventory, split_name="RandSP", seed=seed, fold_id=seed, roles=roles)
        rows.to_csv(split_dir / f"{key}.csv", index=False)
        split_json[key] = {
            role: rows.loc[rows["role"].eq(role), "area_id"].astype(str).tolist()
            for role in ("train", "val", "test")
        }
        summaries.append(_summary_row(rows, split_key=key, stratification=method))

    for division in sorted(inventory["census_division"].dropna().unique().tolist()):
        key = geosp_key(division)
        roles, method = _geosp_roles(inventory, division)
        rows = _split_rows(inventory, split_name="GeoSP", seed=GEOSP_VAL_SEED, fold_id=division, roles=roles)
        rows.to_csv(split_dir / f"{key}.csv", index=False)
        geo_frames.append(rows)
        split_json[key] = {
            role: rows.loc[rows["role"].eq(role), "area_id"].astype(str).tolist()
            for role in ("train", "val", "test")
        }
        summaries.append(_summary_row(rows, split_key=key, stratification=method))

    if geo_frames:
        pd.concat(geo_frames, ignore_index=True).to_csv(split_dir / "GeoSP_all_folds.csv", index=False)

    opp_scores, opp_metadata = _oppsp_scores(root, poi_cols=poi_cols)
    opp_roles, opp_method = _oppsp_roles(opp_scores)
    opp_rows = _oppsp_rows(opp_scores, opp_roles)
    opp_rows_for_summary = opp_rows.merge(
        opp_scores.loc[:, ["area_id", "poi_total", "poi_nonzero_nodes"]],
        on="area_id",
        how="left",
        validate="one_to_one",
    )
    opp_rows.to_csv(split_dir / "OppSP.csv", index=False)
    opp_scores.to_csv(split_dir / "OppSP_scores.csv", index=False)
    opp_summary, opp_summary_json = _oppsp_summary_payload(opp_rows_for_summary, opp_scores, metadata=opp_metadata, stratification=opp_method)
    opp_summary.to_csv(split_dir / "OppSP_summary.csv", index=False)
    (split_dir / "OppSP_summary.json").write_text(json.dumps(opp_summary_json, indent=2, sort_keys=True), encoding="utf-8")
    split_json[oppsp_key()] = {
        role: opp_rows.loc[opp_rows["role"].eq(role), "area_id"].astype(str).tolist()
        for role in ("train", "val", "test")
    }
    summaries.append(_summary_row(opp_rows, split_key=oppsp_key(), stratification=opp_method))

    summary = pd.DataFrame(summaries)
    summary.to_csv(split_dir / "split_summary.csv", index=False)
    (root / "splits.json").write_text(json.dumps(split_json, indent=2, sort_keys=True), encoding="utf-8")
    (root / "split_summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    (root / "split_balance.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    return summary


def load_split_keys(data_root: str | Path, splits_file: str | Path = "splits.json") -> list[str]:
    path = Path(splits_file)
    if not path.is_absolute() and not path.exists():
        path = Path(data_root) / path
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(str(key) for key in payload)


def expand_split_protocols(requested: list[str], available_keys: list[str]) -> list[str]:
    keys = list(available_keys)
    expanded: list[str] = []
    for split in requested:
        name = str(split)
        if name == "RandSP":
            expanded.extend(key for key in keys if key.startswith("RandSP_seed"))
        elif name == "GeoSP":
            expanded.extend(key for key in keys if key.startswith("GeoSP_"))
        elif name == "OppSP":
            expanded.extend(key for key in keys if key == "OppSP")
        else:
            expanded.append(name)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in expanded:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped
