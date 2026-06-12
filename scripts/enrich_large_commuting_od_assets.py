"""Restore raw LargeCommutingOD feature parts and geography assets.

The training conversion keeps the arrays needed by the benchmark, but it
renames some official arrays and collapses demographics/POI features into
``node_feat.npy``. This utility enriches an existing converted root in place
from the original CommutingODGen zip without changing the training schema.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


DEFAULT_ZIP = Path("/Users/liuchangjian/Downloads/CommutingODGen-Dataset-main.zip")
ZIP_ROOT = "CommutingODGen-Dataset-main"
SHAPEFILE_EXTS = (".cpg", ".dbf", ".prj", ".shp", ".shx")


def _converted_area_ids(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if (path / "y_raw.npy").exists())


def _zip_exists(archive: ZipFile, name: str) -> bool:
    try:
        archive.getinfo(name)
        return True
    except KeyError:
        return False


def _load_npy_from_zip(archive: ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(archive.read(name)))


def _link_or_copy(src: Path, dst: Path, *, overwrite: bool = False) -> str:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "exists"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _decode_dbf_value(raw: bytes, field_type: str, decimals: int) -> Any:
    text = raw.decode("latin1", errors="replace").strip()
    if text == "":
        return ""
    if field_type in {"N", "F"}:
        try:
            return float(text) if decimals else int(text)
        except ValueError:
            return text
    if field_type == "L":
        return text.upper() in {"Y", "T", "1"}
    return text


def read_dbf(data: bytes) -> tuple[list[tuple[str, str, int, int]], list[dict[str, Any]]]:
    """Read a dBASE table from shapefile DBF bytes."""
    if len(data) < 32:
        raise ValueError("DBF file is too small")
    n_records = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]
    fields: list[tuple[str, str, int, int]] = []
    pos = 32
    while pos + 32 <= header_len and data[pos] != 0x0D:
        raw_name = data[pos : pos + 11].split(b"\x00", 1)[0]
        name = raw_name.decode("latin1", errors="replace").strip()
        field_type = chr(data[pos + 11])
        length = int(data[pos + 16])
        decimals = int(data[pos + 17])
        fields.append((name, field_type, length, decimals))
        pos += 32

    rows: list[dict[str, Any]] = []
    pos = header_len
    for _ in range(n_records):
        record = data[pos : pos + record_len]
        pos += record_len
        if not record or record[:1] == b"*":
            continue
        cursor = 1
        row: dict[str, Any] = {}
        for name, field_type, length, decimals in fields:
            row[name] = _decode_dbf_value(record[cursor : cursor + length], field_type, decimals)
            cursor += length
        rows.append(row)
    return fields, rows


def read_shp(data: bytes) -> dict[str, Any]:
    """Read enough of an ESRI shapefile for polygon metadata and GeoJSON."""
    if len(data) < 100:
        raise ValueError("SHP file is too small")
    file_code = struct.unpack(">i", data[0:4])[0]
    version, shape_type = struct.unpack("<2i", data[28:36])
    bbox = struct.unpack("<4d", data[36:68])
    records: list[dict[str, Any]] = []
    pos = 100
    while pos + 8 <= len(data):
        record_number, content_words = struct.unpack(">2i", data[pos : pos + 8])
        pos += 8
        end = pos + content_words * 2
        if end > len(data):
            raise ValueError(f"Malformed SHP record {record_number}")
        record_type = struct.unpack("<i", data[pos : pos + 4])[0]
        if record_type == 0:
            records.append({"shape_type": record_type, "parts": [], "bbox": None, "record_number": record_number})
        elif record_type in {3, 5, 13, 15, 23, 25, 31}:
            record_bbox = struct.unpack("<4d", data[pos + 4 : pos + 36])
            n_parts, n_points = struct.unpack("<2i", data[pos + 36 : pos + 44])
            parts = list(struct.unpack("<" + "i" * n_parts, data[pos + 44 : pos + 44 + 4 * n_parts])) if n_parts else []
            point_pos = pos + 44 + 4 * n_parts
            points = [
                struct.unpack("<2d", data[point_pos + idx * 16 : point_pos + idx * 16 + 16])
                for idx in range(n_points)
            ]
            rings = []
            for idx, start in enumerate(parts):
                stop = parts[idx + 1] if idx + 1 < len(parts) else len(points)
                rings.append(points[start:stop])
            records.append(
                {
                    "shape_type": record_type,
                    "parts": rings,
                    "bbox": record_bbox,
                    "record_number": record_number,
                }
            )
        elif record_type == 1:
            x, y = struct.unpack("<2d", data[pos + 4 : pos + 20])
            records.append({"shape_type": record_type, "parts": [[(x, y)]], "bbox": (x, y, x, y), "record_number": record_number})
        else:
            records.append(
                {
                    "shape_type": record_type,
                    "parts": [],
                    "bbox": None,
                    "record_number": record_number,
                    "unsupported": True,
                }
            )
        pos = end
    if file_code != 9994:
        raise ValueError(f"Unexpected shapefile code: {file_code}")
    return {"version": version, "shape_type": shape_type, "bbox": bbox, "records": records}


def _ring_area_and_centroid(ring: list[tuple[float, float]]) -> tuple[float, float, float]:
    if len(ring) < 3:
        if not ring:
            return 0.0, float("nan"), float("nan")
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))
    area2 = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    for idx, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(idx + 1) % len(ring)]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
    area = area2 / 2.0
    if abs(area) <= 1e-18:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))
    return area, cx_acc / (6.0 * area), cy_acc / (6.0 * area)


def _shape_centroid(shape: dict[str, Any]) -> tuple[float, float]:
    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0
    fallback_points: list[tuple[float, float]] = []
    for ring in shape.get("parts", []):
        fallback_points.extend(ring)
        area, cx, cy = _ring_area_and_centroid(ring)
        weight = abs(area)
        if weight > 0 and np.isfinite(cx) and np.isfinite(cy):
            weighted_x += cx * weight
            weighted_y += cy * weight
            total_weight += weight
    if total_weight > 0:
        return weighted_x / total_weight, weighted_y / total_weight
    if fallback_points:
        xs = [point[0] for point in fallback_points]
        ys = [point[1] for point in fallback_points]
        return float(np.mean(xs)), float(np.mean(ys))
    bbox = shape.get("bbox")
    if bbox:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return float("nan"), float("nan")


def _shape_to_geojson(shape: dict[str, Any]) -> dict[str, Any] | None:
    parts = shape.get("parts", [])
    if not parts:
        return None
    if shape.get("shape_type") == 1:
        x, y = parts[0][0]
        return {"type": "Point", "coordinates": [x, y]}
    polygons = []
    for ring in parts:
        coords = [[float(x), float(y)] for x, y in ring]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        if coords:
            polygons.append([coords])
    if not polygons:
        return None
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def _write_region_assets(area_dir: Path, fields: list[tuple[str, str, int, int]], rows: list[dict[str, Any]], shp: dict[str, Any]) -> dict[str, Any]:
    shapes = shp["records"]
    centroids = np.asarray([_shape_centroid(shape) for shape in shapes], dtype=np.float64)
    np.save(area_dir / "centroids.npy", centroids)

    fieldnames = ["node_index"] + [field[0] for field in fields] + [
        "centroid_lon",
        "centroid_lat",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
        "n_parts",
        "n_points",
        "shape_type",
    ]
    with (area_dir / "region_meta.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, shape in enumerate(shapes):
            row = {"node_index": idx}
            if idx < len(rows):
                row.update(rows[idx])
            centroid = centroids[idx] if idx < len(centroids) else (float("nan"), float("nan"))
            bbox = shape.get("bbox") or (float("nan"), float("nan"), float("nan"), float("nan"))
            row.update(
                {
                    "centroid_lon": float(centroid[0]),
                    "centroid_lat": float(centroid[1]),
                    "bbox_min_lon": float(bbox[0]),
                    "bbox_min_lat": float(bbox[1]),
                    "bbox_max_lon": float(bbox[2]),
                    "bbox_max_lat": float(bbox[3]),
                    "n_parts": len(shape.get("parts", [])),
                    "n_points": sum(len(part) for part in shape.get("parts", [])),
                    "shape_type": shape.get("shape_type"),
                }
            )
            writer.writerow(row)

    features = []
    for idx, shape in enumerate(shapes):
        properties = {"node_index": idx}
        if idx < len(rows):
            properties.update(rows[idx])
        geometry = _shape_to_geojson(shape)
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})
    geojson = {
        "type": "FeatureCollection",
        "name": area_dir.name,
        "crs": {"type": "name", "properties": {"name": "EPSG:4269"}},
        "features": features,
    }
    (area_dir / "geometry.geojson").write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    return {
        "n_dbf_records": len(rows),
        "n_shapes": len(shapes),
        "shape_type": shp.get("shape_type"),
        "bbox": list(shp.get("bbox", [])),
    }


def enrich_area(
    archive: ZipFile,
    area_id: str,
    out_root: Path,
    *,
    extract_shapefile: bool,
    write_geojson: bool,
    write_feature_parts: bool,
    write_official_aliases: bool,
    overwrite: bool,
) -> dict[str, Any]:
    area_dir = out_root / area_id
    row: dict[str, Any] = {"area_id": area_id, "status": "ok"}
    y_path = area_dir / "y_raw.npy"
    n_regions = int(np.load(y_path, mmap_mode="r").shape[0])
    row["n_regions"] = n_regions

    if write_feature_parts:
        for raw_name, out_name in (("demos.npy", "demos.npy"), ("pois.npy", "pois.npy")):
            zip_name = f"{ZIP_ROOT}/data/{area_id}/{raw_name}"
            if _zip_exists(archive, zip_name):
                arr = _load_npy_from_zip(archive, zip_name)
                np.save(area_dir / out_name, arr.astype(np.float32, copy=False))
                row[out_name.replace(".npy", "_shape")] = list(arr.shape)
            else:
                row[f"missing_{raw_name}"] = True

    if write_official_aliases:
        alias_status = {}
        for src_name, dst_name in (("y_raw.npy", "od.npy"), ("distance.npy", "dis.npy"), ("adjacency.npy", "adj.npy")):
            src = area_dir / src_name
            dst = area_dir / dst_name
            if src.exists():
                alias_status[dst_name] = _link_or_copy(src, dst, overwrite=overwrite)
        row["official_aliases"] = alias_status

    shp_prefix = f"{ZIP_ROOT}/assets/Boundaries_Regions_within_Areas/{area_id}/{area_id}"
    dbf_name = shp_prefix + ".dbf"
    shp_name = shp_prefix + ".shp"
    has_geometry = _zip_exists(archive, dbf_name) and _zip_exists(archive, shp_name)
    row["has_geometry_source"] = has_geometry
    if extract_shapefile:
        shp_dir = area_dir / "shapefile"
        shp_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        for ext in SHAPEFILE_EXTS:
            zip_name = shp_prefix + ext
            if _zip_exists(archive, zip_name):
                target = shp_dir / f"{area_id}{ext}"
                if overwrite or not target.exists():
                    target.write_bytes(archive.read(zip_name))
                extracted.append(target.name)
        row["shapefile_files"] = extracted

    if write_geojson and has_geometry:
        fields, records = read_dbf(archive.read(dbf_name))
        shp = read_shp(archive.read(shp_name))
        geometry_summary = _write_region_assets(area_dir, fields, records, shp)
        row.update(geometry_summary)
        row["geometry_matches_matrix"] = len(records) == len(shp["records"]) == n_regions
    elif write_geojson:
        row["geometry_matches_matrix"] = False
    return row


def enrich_dataset(
    zip_path: Path,
    out_root: Path,
    *,
    areas: list[str] | None = None,
    extract_shapefile: bool = True,
    write_geojson: bool = True,
    write_feature_parts: bool = True,
    write_official_aliases: bool = True,
    overwrite: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    selected = areas or _converted_area_ids(out_root)
    rows = []
    with ZipFile(zip_path) as archive:
        for idx, area_id in enumerate(selected, start=1):
            try:
                rows.append(
                    enrich_area(
                        archive,
                        area_id,
                        out_root,
                        extract_shapefile=extract_shapefile,
                        write_geojson=write_geojson,
                        write_feature_parts=write_feature_parts,
                        write_official_aliases=write_official_aliases,
                        overwrite=overwrite,
                    )
                )
            except Exception as exc:  # Keep one bad area from hiding the rest.
                rows.append({"area_id": area_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            if idx % 250 == 0:
                print(f"enriched {idx}/{len(selected)} areas", flush=True)

    report = {
        "zip_path": str(zip_path),
        "out_root": str(out_root),
        "n_requested": len(selected),
        "n_ok": sum(row.get("status") == "ok" for row in rows),
        "n_failed": sum(row.get("status") == "failed" for row in rows),
        "n_geometry_match": sum(bool(row.get("geometry_matches_matrix")) for row in rows),
        "rows": rows,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP))
    parser.add_argument("--out-root", default="data/opflow_large_commuting_od")
    parser.add_argument("--area", action="append", default=None, help="Limit to one area; may be repeated.")
    parser.add_argument("--report", default="data/opflow_large_commuting_od/asset_enrichment_report.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-shapefile", action="store_true")
    parser.add_argument("--no-geojson", action="store_true")
    parser.add_argument("--no-feature-parts", action="store_true")
    parser.add_argument("--no-official-aliases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = enrich_dataset(
        Path(args.zip_path),
        Path(args.out_root),
        areas=args.area,
        extract_shapefile=not args.no_shapefile,
        write_geojson=not args.no_geojson,
        write_feature_parts=not args.no_feature_parts,
        write_official_aliases=not args.no_official_aliases,
        overwrite=args.overwrite,
        report_path=Path(args.report),
    )
    print(
        f"enrichment complete: ok={report['n_ok']} failed={report['n_failed']} "
        f"geometry_match={report['n_geometry_match']}/{report['n_requested']} report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
