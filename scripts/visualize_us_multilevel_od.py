"""Build national multi-level OD and origin-scale visualizations.

LargeCommutingOD stores OD matrices inside each area. This script visualizes
those within-area flows at national scale, with state/county aggregation and a
node-level origin production scale surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"}
OLIVE = {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"}
PINK = {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"}
NEUTRAL = {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"}

STATE_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

DISPLAY_BOUNDS = {
    "main": (-125.0, -66.0, 23.4, 50.2),
    "02": (-124.5, -112.6, 24.2, 31.2),
    "15": (-109.7, -104.0, 24.0, 28.2),
}


def _set_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"],
        }
    )


def _read_region_meta(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_display_xy(state: str, lon: float, lat: float, state_bounds: dict[str, tuple[float, float, float, float]]) -> tuple[float, float]:
    if state not in {"02", "15"}:
        return lon, lat
    src_min_x, src_max_x, src_min_y, src_max_y = state_bounds[state]
    dst_min_x, dst_max_x, dst_min_y, dst_max_y = DISPLAY_BOUNDS[state]
    x = dst_min_x + (lon - src_min_x) / max(src_max_x - src_min_x, 1e-9) * (dst_max_x - dst_min_x)
    y = dst_min_y + (lat - src_min_y) / max(src_max_y - src_min_y, 1e-9) * (dst_max_y - dst_min_y)
    return x, y


def _state_bounds_from_points(raw_points: list[tuple[str, float, float]]) -> dict[str, tuple[float, float, float, float]]:
    bounds: dict[str, list[float]] = {}
    for state, lon, lat in raw_points:
        if state not in {"02", "15"}:
            continue
        bounds.setdefault(state, [math.inf, -math.inf, math.inf, -math.inf])
        item = bounds[state]
        item[0] = min(item[0], lon)
        item[1] = max(item[1], lon)
        item[2] = min(item[2], lat)
        item[3] = max(item[3], lat)
    return {state: tuple(values) for state, values in bounds.items()}


def _weighted_average(points: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))
    return float(np.sum(points[:, 0] * weights) / total), float(np.sum(points[:, 1] * weights) / total)


def _top_offdiag_flows(y: np.ndarray, limit: int) -> list[tuple[int, int, float]]:
    if y.shape[0] <= 1 or limit <= 0:
        return []
    off = y.astype(float, copy=True)
    np.fill_diagonal(off, 0.0)
    positive = int(np.count_nonzero(off > 0))
    if positive == 0:
        return []
    k = min(limit, positive)
    flat = off.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    rows: list[tuple[int, int, float]] = []
    n = y.shape[1]
    for flat_idx in idx:
        flow = float(flat[flat_idx])
        if flow <= 0:
            continue
        origin = int(flat_idx // n)
        dest = int(flat_idx % n)
        rows.append((origin, dest, flow))
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    radius = radius if radius is not None else max(2, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    pad = len(kernel) // 2
    padded_x = np.pad(grid, ((0, 0), (pad, pad)), mode="edge")
    smoothed_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x)
    padded_y = np.pad(smoothed_x, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_y)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_data(data_root: Path, per_area_flow_cap: int, top_flows: int) -> dict[str, Any]:
    area_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and (path / "y_raw.npy").exists())
    raw_points: list[tuple[str, float, float]] = []
    for area_dir in area_dirs:
        state = area_dir.name[:2]
        centroids = np.load(area_dir / "centroids.npy")
        raw_points.extend((state, float(lon), float(lat)) for lon, lat in centroids)
    state_bounds = _state_bounds_from_points(raw_points)

    node_rows: list[dict[str, Any]] = []
    county_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    state_acc: dict[str, dict[str, Any]] = {}
    state_hull_points: dict[str, list[tuple[float, float]]] = {}

    for area_dir in area_dirs:
        area_id = area_dir.name
        state = area_id[:2]
        y = np.load(area_dir / "y_raw.npy").astype(float)
        centroids = np.load(area_dir / "centroids.npy").astype(float)
        region_meta = _read_region_meta(area_dir / "region_meta.csv")
        outflow = y.sum(axis=1)
        inflow = y.sum(axis=0)
        total_flow = float(y.sum())
        diagonal_flow = float(np.trace(y))
        offdiag_flow = total_flow - diagonal_flow

        node_display = np.array([_state_display_xy(state, float(lon), float(lat), state_bounds) for lon, lat in centroids])
        county_lon, county_lat = _weighted_average(centroids, outflow)
        county_x, county_y = _state_display_xy(state, county_lon, county_lat, state_bounds)

        county_rows.append(
            {
                "area_id": area_id,
                "state_fips": state,
                "state": STATE_ABBR.get(state, state),
                "n_regions": int(y.shape[0]),
                "total_origin_scale": total_flow,
                "diagonal_flow": diagonal_flow,
                "offdiag_flow": offdiag_flow,
                "weighted_lon": county_lon,
                "weighted_lat": county_lat,
                "display_x": county_x,
                "display_y": county_y,
            }
        )
        state_hull_points.setdefault(state, []).append((county_x, county_y))
        acc = state_acc.setdefault(
            state,
            {
                "state_fips": state,
                "state": STATE_ABBR.get(state, state),
                "n_areas": 0,
                "n_regions": 0,
                "total_origin_scale": 0.0,
                "diagonal_flow": 0.0,
                "offdiag_flow": 0.0,
                "sum_xw": 0.0,
                "sum_yw": 0.0,
            },
        )
        acc["n_areas"] += 1
        acc["n_regions"] += int(y.shape[0])
        acc["total_origin_scale"] += total_flow
        acc["diagonal_flow"] += diagonal_flow
        acc["offdiag_flow"] += offdiag_flow
        acc["sum_xw"] += county_x * total_flow
        acc["sum_yw"] += county_y * total_flow

        for idx, (point, raw_point) in enumerate(zip(node_display, centroids)):
            node_rows.append(
                {
                    "area_id": area_id,
                    "state_fips": state,
                    "state": STATE_ABBR.get(state, state),
                    "node_index": idx,
                    "geoid": region_meta[idx].get("GEOID", ""),
                    "lon": float(raw_point[0]),
                    "lat": float(raw_point[1]),
                    "display_x": float(point[0]),
                    "display_y": float(point[1]),
                    "origin_scale": float(outflow[idx]),
                    "inflow": float(inflow[idx]),
                }
            )

        for origin, dest, flow in _top_offdiag_flows(y, per_area_flow_cap):
            flow_rows.append(
                {
                    "area_id": area_id,
                    "state_fips": state,
                    "state": STATE_ABBR.get(state, state),
                    "origin_node_index": origin,
                    "dest_node_index": dest,
                    "origin_geoid": region_meta[origin].get("GEOID", ""),
                    "dest_geoid": region_meta[dest].get("GEOID", ""),
                    "origin_lon": float(centroids[origin, 0]),
                    "origin_lat": float(centroids[origin, 1]),
                    "dest_lon": float(centroids[dest, 0]),
                    "dest_lat": float(centroids[dest, 1]),
                    "origin_x": float(node_display[origin, 0]),
                    "origin_y": float(node_display[origin, 1]),
                    "dest_x": float(node_display[dest, 0]),
                    "dest_y": float(node_display[dest, 1]),
                    "flow": flow,
                }
            )

    for row in state_acc.values():
        total = max(float(row["total_origin_scale"]), 1e-9)
        row["display_x"] = float(row.pop("sum_xw") / total)
        row["display_y"] = float(row.pop("sum_yw") / total)

    state_rows = sorted(state_acc.values(), key=lambda row: row["total_origin_scale"], reverse=True)
    county_rows.sort(key=lambda row: row["total_origin_scale"], reverse=True)
    flow_rows.sort(key=lambda row: row["flow"], reverse=True)
    return {
        "state_rows": state_rows,
        "county_rows": county_rows,
        "node_rows": node_rows,
        "flow_rows": flow_rows[:top_flows],
        "state_hulls": {state: _convex_hull(points) for state, points in state_hull_points.items()},
        "state_bounds": state_bounds,
    }


def build_scale_grid(
    node_rows: list[dict[str, Any]],
    nx: int = 900,
    ny: int = 460,
    sigma: float = 4.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = DISPLAY_BOUNDS["main"]
    x = np.array([float(row["display_x"]) for row in node_rows])
    y = np.array([float(row["display_y"]) for row in node_rows])
    weights = np.log1p(np.array([float(row["origin_scale"]) for row in node_rows]))
    grid, y_edges, x_edges = np.histogram2d(y, x, bins=(ny, nx), range=((ymin, ymax), (xmin, xmax)), weights=weights)
    smooth = _smooth_grid(grid, sigma=sigma)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return x_centers, y_centers, smooth


def _scale_cmap() -> colors.LinearSegmentedColormap:
    return colors.LinearSegmentedColormap.from_list(
        "origin_scale_peak",
        [TOKENS["panel"], BLUE["xlight"], BLUE["light"], GOLD["light"], ORANGE["base"], PINK["mid"]],
    )


def _draw_insets(ax: plt.Axes) -> None:
    for state, label in (("02", "AK inset"), ("15", "HI inset")):
        x0, x1, y0, y1 = DISPLAY_BOUNDS[state]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=NEUTRAL["mid"], linewidth=0.9, linestyle="--"))
        ax.text(x0 + 0.25, y1 - 0.35, label, ha="left", va="top", fontsize=8, color=TOKENS["muted"])


def _format_map_axis(ax: plt.Axes) -> None:
    xmin, xmax, ymin, ymax = DISPLAY_BOUNDS["main"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Display longitude; AK and HI are inset-transformed", fontsize=9)
    ax.set_ylabel("Latitude / display latitude", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.grid(color=TOKENS["grid"], linewidth=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_multilevel_map(data: dict[str, Any], out_dir: Path, png_name: str, svg_name: str) -> tuple[Path, Path]:
    state_rows = data["state_rows"]
    county_rows = data["county_rows"]
    node_rows = data["node_rows"]
    flow_rows = data["flow_rows"]
    state_hulls = data["state_hulls"]
    x_grid, y_grid, z = build_scale_grid(node_rows)
    z_positive = z[z > 0]
    vmax = float(np.percentile(z_positive, 99.3)) if z_positive.size else 1.0
    norm = colors.PowerNorm(gamma=0.62, vmin=0.0, vmax=max(vmax, 1.0))

    _set_theme()
    fig = plt.figure(figsize=(18, 12), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.6, 0.82],
        height_ratios=[1.0, 0.84],
        left=0.055,
        right=0.965,
        bottom=0.06,
        top=0.875,
        hspace=0.28,
        wspace=0.18,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_state = fig.add_subplot(gs[0, 1])
    ax_county = fig.add_subplot(gs[1, 1])

    extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
    image = ax_map.imshow(z, extent=extent, origin="lower", cmap=_scale_cmap(), norm=norm, interpolation="bilinear", alpha=0.88, zorder=1)

    for state, hull in state_hulls.items():
        if len(hull) < 3:
            continue
        xs = [p[0] for p in hull] + [hull[0][0]]
        ys = [p[1] for p in hull] + [hull[0][1]]
        ax_map.plot(xs, ys, color=NEUTRAL["mid"], linewidth=0.65, alpha=0.45, zorder=2)

    county_scale = np.array([float(row["total_origin_scale"]) for row in county_rows])
    county_x = np.array([float(row["display_x"]) for row in county_rows])
    county_y = np.array([float(row["display_y"]) for row in county_rows])
    county_sizes = 5 + 80 * np.sqrt(np.log1p(county_scale) / max(np.log1p(county_scale).max(), 1e-9))
    ax_map.scatter(
        county_x,
        county_y,
        s=county_sizes,
        facecolors=BLUE["base"],
        edgecolors=BLUE["dark"],
        linewidths=0.35,
        alpha=0.38,
        zorder=3,
    )

    flow_values = np.array([float(row["flow"]) for row in flow_rows])
    if len(flow_rows):
        segments = [[(float(row["origin_x"]), float(row["origin_y"])), (float(row["dest_x"]), float(row["dest_y"]))] for row in flow_rows]
        widths = 0.15 + 1.9 * np.sqrt(np.log1p(flow_values) / max(np.log1p(flow_values).max(), 1e-9))
        line_collection = LineCollection(segments, colors=ORANGE["mid"], linewidths=widths, alpha=0.28, zorder=4)
        ax_map.add_collection(line_collection)
        for row in flow_rows[:120]:
            flow = float(row["flow"])
            scale = math.sqrt(math.log1p(flow) / max(math.log1p(float(flow_values.max())), 1e-9))
            arrow = FancyArrowPatch(
                (float(row["origin_x"]), float(row["origin_y"])),
                (float(row["dest_x"]), float(row["dest_y"])),
                arrowstyle="-|>",
                mutation_scale=4.0 + 6.0 * scale,
                linewidth=0.35 + 1.3 * scale,
                color=ORANGE["dark"],
                alpha=0.32,
                connectionstyle="arc3,rad=0.08",
                zorder=5,
            )
            ax_map.add_patch(arrow)

    state_scale = np.array([float(row["total_origin_scale"]) for row in state_rows])
    state_x = np.array([float(row["display_x"]) for row in state_rows])
    state_y = np.array([float(row["display_y"]) for row in state_rows])
    state_sizes = 160 + 950 * np.sqrt(state_scale / max(state_scale.max(), 1e-9))
    ax_map.scatter(state_x, state_y, s=state_sizes, facecolors="none", edgecolors=PINK["dark"], linewidths=1.15, alpha=0.7, zorder=6)
    for row in state_rows[:10]:
        ax_map.text(
            float(row["display_x"]),
            float(row["display_y"]),
            str(row["state"]),
            ha="center",
            va="center",
            fontsize=7.5,
            color=TOKENS["ink"],
            fontweight="semibold",
            zorder=7,
        )

    _draw_insets(ax_map)
    _format_map_axis(ax_map)
    ax_map.set_title("National view: origin-scale peak field, county-scale points, and strongest within-area OD flows", loc="left", fontsize=12, fontweight="semibold")
    cb = fig.colorbar(image, ax=ax_map, fraction=0.025, pad=0.012)
    cb.set_label("Smoothed log1p(origin scale)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)

    top_states = list(reversed(state_rows[:12]))
    ax_state.barh(
        [row["state"] for row in top_states],
        [row["total_origin_scale"] for row in top_states],
        color=PINK["base"],
        edgecolor=PINK["dark"],
        linewidth=0.8,
    )
    ax_state.set_title("Top states by total origin scale", loc="left", fontsize=11, fontweight="semibold")
    ax_state.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / 1e6:.1f}M"))
    ax_state.tick_params(axis="both", labelsize=8, length=0)
    ax_state.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax_state.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax_state.spines[side].set_visible(False)

    top_counties = list(reversed(county_rows[:14]))
    labels = [f"{row['area_id']} ({row['state']})" for row in top_counties]
    ax_county.barh(
        labels,
        [row["total_origin_scale"] for row in top_counties],
        color=BLUE["base"],
        edgecolor=BLUE["dark"],
        linewidth=0.8,
    )
    ax_county.set_title("Top counties/areas by total origin scale", loc="left", fontsize=11, fontweight="semibold")
    ax_county.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value / 1e3:.0f}K"))
    ax_county.tick_params(axis="both", labelsize=8, length=0)
    ax_county.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax_county.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax_county.spines[side].set_visible(False)

    total_flow = sum(float(row["total_origin_scale"]) for row in county_rows)
    subtitle = (
        f"{len(county_rows):,} areas, {len(node_rows):,} origin nodes, total within-area OD count {total_flow:,.0f}. "
        "Flows are within each area only; this does not contain cross-county or cross-state OD pairs."
    )
    fig.text(0.055, 0.965, "LargeCommutingOD multi-level OD and origin-scale map", ha="left", va="top", fontsize=17, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.055, 0.928, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / png_name
    svg_path = out_dir / svg_name
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def draw_peak_surface_3d(data: dict[str, Any], out_dir: Path, png_name: str, svg_name: str) -> tuple[Path, Path]:
    node_rows = [row for row in data["node_rows"] if row["state_fips"] not in {"02", "15"}]
    x_grid, y_grid, z = build_scale_grid(node_rows, nx=420, ny=230, sigma=3.0)
    z_positive = z[z > 0]
    vmax = float(np.percentile(z_positive, 99.5)) if z_positive.size else 1.0
    z_plot = np.clip(z / max(vmax, 1e-9), 0.0, 1.2)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    _set_theme()
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection="3d")
    stride_x = max(1, int(x_mesh.shape[1] / 180))
    stride_y = max(1, int(x_mesh.shape[0] / 110))
    surface = ax.plot_surface(
        x_mesh,
        y_mesh,
        z_plot,
        rstride=stride_y,
        cstride=stride_x,
        cmap=_scale_cmap(),
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        shade=True,
    )
    ax.view_init(elev=45, azim=-65)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_zlim(0, max(1.0, float(z_plot.max()) * 1.05))
    ax.set_xlabel("Longitude", labelpad=9)
    ax.set_ylabel("Latitude", labelpad=9)
    ax.set_zlabel("Relative log origin scale", labelpad=9)
    ax.set_title("Origin-scale peak surface across the contiguous United States", pad=20, fontsize=14, fontweight="semibold")
    ax.text2D(
        0.06,
        0.93,
        "Each peak comes from node-level origin production scale, smoothed for a national surface; color and height encode relative scale.",
        transform=ax.transAxes,
        fontsize=9,
        color=TOKENS["muted"],
    )
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.grid(False)
    cb = fig.colorbar(surface, ax=ax, shrink=0.58, pad=0.02)
    cb.set_label("Relative log origin scale", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / png_name
    svg_path = out_dir / svg_name
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def write_outputs(data: dict[str, Any], out_dir: Path) -> dict[str, str]:
    _write_csv(
        out_dir / "state_origin_scale.csv",
        data["state_rows"],
        ["state_fips", "state", "n_areas", "n_regions", "total_origin_scale", "diagonal_flow", "offdiag_flow", "display_x", "display_y"],
    )
    _write_csv(
        out_dir / "county_origin_scale.csv",
        data["county_rows"],
        [
            "area_id",
            "state_fips",
            "state",
            "n_regions",
            "total_origin_scale",
            "diagonal_flow",
            "offdiag_flow",
            "weighted_lon",
            "weighted_lat",
            "display_x",
            "display_y",
        ],
    )
    _write_csv(
        out_dir / "origin_scale_points.csv",
        data["node_rows"],
        ["area_id", "state_fips", "state", "node_index", "geoid", "lon", "lat", "display_x", "display_y", "origin_scale", "inflow"],
    )
    _write_csv(
        out_dir / "top_internal_od_flows.csv",
        data["flow_rows"],
        [
            "area_id",
            "state_fips",
            "state",
            "origin_node_index",
            "dest_node_index",
            "origin_geoid",
            "dest_geoid",
            "origin_lon",
            "origin_lat",
            "dest_lon",
            "dest_lat",
            "origin_x",
            "origin_y",
            "dest_x",
            "dest_y",
            "flow",
        ],
    )
    map_png, map_svg = draw_multilevel_map(data, out_dir, "us_multilevel_od_map.png", "us_multilevel_od_map.svg")
    surface_png, surface_svg = draw_peak_surface_3d(data, out_dir, "us_origin_scale_peak_surface_3d.png", "us_origin_scale_peak_surface_3d.svg")
    summary = {
        "n_states": len(data["state_rows"]),
        "n_areas": len(data["county_rows"]),
        "n_origin_nodes": len(data["node_rows"]),
        "n_top_flows": len(data["flow_rows"]),
        "total_origin_scale": sum(float(row["total_origin_scale"]) for row in data["county_rows"]),
        "map_png": str(map_png.resolve()),
        "map_svg": str(map_svg.resolve()),
        "surface_png": str(surface_png.resolve()),
        "surface_svg": str(surface_svg.resolve()),
    }
    (out_dir / "us_multilevel_od_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/opflow_large_commuting_od"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_world/us_multilevel_od"))
    parser.add_argument("--per-area-flow-cap", type=int, default=4)
    parser.add_argument("--top-flows", type=int, default=900)
    args = parser.parse_args()

    data = collect_data(args.data_root, per_area_flow_cap=args.per_area_flow_cap, top_flows=args.top_flows)
    summary = write_outputs(data, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
