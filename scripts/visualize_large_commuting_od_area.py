"""Visualize one LargeCommutingOD area with OD flows and spatial geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, Polygon


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

BLUE = {
    "xlight": "#EAF1FE",
    "light": "#CEDFFE",
    "base": "#A3BEFA",
    "mid": "#5477C4",
    "dark": "#2E4780",
}

ORANGE = {
    "xlight": "#FFEDDE",
    "light": "#FFBDA1",
    "base": "#F0986E",
    "mid": "#CC6F47",
    "dark": "#804126",
}

OLIVE = {
    "xlight": "#D8ECBD",
    "light": "#BEEB96",
    "base": "#A3D576",
    "mid": "#71B436",
    "dark": "#386411",
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


def _load_region_meta(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _geometry_rings(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    if geometry.get("type") == "Polygon":
        return [[(float(x), float(y)) for x, y in ring] for ring in geometry.get("coordinates", [])]
    if geometry.get("type") == "MultiPolygon":
        rings: list[list[tuple[float, float]]] = []
        for polygon in geometry.get("coordinates", []):
            rings.extend([[(float(x), float(y)) for x, y in ring] for ring in polygon])
        return rings
    return []


def _draw_boundaries(ax: plt.Axes, geojson: dict[str, Any], row_flow: np.ndarray) -> None:
    positive = row_flow[row_flow > 0]
    vmax = float(np.percentile(positive, 90)) if positive.size else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = colors.LinearSegmentedColormap.from_list("flow_fill", [TOKENS["panel"], BLUE["xlight"], BLUE["light"], BLUE["base"]])

    for feature in geojson["features"]:
        node_index = int(feature["properties"]["node_index"])
        fill = cmap(norm(float(row_flow[node_index])))
        rings = _geometry_rings(feature["geometry"])
        for ring_idx, ring in enumerate(rings):
            if len(ring) < 3:
                continue
            patch = Polygon(
                ring,
                closed=True,
                facecolor=fill if ring_idx == 0 else TOKENS["panel"],
                edgecolor="#9AA3B5",
                linewidth=0.55,
                alpha=0.88,
                zorder=1,
            )
            ax.add_patch(patch)


def _top_flows(y: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    flows: list[tuple[int, int, float]] = []
    n = y.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and y[i, j] > 0:
                flows.append((i, j, float(y[i, j])))
    return sorted(flows, key=lambda row: row[2], reverse=True)[:top_k]


def _draw_flow_map(
    ax: plt.Axes,
    geojson: dict[str, Any],
    centroids: np.ndarray,
    y: np.ndarray,
    top_k: int,
) -> None:
    outflow = y.sum(axis=1)
    _draw_boundaries(ax, geojson, outflow)

    top = _top_flows(y, top_k)
    max_flow = max((flow for _, _, flow in top), default=1.0)
    for origin, dest, flow in reversed(top):
        x0, y0 = centroids[origin]
        x1, y1 = centroids[dest]
        width = 0.45 + 3.2 * (flow / max_flow) ** 0.65
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8 + 8 * (flow / max_flow) ** 0.5,
            linewidth=width,
            color=ORANGE["mid"],
            alpha=0.42,
            connectionstyle="arc3,rad=0.08",
            zorder=3,
        )
        ax.add_patch(arrow)

    node_sizes = 18 + 120 * np.sqrt(outflow / max(outflow.max(), 1.0))
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=node_sizes,
        facecolor=OLIVE["base"],
        edgecolor=OLIVE["dark"],
        linewidth=0.7,
        alpha=0.88,
        zorder=4,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.grid(False)
    ax.set_title(f"Top {len(top)} off-diagonal OD flows", loc="left", fontsize=11, fontweight="semibold")
    ax.text(
        0.01,
        0.01,
        "Polygon fill and node size scale with origin outflow; arrows show largest off-diagonal OD counts.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )


def _draw_heatmap(ax: plt.Axes, y: np.ndarray) -> None:
    log_y = np.log1p(y)
    cmap = colors.LinearSegmentedColormap.from_list("od_heat", [TOKENS["panel"], BLUE["xlight"], BLUE["light"], BLUE["base"], BLUE["mid"]])
    image = ax.imshow(log_y, cmap=cmap, aspect="auto")
    ax.set_title("OD matrix intensity", loc="left", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Destination node index", fontsize=9)
    ax.set_ylabel("Origin node index", fontsize=9)
    ax.tick_params(axis="both", labelsize=7, length=0)
    cb = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log1p(count)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)


def _draw_ranked_nodes(ax: plt.Axes, region_meta: list[dict[str, Any]], y: np.ndarray, top_n: int) -> None:
    outflow = y.sum(axis=1)
    inflow = y.sum(axis=0)
    top_idx = np.argsort(outflow)[::-1][:top_n]
    labels = []
    for idx in top_idx:
        geoid = region_meta[idx].get("GEOID", str(idx))
        labels.append(f"{idx}: {geoid}")
    order = np.arange(len(top_idx))
    height = 0.34
    ax.barh(order + height / 2, outflow[top_idx], height=height, color=BLUE["base"], edgecolor=BLUE["dark"], linewidth=0.8, label="Outflow")
    ax.barh(order - height / 2, inflow[top_idx], height=height, color=ORANGE["base"], edgecolor=ORANGE["dark"], linewidth=0.8, label="Inflow")
    ax.set_yticks(order, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Commuters", fontsize=9)
    ax.set_title(f"Top {len(top_idx)} nodes by outflow", loc="left", fontsize=11, fontweight="semibold")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def visualize_area(data_root: Path, area_id: str, out_dir: Path, top_k: int = 35, top_nodes: int = 10) -> dict[str, Any]:
    area_dir = data_root / area_id
    if not area_dir.exists():
        raise FileNotFoundError(area_dir)
    y = np.load(area_dir / "y_raw.npy").astype(float)
    centroids = np.load(area_dir / "centroids.npy").astype(float)
    geojson = json.loads((area_dir / "geometry.geojson").read_text(encoding="utf-8"))
    region_meta = _load_region_meta(area_dir / "region_meta.csv")

    if y.shape[0] != centroids.shape[0] or y.shape[0] != len(region_meta):
        raise ValueError(f"area {area_id} has inconsistent node counts")

    total_flow = float(y.sum())
    diagonal_flow = float(np.trace(y))
    off_diagonal_flow = total_flow - diagonal_flow
    valid_top = min(top_k, int(np.count_nonzero(y - np.diag(np.diag(y)))))

    _set_theme()
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 0.82],
        left=0.055,
        right=0.975,
        bottom=0.07,
        top=0.86,
        hspace=0.28,
        wspace=0.22,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    ax_rank = fig.add_subplot(grid[1, 1])

    _draw_flow_map(ax_map, geojson, centroids, y, valid_top)
    _draw_heatmap(ax_heat, y)
    _draw_ranked_nodes(ax_rank, region_meta, y, top_nodes)

    title = f"Commuting OD structure for area {area_id}"
    subtitle = (
        f"{y.shape[0]} region nodes; total OD count {total_flow:,.0f}; "
        f"off-diagonal share {off_diagonal_flow / max(total_flow, 1.0):.1%}; "
        "source: converted LargeCommutingOD arrays and shapefile-derived geometry."
    )
    fig.text(0.055, 0.955, title, ha="left", va="top", fontsize=16, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.055, 0.918, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{area_id}_od_visualization.png"
    svg_path = out_dir / f"{area_id}_od_visualization.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    return {
        "area_id": area_id,
        "n_regions": int(y.shape[0]),
        "total_flow": total_flow,
        "diagonal_flow": diagonal_flow,
        "off_diagonal_flow": off_diagonal_flow,
        "top_flows_plotted": valid_top,
        "png": str(png_path.resolve()),
        "svg": str(svg_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/opflow_large_commuting_od"))
    parser.add_argument("--area-id", default="01003")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_world/od_visualizations"))
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--top-nodes", type=int, default=10)
    args = parser.parse_args()

    summary = visualize_area(args.data_root, args.area_id, args.out_dir, top_k=args.top_k, top_nodes=args.top_nodes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
