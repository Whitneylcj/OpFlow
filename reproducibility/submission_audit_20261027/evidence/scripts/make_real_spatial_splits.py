"""Generate RandSP, GeoSP, and OppSP county-level real-world split files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opflow_benchmark.real_spatial_splits import generate_spatial_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/opflow_large_commuting_od")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--poi-cols",
        default=None,
        help="Comma-separated node_feat column indices for raw nonnegative POI counts; used only if metadata cannot identify POI columns.",
    )
    args = parser.parse_args()
    summary = generate_spatial_splits(args.data_root, output_dir=args.output_dir, poi_cols=args.poi_cols)
    display_cols = [
        "split_key",
        "split_name",
        "fold_id",
        "train_counties",
        "val_counties",
        "test_counties",
        "train_tract_nodes",
        "val_tract_nodes",
        "test_tract_nodes",
        "stratification",
    ]
    print(summary[[col for col in display_cols if col in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
