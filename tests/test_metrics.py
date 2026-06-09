"""Unit tests for OpFlow benchmark metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from opflow_experiments.core.outputs import STANDARD_METRIC_COLUMNS, write_standard_outputs
from opflow_experiments.data.real_world_registry import get_real_world_dataset
from opflow_experiments.utils.config import deep_merge, parse_cli_overrides
from opflow_benchmark.metrics import (
    evaluate_area,
    global_cpc,
    matrix_to_share,
    rank_corr,
    row_cpc,
    row_kl,
    topk_overlap_with_coverage,
)


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        y = np.array([[0, 2, 1], [3, 0, 1]], dtype=float)
        p, _, valid = matrix_to_share(y)
        self.assertLess(row_kl(p, p, valid), 1e-7)
        self.assertGreater(row_cpc(p, p, valid), 1 - 1e-7)

    def test_global_cpc_identity(self) -> None:
        y = np.array([[1, 2], [3, 4]], dtype=float)
        self.assertAlmostEqual(global_cpc(y, y), 1.0)

    def test_topk_excludes_small_rows(self) -> None:
        p = np.array([[0.5, 0.5], [0.2, 0.3]], dtype=float)
        out = topk_overlap_with_coverage(p, p, k=3)
        self.assertTrue(math.isnan(out["value"]))
        self.assertEqual(out["coverage"], 0.0)

    def test_row_normalization(self) -> None:
        y = np.array([[0, 2, 3], [0, 0, 0], [1, 1, 0]], dtype=float)
        p, _, valid = matrix_to_share(y)
        self.assertTrue(np.allclose(p[valid].sum(axis=1), 1.0))

    def test_sparse_reasonable_no_nan_core(self) -> None:
        y = np.array([[0, 2, 0, 5], [0, 0, 0, 0], [1, 0, 3, 0], [0, 4, 0, 1]], dtype=float)
        p, _, valid = matrix_to_share(y)
        metrics = evaluate_area(y, y + 0.1, p_true=p, valid_origin=valid)
        for key in ["rmse", "mae", "log_rmse", "global_cpc", "row_kl", "row_cpc", "row_rmse", "row_mae"]:
            self.assertFalse(math.isnan(metrics[key]), key)

    def test_rank_corr_skips_constant_rows(self) -> None:
        p_true = np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.2, 0.1, 0.0]], dtype=float)
        p_pred = np.array([[0.4, 0.3, 0.2, 0.1], [0.6, 0.3, 0.1, 0.0]], dtype=float)
        valid = np.array([True, True])
        self.assertTrue(math.isfinite(rank_corr(p_true, p_pred, valid)))
        self.assertTrue(math.isnan(rank_corr(p_true[:1], p_pred[:1], valid[:1])))

    def test_uniform_prediction_is_worse_than_perfect(self) -> None:
        y = np.array([[0, 8, 1, 1], [4, 1, 0, 0]], dtype=float)
        p, _, valid = matrix_to_share(y)
        uniform = np.full_like(p, 1.0 / p.shape[1])
        self.assertLess(row_cpc(p, uniform, valid), row_cpc(p, p, valid))
        self.assertGreater(row_kl(p, uniform, valid), row_kl(p, p, valid))


class RefactorInfrastructureTests(unittest.TestCase):
    def test_dotted_override_deep_merge(self) -> None:
        overrides = parse_cli_overrides(["training.epochs=3", "models=[gravity]", "dataset.name=large_commuting_od"])
        merged = deep_merge({"training": {"epochs": 20, "lr": 0.1}, "models": []}, overrides)
        self.assertEqual(merged["training"]["epochs"], 3)
        self.assertEqual(merged["training"]["lr"], 0.1)
        self.assertEqual(merged["models"], ["gravity"])
        self.assertEqual(merged["dataset"]["name"], "large_commuting_od")

    def test_real_world_dataset_registry(self) -> None:
        spec = get_real_world_dataset({"dataset": {"name": "large_commuting_od", "data_root": "data/opflow_large_commuting_od"}})
        self.assertEqual(spec.name, "large_commuting_od")
        self.assertTrue(str(spec.splits_file).endswith("splits.json"))

    def test_standard_output_columns_available(self) -> None:
        for column in ["ood_rmse", "row_kl", "rank_corr", "topk_overlap", "gate_distance_corr"]:
            self.assertIn(column, STANDARD_METRIC_COLUMNS)


if __name__ == "__main__":
    unittest.main()
