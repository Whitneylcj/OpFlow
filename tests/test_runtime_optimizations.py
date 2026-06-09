import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from opflow_benchmark.baselines.common import evaluate_area_allocation, evaluate_area_predictions, tensor_area
from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline, OpFlowRealNet
from opflow_benchmark.metrics import evaluate_area
from opflow_experiments.pipelines import real_world_large_commuting_od as real_pipeline


def _toy_area(n: int = 5, node_dim: int = 4) -> dict:
    rng = np.random.default_rng(123)
    y = torch.as_tensor(rng.poisson(2.0, size=(n, n)).astype(np.float32))
    y.fill_diagonal_(0.0)
    outflow = y.sum(dim=1)
    valid = outflow > 1e-8
    p = torch.zeros_like(y)
    p[valid] = y[valid] / outflow[valid, None]
    coords = torch.linspace(0.0, 1.0, n)
    dist = torch.abs(coords[:, None] - coords[None, :]) + torch.eye(n)
    return {
        "area_id": "toy",
        "y_raw": y,
        "p_share": p,
        "outflow": outflow,
        "valid_origin": valid,
        "node_feat": torch.as_tensor(rng.uniform(0.0, 3.0, size=(n, node_dim)).astype(np.float32)),
        "distance": dist.float(),
        "adjacency": (dist < 0.6).float(),
    }


class RuntimeOptimizationTests(unittest.TestCase):
    def test_shared_allocation_metrics_match_full_evaluate_area(self) -> None:
        area = _toy_area()
        logits = torch.randn(area["y_raw"].shape, generator=torch.Generator().manual_seed(7))
        p_hat = torch.softmax(logits, dim=1)
        y_hat = (area["outflow"][:, None] + 1.5) * p_hat

        full = evaluate_area(
            area["y_raw"].numpy(),
            y_hat.numpy(),
            p_true=area["p_share"].numpy(),
            p_pred=p_hat.numpy(),
            valid_origin=area["valid_origin"].numpy(),
            topk=5,
        )
        shared = evaluate_area_predictions(
            area,
            y_hat,
            p_hat,
            topk=5,
            allocation=evaluate_area_allocation(area, p_hat, topk=5),
        )
        for key, expected in full.items():
            actual = shared[key]
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual), key)
            else:
                self.assertAlmostEqual(float(actual), float(expected), places=9, msg=key)

    def test_memory_aware_workers_cap_requested_parallelism(self) -> None:
        config = {
            "splits": ["random"],
            "models": ["opflow_real", "pair_mlp"],
            "training": {
                "process_workers": 8,
                "process_workers_memory_fraction": 0.65,
                "process_workers_reserved_memory_gb": 2.0,
            },
        }
        tasks = [{"idx": idx} for idx in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.log"
            with mock.patch.object(real_pipeline, "_available_memory_bytes", return_value=8 * 1024**3), mock.patch.object(
                real_pipeline,
                "_estimate_worker_memory_bytes",
                return_value=(1 * 1024**3, {"max_n": 1000, "max_dense_matrix_mb": 4.0}),
            ):
                self.assertEqual(real_pipeline._resolve_process_workers(config, tasks, progress), 3)

    def test_disabled_real_opflow_exposure_ops_skip_rank_and_local(self) -> None:
        area = _toy_area(n=4, node_dim=3)
        model = OpFlowRealBaseline(model_variant="opflow_no_exposure_ops", hidden_dim=4, row_chunk_size=2, device="cpu")
        model.stats.fit([area])
        batch = tensor_area(area, model.device, model.stats)
        model._ensure_model(batch["x"].shape[1])
        with mock.patch.object(OpFlowRealNet, "rank_prefix_from_impedance", side_effect=AssertionError("rank prefix called")), mock.patch(
            "opflow_benchmark.baselines.opflow_real.local_destination_exposure",
            side_effect=AssertionError("local exposure called"),
        ):
            parts = model._operator_parts(batch)
        self.assertTrue(torch.allclose(parts["prefix"], torch.zeros_like(parts["prefix"])))
        self.assertTrue(torch.allclose(parts["local"], torch.zeros_like(parts["local"])))


if __name__ == "__main__":
    unittest.main()
