"""Focused tests for the constrained-SI real-world V2 surface."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from opflow_benchmark.baselines.common import mean_scale_prediction, set_seed
from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline
from opflow_benchmark.baselines.radiation import RadiationBaseline
from opflow_benchmark.baselines.raw_count_mlp import RawCountMLPBaseline
from opflow_benchmark.exposure import local_destination_exposure, rank_prefix_exposure
from opflow_experiments.pipelines.realworld_constrained_si import (
    _scale_diagnostics_v2,
    _temporal_infeasible,
    _write_exposure_signatures,
    apply_constrained_si_debug,
)


def _toy_area(n: int = 5) -> dict:
    rng = np.random.default_rng(2027)
    node_feat = rng.uniform(0.2, 3.0, size=(n, 4)).astype("float32")
    coords = rng.normal(size=(n, 2)).astype("float32")
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)).astype("float32")
    adjacency = (distance <= np.quantile(distance, 0.5)).astype("float32")
    y = rng.poisson(lam=np.maximum(1.0, 5.0 - distance)).astype("float32")
    np.fill_diagonal(y, 0.0)
    outflow = y.sum(axis=1).astype("float32")
    valid = outflow > 0
    p = np.zeros_like(y, dtype="float32")
    p[valid] = y[valid] / outflow[valid, None]
    return {
        "area_id": "toy",
        "node_feat": torch.as_tensor(node_feat),
        "distance": torch.as_tensor(distance),
        "adjacency": torch.as_tensor(adjacency),
        "y_raw": torch.as_tensor(y),
        "outflow": torch.as_tensor(outflow),
        "valid_origin": torch.as_tensor(valid),
        "p_share": torch.as_tensor(p),
    }


def _write_area(root: Path, area_id: str, area: dict) -> None:
    area_dir = root / area_id
    area_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("y_raw", "y_raw.npy"),
        ("p_share", "p_share.npy"),
        ("outflow", "outflow.npy"),
        ("valid_origin", "valid_origin.npy"),
        ("distance", "distance.npy"),
        ("adjacency", "adjacency.npy"),
        ("node_feat", "node_feat.npy"),
    ):
        np.save(area_dir / filename, area[key].detach().cpu().numpy())


class RealWorldConstrainedSITests(unittest.TestCase):
    def test_rank_prefix_excludes_origin_self_and_handles_ties(self) -> None:
        opportunity = torch.tensor([1.0, 2.0, 3.0])
        distance = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ]
        )
        prefix = rank_prefix_exposure(opportunity, distance, row_chunk_size=2)
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(prefix, expected))

    def test_local_destination_exposure_excludes_self_for_both_modes(self) -> None:
        opportunity = torch.tensor([1.0, 2.0, 4.0])
        distance = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        kernel = local_destination_exposure(opportunity, distance, omega=1.0, mode="kernel_sum")
        softmax = local_destination_exposure(opportunity, distance, omega=1.0, mode="softmax")
        kernel_manual = torch.tensor(
            [
                torch.exp(torch.tensor(-1.0)) * 2.0 + torch.exp(torch.tensor(-2.0)) * 4.0,
                torch.exp(torch.tensor(-1.0)) * 1.0 + torch.exp(torch.tensor(-3.0)) * 4.0,
                torch.exp(torch.tensor(-2.0)) * 1.0 + torch.exp(torch.tensor(-3.0)) * 2.0,
            ]
        )
        self.assertTrue(torch.allclose(kernel, kernel_manual, atol=1e-6))
        self.assertTrue(torch.all(softmax > 0))
        self.assertLess(float(softmax[0]), 4.0)

    def test_radiation_allocation_is_row_normalized(self) -> None:
        area = _toy_area()
        model = RadiationBaseline(device="cpu", rank_prefix_row_chunk_size=3)
        model.fit([area], [])
        _, p_hat, _ = model.predict_area(area, "radiation_learned_scale")
        self.assertTrue(torch.all(torch.isfinite(p_hat)))
        self.assertTrue(torch.allclose(p_hat.sum(dim=1), torch.ones(p_hat.shape[0]), atol=1e-5))

    def test_raw_count_mlp_derives_normalized_allocation(self) -> None:
        area = _toy_area()
        model = RawCountMLPBaseline(epochs=0, hidden_dim=6, device="cpu")
        model.fit([area], [])
        _, p_hat, _ = model.predict_area(area)
        self.assertTrue(torch.all(torch.isfinite(p_hat)))
        self.assertTrue(torch.allclose(p_hat.sum(dim=1), torch.ones(p_hat.shape[0]), atol=1e-5))

    def test_opflow_no_scale_and_no_row_centering_flags(self) -> None:
        no_scale = OpFlowRealBaseline(model_variant="opflow_no_scale_branch", device="cpu")
        no_center = OpFlowRealBaseline(model_variant="opflow_no_row_centering", device="cpu")
        self.assertFalse(no_scale.use_scale_branch)
        self.assertTrue(no_scale.use_row_centering)
        self.assertTrue(no_center.use_scale_branch)
        self.assertFalse(no_center.use_row_centering)

    def test_removed_realworld_variants_raise(self) -> None:
        for name in ("leaky_opflow", "opflow_no_routing", "opflow_no_diagnostic"):
            with self.assertRaises(ValueError):
                OpFlowRealBaseline(model_variant=name, epochs=0, hidden_dim=6, device="cpu")

    def test_validation_selection_uses_allocation_objective(self) -> None:
        area = _toy_area()
        model = OpFlowRealBaseline(model_variant="opflow_full", epochs=1, hidden_dim=6, device="cpu")
        calls: list[str] = []

        def fake_parts(_area) -> dict:
            calls.append("parts")
            assert model.model is not None
            scalar = next(model.model.parameters()).sum() * 0.0
            return {
                "alloc": scalar + torch.tensor(1.0),
                "scale": scalar + torch.tensor(10.0),
                "count": scalar + torch.tensor(10.0),
                "op_penalty": scalar,
                "op_monotone_violation_D": scalar.detach(),
                "op_monotone_violation_S": scalar.detach(),
                "op_monotone_violation_B": scalar.detach(),
            }

        with patch.object(model, "_loss_area_parts", side_effect=fake_parts):
            model.fit([area], [area])
        self.assertGreater(len(calls), 0)

    def test_no_scale_branch_keeps_allocation_identical_with_same_weights(self) -> None:
        area = _toy_area()
        kwargs = dict(epochs=0, hidden_dim=6, row_chunk_size=2, device="cpu")
        set_seed(19)
        full = OpFlowRealBaseline(model_variant="opflow_full", **kwargs)
        full.fit([area], [area])
        no_scale = OpFlowRealBaseline(model_variant="opflow_no_scale_branch", **kwargs)
        no_scale.fit([area], [area])
        assert full.model is not None
        assert no_scale.model is not None
        no_scale.model.load_state_dict(full.model.state_dict())
        _, p_full, _ = full.predict_area(area)
        _, p_no_scale, _ = no_scale.predict_area(area)
        self.assertTrue(torch.allclose(p_full, p_no_scale, atol=1e-6))

    def test_exposure_artifact_generation_writes_summary_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            _write_area(root, "a0", _toy_area(4))
            _write_area(root, "a1", _toy_area(5))
            out_dir = Path(tmp) / "out"
            _write_exposure_signatures(
                out_dir,
                {
                    "dataset": {"data_root": str(root)},
                    "training": {"omega_grid": [10.0], "rank_prefix_row_chunk_size": 2},
                    "reporting": {"exposure_signature_max_areas": 1, "exposure_full_arrays_max_areas": 1},
                },
            )
            summary = pd.read_csv(out_dir / "exposure_signature_summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertTrue((out_dir / "exposure_arrays" / "a0_exposure_signature.npz").exists())

    def test_temporal_infeasible_report_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            _temporal_infeasible(out_dir, {"dataset": {"data_root": "data/static"}})
            text = (out_dir / "temporal_ood_infeasible.md").read_text(encoding="utf-8")
            self.assertIn("no date, period, environment, or chronological identifier", text)
            self.assertIn("does not claim temporal OOD evidence", text)
            self.assertIn("diagnostic cross-area stress splits", text)

    def test_debug_overrides_cap_expensive_artifacts(self) -> None:
        cfg = apply_constrained_si_debug({"suite": "constrained_si", "reporting": {"exposure_signature_max_areas": 100}})
        self.assertLessEqual(cfg["reporting"]["exposure_signature_max_areas"], 8)
        self.assertLessEqual(cfg["training"]["max_pair_mlp_train_areas"], 3)

    def test_mean_scale_reconstruction_and_scale_diagnostic_columns(self) -> None:
        area = _toy_area()
        p_hat = area["p_share"]
        mean_y = mean_scale_prediction(area, p_hat, mean_outflow=3.0)
        self.assertTrue(torch.allclose(mean_y.sum(dim=1), torch.full((p_hat.shape[0],), 3.0), atol=1e-6))
        rows = pd.DataFrame(
            [
                {
                    "split_name": "random",
                    "model": "opflow_full",
                    "seed": 2026,
                    "part": "test",
                    "variant": "opflow_full_learned_scale",
                    "true_outflow_mean": 4.0,
                    "true_outflow_std": 1.0,
                    "pred_outflow_mean": 3.0,
                    "pred_outflow_std": 0.5,
                    "total_flow_ratio": 0.75,
                    "outflow_log_rmse": 0.2,
                    "outflow_corr": 0.9,
                    "global_cpc": 0.4,
                    "log_rmse": 1.0,
                },
                {
                    "split_name": "random",
                    "model": "opflow_full",
                    "seed": 2026,
                    "part": "test",
                    "variant": "opflow_full_oracle_scale",
                    "global_cpc": 0.6,
                    "log_rmse": 0.8,
                    "row_cpc": 0.7,
                    "row_kl": 0.3,
                },
                {
                    "split_name": "random",
                    "model": "opflow_full",
                    "seed": 2026,
                    "part": "test",
                    "variant": "opflow_full_mean_scale",
                    "global_cpc": 0.5,
                    "log_rmse": 0.9,
                    "outflow_log_rmse": 0.25,
                },
            ]
        )
        diag = _scale_diagnostics_v2(rows)
        for col in ("learned_scale_global_cpc", "oracle_global_cpc", "mean_scale_global_cpc", "true_outflow_mean", "pred_outflow_mean"):
            self.assertIn(col, diag.columns)


if __name__ == "__main__":
    unittest.main()
