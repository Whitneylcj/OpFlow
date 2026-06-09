"""Focused tests for the real-world P-only OpFlow surface."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from opflow_benchmark.baselines import build_model
from opflow_benchmark.baselines.common import oracle_scale_prediction, set_seed
from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline
from opflow_benchmark.metrics import evaluate_area
from opflow_benchmark.splits import matched_extreme_split
from opflow_experiments.pipelines.realworld_upgrade import FEATURE_PARTITION


def _toy_area(n: int = 6, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    node_feat = rng.uniform(0.1, 2.0, size=(n, 3)).astype("float32")
    coords = rng.normal(size=(n, 2))
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)).astype("float32")
    adjacency = (distance < np.quantile(distance, 0.45)).astype("float32")
    y = rng.poisson(lam=np.maximum(1.0, 4.0 - distance)).astype("float32")
    np.fill_diagonal(y, 0.0)
    outflow = y.sum(axis=1).astype("float32")
    valid = outflow > 0
    p = np.zeros_like(y, dtype="float32")
    p[valid] = y[valid] / outflow[valid, None]
    return {
        "area_id": f"toy_{seed}",
        "node_feat": torch.as_tensor(node_feat),
        "distance": torch.as_tensor(distance),
        "adjacency": torch.as_tensor(adjacency),
        "y_raw": torch.as_tensor(y),
        "outflow": torch.as_tensor(outflow),
        "valid_origin": torch.as_tensor(valid),
        "p_share": torch.as_tensor(p),
    }


def _fit_opflow(area: dict | None = None, **kwargs) -> OpFlowRealBaseline:
    area = area or _toy_area()
    set_seed(7)
    model = OpFlowRealBaseline(epochs=1, hidden_dim=8, row_chunk_size=3, area_batch_size=1, device="cpu", **kwargs)
    model.fit([area], [area])
    return model


class RealWorldUpgradeTests(unittest.TestCase):
    def test_feature_partition_keeps_labels_out_of_deployed_inputs(self) -> None:
        deployed = " ".join(FEATURE_PARTITION["deployed_geographic_opportunity_variables"]).lower()
        forbidden = ("y_raw", "p_share", "split label", "environment id", "od labels")
        for token in forbidden:
            self.assertNotIn(token, deployed)

    def test_scale_shift_split_constructs_matched_tails(self) -> None:
        records = pd.DataFrame(
            {
                "area_id": [f"a{i}" for i in range(80)],
                "log_total_flow": np.linspace(1.0, 6.0, 80),
                "log_n_regions": np.tile(np.linspace(1.0, 2.0, 8), 10),
                "flow_density": np.repeat(np.linspace(0.05, 0.5, 10), 8),
            }
        )
        split, bins = matched_extreme_split(
            records,
            target_col="log_total_flow",
            control_cols=("log_n_regions", "flow_density"),
            n_bins=2,
            high_to_low=False,
        )
        self.assertGreater(len(split["train"]), 0)
        self.assertGreater(len(split["test"]), 0)
        self.assertFalse(bins.empty)

    def test_oracle_scale_preserves_allocation_metrics(self) -> None:
        area = _toy_area()
        p_hat = torch.flip(area["p_share"], dims=[1]) * 0.8 + area["p_share"] * 0.2
        y_hat = area["outflow"][:, None] * p_hat
        oracle = oracle_scale_prediction(area, p_hat)
        learned_metrics = evaluate_area(area["y_raw"].numpy(), y_hat.numpy(), area["p_share"].numpy(), p_hat.numpy(), area["valid_origin"].numpy(), topk=3)
        oracle_metrics = evaluate_area(area["y_raw"].numpy(), oracle.numpy(), area["p_share"].numpy(), p_hat.numpy(), area["valid_origin"].numpy(), topk=3)
        self.assertAlmostEqual(learned_metrics["row_kl"], oracle_metrics["row_kl"])
        self.assertAlmostEqual(learned_metrics["row_cpc"], oracle_metrics["row_cpc"])

    def test_removed_realworld_variants_raise(self) -> None:
        for name in ("leaky_opflow", "opflow_no_routing", "opflow_no_diagnostic"):
            with self.assertRaises(ValueError):
                build_model(name, device="cpu")
            with self.assertRaises(ValueError):
                OpFlowRealBaseline(model_variant=name, device="cpu")

    def test_operator_outputs_are_row_normalized_and_centered(self) -> None:
        area = _toy_area()
        model = _fit_opflow(area)
        y_hat, p_hat, extra = model.predict_area(area)
        self.assertTrue(torch.all(torch.isfinite(y_hat)))
        self.assertTrue(torch.allclose(p_hat.sum(dim=1), torch.ones(p_hat.shape[0]), atol=1e-6))
        self.assertLess(extra["psi_bar_row_mean_abs"], 1e-6)
        self.assertIn("operator_mean_o_tilde", extra)
        self.assertIn("operator_mean_c_tilde", extra)

    def test_rank_prefix_uses_learned_impedance_and_excludes_self(self) -> None:
        area = _toy_area()
        model = _fit_opflow(area)
        assert model.model is not None
        opportunity = torch.tensor([1.0, 2.0, 3.0])
        impedance = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ]
        )
        prefix = model.model.rank_prefix_from_impedance(opportunity, impedance, row_chunk_size=2)
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(prefix, expected))

    def test_count_loss_does_not_update_allocation_modules(self) -> None:
        area = _toy_area()
        model = _fit_opflow(area)
        assert model.model is not None
        parts = model._loss_area_parts(area)
        allocation_modules = [
            model.model.opportunity,
            model.model.impedance,
            model.model.deterrence,
            model.model.benefit,
            model.model.exposure,
            model.model.competition,
            model.model.modifier,
            model.model.generic_potential,
        ]
        model.model.zero_grad(set_to_none=True)
        parts["count"].backward()
        grads = [
            float(param.grad.detach().abs().max().item())
            for module in allocation_modules
            for param in module.parameters()
            if param.grad is not None
        ]
        self.assertLessEqual(max(grads) if grads else 0.0, 1e-7)

    def test_groupdro_and_vrex_area_risks_are_logged(self) -> None:
        area_a = _toy_area(seed=11)
        area_b = _toy_area(seed=12)
        model = _fit_opflow(area_a)
        assert model.model is not None
        loss, info = model._loss_batch_parts([area_a, area_b])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(info["alloc_risk_max"], info["alloc_risk_mean"] - 1e-8)
        self.assertGreaterEqual(info["vrex_alloc_var"], 0.0)
        single_loss, single_info = model._loss_batch_parts([area_a])
        self.assertTrue(torch.isfinite(single_loss))
        self.assertEqual(single_info["vrex_alloc_var"], 0.0)

    def test_variant_flags(self) -> None:
        no_scale = OpFlowRealBaseline(model_variant="opflow_no_scale_branch", device="cpu")
        no_center = OpFlowRealBaseline(model_variant="opflow_no_row_centering", device="cpu")
        generic = OpFlowRealBaseline(model_variant="opflow_generic_potential_mlp", device="cpu")
        no_groupdro = OpFlowRealBaseline(model_variant="opflow_no_groupdro", device="cpu")
        no_vrex = OpFlowRealBaseline(model_variant="opflow_no_vrex", device="cpu")
        no_monotone = OpFlowRealBaseline(model_variant="opflow_no_monotone_penalty", device="cpu")
        self.assertFalse(no_scale.use_scale_branch)
        self.assertFalse(no_center.use_row_centering)
        self.assertFalse(generic.use_sit_operator_potential)
        self.assertFalse(no_groupdro.use_groupdro)
        self.assertFalse(no_vrex.use_vrex)
        self.assertFalse(no_monotone.use_monotone_penalty)


if __name__ == "__main__":
    unittest.main()
