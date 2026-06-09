"""Tests for the final single-regime DGP."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from opflow_experiments.data.dgp import generate_dgp_data, local_competition, rank_prefix_exposure, row_softmax_masked
from opflow_experiments.exposure_models import build_exposure_model
from opflow_experiments.pipelines.run_dgp_main import _run_one, load_exposure_config


def _small_config() -> dict:
    config = load_exposure_config("configs/synthetic/DGP.yaml")
    config["dgp"].update(
        {
            "N_o": 8,
            "N_d": 12,
            "K_mean": 6,
            "K_std": 1.5,
            "K_min": 4,
            "K_max": 8,
            "num_train_env": 2,
            "num_val_env": 1,
            "num_test_env": 1,
            "poisson": False,
        }
    )
    config["exposure_model"].update({"hidden_dim": 8, "epochs": 2, "patience": 2})
    return config


def _sorted_records(data):
    return data.records.sort_values(["env_index", "origin_id", "candidate_slot", "dest_id"]).reset_index(drop=True)


class DGPTests(unittest.TestCase):
    def test_config_dgp_shape_and_candidate_mask(self) -> None:
        data = generate_dgp_data(load_exposure_config("configs/synthetic/DGP.yaml"), seed=0)
        self.assertEqual(data.metadata["N_o"], 80)
        self.assertEqual(data.metadata["N_d"], 80)
        self.assertEqual(data.candidate_indices.shape, data.candidate_mask.shape)
        self.assertEqual(data.candidate_indices.shape[0], 80)
        sizes = np.asarray(data.metadata["candidate_sizes"], dtype=float)
        self.assertAlmostEqual(float(sizes.mean()), 25.0, delta=2.0)
        self.assertAlmostEqual(float(sizes.std(ddof=0)), 5.0, delta=2.5)
        self.assertTrue(np.all(data.candidate_indices[data.candidate_mask] >= 0))
        self.assertTrue(np.all(data.candidate_indices[~data.candidate_mask] == -1))

    def test_masked_softmax_and_rank_prefix(self) -> None:
        scores = np.array([[1.0, 2.0, -100.0], [3.0, 0.0, 1.0]])
        mask = np.array([[True, True, False], [True, False, True]])
        pi = row_softmax_masked(scores, mask, tau=0.7)
        self.assertTrue(np.allclose(pi.sum(axis=1), 1.0))
        self.assertEqual(float(pi[0, 2]), 0.0)
        opp = np.array([[2.0, 5.0, 3.0]])
        cost = np.array([[0.3, 0.1, 0.2]])
        exposure, ranks = rank_prefix_exposure(opp, cost)
        self.assertEqual(ranks.tolist(), [[2, 0, 1]])
        self.assertTrue(np.allclose(exposure, [[8.0, 0.0, 5.0]]))

    def test_local_competition_excludes_self(self) -> None:
        destinations = np.array([[0.0, 0.0], [0.0, 1.0]])
        opp = np.array([10.0, 1.0])
        comp = local_competition(destinations, opp, omega=0.5)
        self.assertLess(comp[0], 1.0)
        self.assertLess(comp[1], 10.0)

    def test_environment_opportunity_and_oracle_checks(self) -> None:
        data = generate_dgp_data(_small_config(), seed=1)
        meta = data.env_meta
        self.assertTrue(np.allclose(meta["opportunity_mean"].to_numpy(dtype=float), 1.0, atol=1e-6))
        self.assertLess(float(data.metadata["opportunity_env_corr_max"]), 0.995)
        self.assertEqual(data.metadata["standardizer_fit_split"], "train")
        records = data.records
        psi_mean = records.groupby(["env_id", "origin_id"], sort=False)["psi_oracle"].mean().abs().max()
        pi_sum = records.groupby(["env_id", "origin_id"], sort=False)["pi_oracle"].sum()
        lam_err = np.max(np.abs(records["lambda_oracle"] - records["R_oracle"] * records["pi_oracle"]))
        self.assertLess(float(psi_mean), 1e-8)
        self.assertTrue(np.allclose(pi_sum.to_numpy(dtype=float), 1.0, atol=1e-6))
        self.assertLess(float(lam_err), 1e-8)
        self.assertTrue(np.all(records["y_obs"].to_numpy(dtype=float) >= 0.0))

    def test_default_opflow_ignores_shortcuts_and_oracle_latents(self) -> None:
        config = _small_config()
        data = generate_dgp_data(config, seed=2)
        records = _sorted_records(data)
        train = records.loc[records["split"] == "train"].copy()
        val = records.loc[records["split"] == "val"].copy()
        model = build_exposure_model("OpFlow", config, seed=2)
        model.fit(train, val)
        base = model.predict(records)["pi_hat"].numpy()
        mutated = records.copy()
        for col in [
            "x_res_0",
            "x_res_1",
            "o_true",
            "s_true",
            "ell_true",
            "g_oracle",
            "psi_oracle",
            "pi_oracle",
            "lambda_oracle",
        ]:
            mutated[col] = np.linspace(-1000.0, 1000.0, len(mutated))
        changed = model.predict(mutated)["pi_hat"].numpy()
        self.assertTrue(np.allclose(base, changed, atol=1e-7))

    def test_pipeline_smoke_writes_dgp_artifacts(self) -> None:
        config = _small_config()
        with tempfile.TemporaryDirectory() as tmp:
            metrics, diagnostics = _run_one(
                config,
                regime="DGP",
                seed=3,
                methods=["OpFlow", "ShortcutOnly", "OraclePotential"],
                run_dir=Path(tmp),
                show_progress=False,
            )
            self.assertFalse(metrics.empty)
            self.assertFalse(diagnostics.empty)
            self.assertEqual(set(metrics["regime"]), {"DGP"})
            self.assertTrue((Path(tmp) / "artifacts" / "DGP" / "seed_3" / "env_meta.csv").exists())
            self.assertTrue((Path(tmp) / "artifacts" / "DGP" / "seed_3" / "gradient_audit.csv").exists())


if __name__ == "__main__":
    unittest.main()
