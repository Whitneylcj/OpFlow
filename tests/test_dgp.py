"""Tests for the final four-question DGP."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from opflow_experiments.data.dgp import apply_residual_behavioral_drift, generate_dgp_data, local_competition, rank_prefix_exposure, row_softmax_masked
from opflow_experiments.exposure_models import build_exposure_model
from opflow_experiments.pipelines.run_dgp_main import STRESS_RESIDUAL_DRIFT, _run_one, _run_residual_drift, _write_four_question_outputs, load_exposure_config


def _small_config(n: int = 12) -> dict:
    config = load_exposure_config("configs/synthetic/DGP.yaml")
    config.update({"N": n, "N_o": n, "N_d": n})
    config["dgp"].update(
        {
            "N": n,
            "N_o": n,
            "N_d": n,
            "K_mean": min(6, n - 2),
            "K_std": 1.5,
            "K_min": min(4, n - 2),
            "K_max": min(8, n - 1),
            "num_train_env": 2,
            "num_val_env": 1,
            "num_test_env": 1,
            "poisson": False,
        }
    )
    config["exposure_model"].update({"hidden_dim": 8, "epochs": 2, "patience": 1})
    return config


def _sorted_records(data):
    return data.records.sort_values(["env_index", "origin_id", "candidate_slot", "dest_id"]).reset_index(drop=True)


class DGPTests(unittest.TestCase):
    def test_config_dgp_shape_candidate_mask_and_self_exclusion(self) -> None:
        data = generate_dgp_data(load_exposure_config("configs/synthetic/DGP.yaml"), seed=0)
        self.assertEqual(data.metadata["N_o"], 80)
        self.assertEqual(data.metadata["N_d"], 80)
        self.assertTrue(data.metadata["shared_support"])
        self.assertTrue(np.allclose(data.origins, data.destinations))
        self.assertEqual(data.candidate_indices.shape, data.candidate_mask.shape)
        self.assertEqual(data.candidate_indices.shape[0], 80)
        sizes = np.asarray(data.metadata["candidate_sizes"], dtype=float)
        self.assertAlmostEqual(float(sizes.mean()), 25.0, delta=2.0)
        self.assertAlmostEqual(float(sizes.std(ddof=0)), 5.0, delta=2.5)
        self.assertTrue(np.all(data.candidate_indices[data.candidate_mask] >= 0))
        self.assertTrue(np.all(data.candidate_indices[~data.candidate_mask] == -1))
        for origin_id, row in enumerate(data.candidate_indices):
            self.assertNotIn(origin_id, row[data.candidate_mask[origin_id]].tolist())

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

    def test_q1_changes_train_val_scale_heterogeneity_not_allocation(self) -> None:
        base = _small_config(n=14)
        base["dgp"]["poisson"] = True
        low = {**base, "scenario": {"question": "q1_scale", "scale_severity": 0.0}}
        high = {**base, "scenario": {"question": "q1_scale", "scale_severity": 1.5}}
        low_data = generate_dgp_data(low, seed=4)
        high_data = generate_dgp_data(high, seed=4)
        low_records = _sorted_records(low_data)
        high_records = _sorted_records(high_data)
        identity_cols = [
            "env_id",
            "origin_id",
            "dest_id",
            "candidate_slot",
        ]
        invariant_cols = [
            "cost",
            "opp_proxy",
            "o_true",
            "s_true",
            "ell_true",
            "g_oracle",
            "psi_oracle",
            "pi_oracle",
        ]
        for col in identity_cols:
            self.assertTrue(low_records[col].equals(high_records[col]), col)
        for col in invariant_cols:
            self.assertTrue(np.allclose(low_records[col], high_records[col], atol=1e-10), col)
        train_val = low_records["split"].isin(["train", "val"]).to_numpy()
        test = (low_records["split"] == "test").to_numpy()
        self.assertFalse(np.allclose(low_records.loc[train_val, "R_oracle"], high_records.loc[train_val, "R_oracle"]))
        self.assertFalse(np.allclose(low_records.loc[train_val, "lambda_oracle"], high_records.loc[train_val, "lambda_oracle"]))
        self.assertFalse(np.allclose(low_records.loc[train_val, "y_obs"], high_records.loc[train_val, "y_obs"]))
        self.assertTrue(np.allclose(low_records.loc[test, "R_oracle"], high_records.loc[test, "R_oracle"], atol=1e-10))
        self.assertTrue(np.allclose(low_records.loc[test, "lambda_oracle"], high_records.loc[test, "lambda_oracle"], atol=1e-10))
        self.assertTrue(np.allclose(low_records.loc[test, "y_obs"], high_records.loc[test, "y_obs"], atol=1e-10))
        high_meta = high_data.env_meta
        self.assertTrue(np.allclose(high_meta.query("split in ['train', 'val']")["scale_sigma_applied"], 1.5))
        self.assertTrue(np.allclose(high_meta.query("split == 'test'")["scale_sigma_applied"], 0.0))
        low_std = float(low_data.records.query("split in ['train', 'val']").groupby(["env_id", "origin_id"])["R_oracle"].first().std(ddof=0))
        high_std = float(high_data.records.query("split in ['train', 'val']").groupby(["env_id", "origin_id"])["R_oracle"].first().std(ddof=0))
        self.assertGreater(high_std, low_std)

    def test_q2_opportunity_severity_changes_allocation(self) -> None:
        base = _small_config(n=16)
        low = {**base, "scenario": {"question": "q2_opportunity", "opp_severity": 0.0}}
        high = {**base, "scenario": {"question": "q2_opportunity", "opp_severity": 0.9}}
        low_data = generate_dgp_data(low, seed=5)
        high_data = generate_dgp_data(high, seed=5)
        low_records = _sorted_records(low_data).query("split == 'test'").reset_index(drop=True)
        high_records = _sorted_records(high_data).query("split == 'test'").reset_index(drop=True)
        self.assertFalse(np.allclose(low_records["pi_oracle"], high_records["pi_oracle"]))
        self.assertGreater(float(high_data.env_meta.query("split == 'test'")["opp_severity"].max()), 0.8)

    def test_q3_component_correlation_follows_target_direction(self) -> None:
        config = _small_config(n=36)
        config["scenario"] = {"question": "q3_component_corr", "alpha_train": 0.7, "alpha_val": 0.5, "alpha_test": -0.6}
        data = generate_dgp_data(config, seed=6)
        train_corr = float(data.env_meta.query("split == 'train'")["empirical_corr_log_o_accessibility"].mean())
        test_corr = float(data.env_meta.query("split == 'test'")["empirical_corr_log_o_accessibility"].mean())
        self.assertGreater(train_corr, 0.2)
        self.assertLess(test_corr, -0.1)

    def test_residual_drift_kappa_zero_reproduces_base_test_oracle(self) -> None:
        config = _small_config(n=16)
        config["dgp"]["poisson"] = True
        base_data = generate_dgp_data(config, seed=9)
        drifted = apply_residual_behavioral_drift(base_data, config, seed=9, kappa=0.0)
        base = _sorted_records(base_data)
        records = _sorted_records(drifted)
        for col in ["g_oracle", "psi_oracle", "pi_oracle", "R_oracle", "lambda_oracle", "y_obs"]:
            self.assertTrue(np.allclose(base[col], records[col], atol=1e-10), col)

    def test_residual_drift_changes_only_test_response_law(self) -> None:
        config = _small_config(n=16)
        config["dgp"]["poisson"] = False
        base_data = generate_dgp_data(config, seed=10)
        drifted = apply_residual_behavioral_drift(base_data, config, seed=10, kappa=0.5)
        base = _sorted_records(base_data)
        records = _sorted_records(drifted)
        train_val = base["split"].isin(["train", "val"]).to_numpy()
        test = (base["split"] == "test").to_numpy()
        invariant_cols = [
            "origin_x",
            "origin_y",
            "dest_x",
            "dest_y",
            "opp_proxy",
            "cost",
            "distance",
            "dx",
            "dy",
            "cost_rank_frac",
            "row_standardized_cost",
            "o_true",
            "s_true",
            "ell_true",
            "R_oracle",
        ]
        for col in invariant_cols:
            self.assertTrue(np.allclose(base[col], records[col], atol=1e-10), col)
        for col in ["g_oracle", "psi_oracle", "pi_oracle", "lambda_oracle", "y_obs"]:
            self.assertTrue(np.allclose(base.loc[train_val, col], records.loc[train_val, col], atol=1e-10), col)
            self.assertFalse(np.allclose(base.loc[test, col], records.loc[test, col]), col)
        pi_sum = records.groupby(["env_id", "origin_id"], sort=False)["pi_oracle"].sum()
        lam_err = np.max(np.abs(records["lambda_oracle"] - records["R_oracle"] * records["pi_oracle"]))
        self.assertTrue(np.allclose(pi_sum.to_numpy(dtype=float), 1.0, atol=1e-6))
        self.assertLess(float(lam_err), 1e-8)

    def test_model_aliases_and_no_shortcut_columns(self) -> None:
        data = generate_dgp_data(_small_config(), seed=7)
        self.assertFalse(any(col.startswith("x_res_") for col in data.records.columns))
        self.assertEqual(build_exposure_model("RawCountMLP", _small_config(), seed=0).raw_count_loss, "mse")
        self.assertEqual(build_exposure_model("PairAllocMLP", _small_config(), seed=0).method, "PairMLP")
        self.assertEqual(build_exposure_model("FixedShare", _small_config(), seed=0).method, "ShareStable")
        self.assertEqual(build_exposure_model("NoRankPrefix", _small_config(), seed=0).method, "NoRankExposure")
        self.assertEqual(build_exposure_model("NoLocalField", _small_config(), seed=0).method, "NoCompetition")
        self.assertEqual(build_exposure_model("NoScaleIsolation", _small_config(), seed=0).method, "NoScaleDetach")
        self.assertEqual(build_exposure_model("NoMonotonicity", _small_config(), seed=0).method, "NoMonotonePenalty")
        self.assertEqual(build_exposure_model("NoEnvSelection", _small_config(), seed=0).method, "NoEnvSelection")

    def test_default_opflow_ignores_oracle_latents(self) -> None:
        config = _small_config()
        data = generate_dgp_data(config, seed=2)
        records = _sorted_records(data)
        train = records.loc[records["split"] == "train"].copy()
        val = records.loc[records["split"] == "val"].copy()
        model = build_exposure_model("OpFlow", config, seed=2)
        model.fit(train, val)
        base = model.predict(records)["pi_hat"].numpy()
        mutated = records.copy()
        for col in ["o_true", "s_true", "ell_true", "g_oracle", "psi_oracle", "pi_oracle", "lambda_oracle"]:
            mutated[col] = np.linspace(-1000.0, 1000.0, len(mutated))
        changed = model.predict(mutated)["pi_hat"].numpy()
        self.assertTrue(np.allclose(base, changed, atol=1e-7))

    def test_opflow_uses_leo_selection_and_no_env_selection_only_changes_protocol(self) -> None:
        config = _small_config()
        data = generate_dgp_data(config, seed=8)
        records = _sorted_records(data)
        train = records.loc[records["split"] == "train"].copy()
        val = records.loc[records["split"] == "val"].copy()
        full = build_exposure_model("OpFlow", config, seed=8)
        full.fit(train, val)
        no_env = build_exposure_model("NoEnvSelection", config, seed=8)
        no_env.fit(train, val)
        self.assertEqual(full.selection_summary_["selection_protocol"], "leo")
        self.assertEqual(full.selection_summary_["selection_metric"], "alloc_loss")
        self.assertEqual(full.selection_summary_["num_folds"], train["env_id"].nunique())
        self.assertEqual(len(full.selection_summary_["mean_score_by_epoch"]), config["exposure_model"]["epochs"])
        self.assertTrue(full.selection_summary_["retrained_on_all_train_envs"])
        self.assertIn(full.selected_epoch_, range(1, config["exposure_model"]["epochs"] + 1))
        self.assertEqual(no_env.selection_summary_["selection_protocol"], "train_loss")
        self.assertFalse(no_env.selection_summary_["retrained_on_all_train_envs"])
        for attr in (
            "use_rank_exposure",
            "use_competition",
            "use_opportunity_field",
            "use_row_center",
            "detach_count_allocation",
            "use_groupdro",
            "use_vrex",
            "use_monotone_penalty",
            "use_sit_operator_potential",
        ):
            self.assertEqual(getattr(full, attr), getattr(no_env, attr), attr)

    def test_pipeline_smoke_writes_question_artifacts(self) -> None:
        config = _small_config()
        config["scenario"] = {"question": "q2_opportunity", "opp_severity": 0.9}
        with tempfile.TemporaryDirectory() as tmp:
            metrics, diagnostics = _run_one(
                config,
                regime="DGP",
                seed=3,
                methods=["OpFlow", "PairAllocMLP", "FixedShare"],
                run_dir=Path(tmp),
                question="q2_opportunity",
                scenario_name="opp_0.9",
                scenario_value_name="severity",
                scenario_value=0.9,
                show_progress=False,
            )
            self.assertFalse(metrics.empty)
            self.assertFalse(diagnostics.empty)
            self.assertEqual(set(metrics["regime"]), {"DGP"})
            self.assertEqual(set(metrics["question"]), {"q2_opportunity"})
            artifact = Path(tmp) / "artifacts" / "DGP" / "q2_opportunity" / "opp_0.9" / "seed_3"
            self.assertTrue((artifact / "env_meta.csv").exists())
            self.assertTrue((artifact / "gradient_audit.csv").exists())
            self.assertTrue((artifact / "selection_diagnostics.csv").exists())

    def test_residual_drift_runner_trains_once_and_writes_outputs(self) -> None:
        config = _small_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            metrics, diagnostics = _run_residual_drift(
                config,
                regime="DGP",
                seed=11,
                methods=["OpFlow"],
                kappa_values=[0.0, 0.5],
                run_dir=run_dir,
                show_progress=False,
            )
            self.assertEqual(set(metrics["question"]), {STRESS_RESIDUAL_DRIFT})
            self.assertEqual(set(metrics["split"]), {"test"})
            self.assertEqual(sorted(metrics["scenario_value"].unique().tolist()), [0.0, 0.5])
            self.assertFalse(diagnostics.empty)
            artifact = run_dir / "artifacts" / "DGP" / STRESS_RESIDUAL_DRIFT / "seed_11"
            self.assertEqual(len((artifact / "selection_diagnostics.csv").read_text(encoding="utf-8").strip().splitlines()) - 1, 1)
            self.assertEqual(len((artifact / "gradient_audit.csv").read_text(encoding="utf-8").strip().splitlines()) - 1, 1)
            written = _write_four_question_outputs(metrics, diagnostics, config, run_dir, Path(tmp) / "results")
            self.assertIn(STRESS_RESIDUAL_DRIFT, written)
            stress_metrics = Path(tmp) / "results" / STRESS_RESIDUAL_DRIFT / "metrics.csv"
            self.assertTrue(stress_metrics.exists())
            frame = pd.read_csv(stress_metrics)
            self.assertEqual(frame.columns.tolist(), ["seed", "kappa", "model", "row_kl", "row_cpc", "pmse", "log_rmse", "global_cpc"])
            self.assertTrue(Path("figures/stress_residual_drift.png").exists())
            self.assertTrue(Path("figures/stress_residual_drift.pdf").exists())


if __name__ == "__main__":
    unittest.main()
