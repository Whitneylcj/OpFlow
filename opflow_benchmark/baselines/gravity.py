"""Classical gravity-style benchmark baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from opflow_benchmark.baselines.common import (
    timed_evaluate_area_allocation,
    mean_scale_prediction,
    oracle_scale_prediction,
    profile_record,
    timed_evaluate_area_predictions,
)


EPS = 1e-8


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    xtx = x.T @ x
    reg = alpha * np.eye(xtx.shape[0], dtype=float)
    reg[0, 0] = 0.0
    try:
        coef = np.linalg.solve(xtx + reg, x.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx + reg, x.T @ y, rcond=None)[0]
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(coef, -20.0, 20.0)


class GravityBaseline:
    """Log-linear gravity-style model with learned origin scale."""

    def __init__(self, **_: Any) -> None:
        self.scale_coef: np.ndarray | None = None
        self.count_coef: np.ndarray | None = None
        self.scale_mean: np.ndarray | None = None
        self.scale_std: np.ndarray | None = None
        self.count_mean: np.ndarray | None = None
        self.count_std: np.ndarray | None = None
        self.mean_outflow = 1.0
        self.profile: dict[str, float] = {}

    def _node_proxy(self, area: dict[str, Any]) -> np.ndarray:
        x = area["node_feat"].detach().cpu().numpy()
        return np.log1p(np.maximum(x, 0.0).sum(axis=1))

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        scale_x = []
        scale_y = []
        count_x = []
        count_y = []
        for area in train_areas:
            y = area["y_raw"].detach().cpu().numpy()
            out = y.sum(axis=1)
            proxy = self._node_proxy(area)
            distance = area["distance"].detach().cpu().numpy()
            adj = area["adjacency"].detach().cpu().numpy()
            n = y.shape[0]
            scale_x.append(np.column_stack([np.ones(n), proxy]))
            scale_y.append(np.log1p(out))
            origin_proxy = np.broadcast_to(proxy[:, None], (n, n))
            dest_proxy = np.broadcast_to(proxy[None, :], (n, n))
            design = np.stack(
                [
                    np.ones(n * n),
                    origin_proxy.reshape(-1),
                    dest_proxy.reshape(-1),
                    -np.log1p(np.maximum(distance, 0.0)).reshape(-1),
                    adj.reshape(-1),
                ],
                axis=1,
            )
            count_x.append(design)
            count_y.append(np.log1p(y.reshape(-1)))
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            self.mean_outflow = float(torch.cat(valid_outflows).mean().item())
        sx = np.concatenate(scale_x)
        cx = np.concatenate(count_x)
        self.scale_mean = sx[:, 1:].mean(axis=0)
        self.scale_std = sx[:, 1:].std(axis=0)
        self.scale_std[self.scale_std <= EPS] = 1.0
        self.count_mean = cx[:, 1:].mean(axis=0)
        self.count_std = cx[:, 1:].std(axis=0)
        self.count_std[self.count_std <= EPS] = 1.0
        sx_scaled = sx.copy()
        cx_scaled = cx.copy()
        sx_scaled[:, 1:] = (sx[:, 1:] - self.scale_mean) / self.scale_std
        cx_scaled[:, 1:] = (cx[:, 1:] - self.count_mean) / self.count_std
        sx_scaled = np.nan_to_num(sx_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        cx_scaled = np.nan_to_num(cx_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        sx_scaled[:, 1:] = np.clip(sx_scaled[:, 1:], -20.0, 20.0)
        cx_scaled[:, 1:] = np.clip(cx_scaled[:, 1:], -20.0, 20.0)
        self.scale_coef = _ridge_fit(sx_scaled, np.concatenate(scale_y))
        self.count_coef = _ridge_fit(cx_scaled, np.concatenate(count_y))
        return {"train_loss": [], "val_loss": []}

    def _predict_logits_and_scale(self, area: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if self.scale_coef is None or self.count_coef is None:
            raise RuntimeError("GravityBaseline is not fitted")
        y = area["y_raw"].detach().cpu().numpy()
        n = y.shape[0]
        proxy = self._node_proxy(area)
        distance = area["distance"].detach().cpu().numpy()
        adj = area["adjacency"].detach().cpu().numpy()
        scale_x = np.column_stack([np.ones(n), proxy])
        scale_x[:, 1:] = (scale_x[:, 1:] - self.scale_mean) / self.scale_std
        scale_x = np.nan_to_num(scale_x, nan=0.0, posinf=0.0, neginf=0.0)
        scale_x[:, 1:] = np.clip(scale_x[:, 1:], -20.0, 20.0)
        outflow_hat = np.maximum(np.expm1(np.clip(scale_x @ self.scale_coef, -5.0, 16.0)), EPS)
        origin_proxy = np.broadcast_to(proxy[:, None], (n, n))
        dest_proxy = np.broadcast_to(proxy[None, :], (n, n))
        design = np.stack(
            [
                np.ones(n * n),
                origin_proxy.reshape(-1),
                dest_proxy.reshape(-1),
                -np.log1p(np.maximum(distance, 0.0)).reshape(-1),
                adj.reshape(-1),
            ],
            axis=1,
        )
        design[:, 1:] = (design[:, 1:] - self.count_mean) / self.count_std
        design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
        design[:, 1:] = np.clip(design[:, 1:], -20.0, 20.0)
        linear = design[:, 0] * self.count_coef[0] + np.sum(design[:, 1:] * self.count_coef[1:], axis=1)
        linear = np.nan_to_num(linear, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = np.clip(linear.reshape(n, n), -30.0, 30.0)
        return logits, outflow_hat

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logits_np, outflow_np = self._predict_logits_and_scale(area)
        logits = torch.as_tensor(logits_np, dtype=torch.float32)
        p_hat = torch.softmax(logits, dim=1)
        outflow = torch.as_tensor(outflow_np, dtype=torch.float32)
        y_hat = outflow[:, None] * p_hat
        return y_hat, p_hat, {"mean_pred_outflow": float(outflow.mean().item())}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload: dict[str, Any] = {"gravity_learned_scale": {}, "gravity_oracle_scale": {}, "gravity_mean_scale": {}}
        for split, areas in split_areas.items():
            learned_rows = []
            oracle_rows = []
            mean_rows = []
            for area in areas:
                start = time.perf_counter()
                y_hat, p_hat, extra = self.predict_area(area)
                profile_record(self.profile, "predict_seconds", time.perf_counter() - start)
                self.profile["areas"] += 1.0
                allocation = timed_evaluate_area_allocation(self.profile, area, p_hat, topk=topk)
                learned_rows.append(timed_evaluate_area_predictions(self.profile, area, y_hat, p_hat, topk=topk, extra=extra, allocation=allocation))
                oracle_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        oracle_scale_prediction(area, p_hat),
                        p_hat,
                        topk=topk,
                        extra={**extra, "oracle_scale": 1.0},
                        allocation=allocation,
                    )
                )
                mean_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        mean_scale_prediction(area, p_hat, self.mean_outflow),
                        p_hat,
                        topk=topk,
                        extra={**extra, "mean_scale": 1.0},
                        allocation=allocation,
                    )
                )
            payload["gravity_learned_scale"][split] = learned_rows
            payload["gravity_oracle_scale"][split] = oracle_rows
            payload["gravity_mean_scale"][split] = mean_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": "gravity",
            "scale_coef": None if self.scale_coef is None else self.scale_coef.tolist(),
            "count_coef": None if self.count_coef is None else self.count_coef.tolist(),
            "scale_mean": None if self.scale_mean is None else self.scale_mean.tolist(),
            "scale_std": None if self.scale_std is None else self.scale_std.tolist(),
            "count_mean": None if self.count_mean is None else self.count_mean.tolist(),
            "count_std": None if self.count_std is None else self.count_std.tolist(),
            "mean_outflow": self.mean_outflow,
            "config": {},
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.scale_coef = np.asarray(checkpoint["scale_coef"], dtype=float)
        self.count_coef = np.asarray(checkpoint["count_coef"], dtype=float)
        self.scale_mean = np.asarray(checkpoint["scale_mean"], dtype=float)
        self.scale_std = np.asarray(checkpoint["scale_std"], dtype=float)
        self.count_mean = np.asarray(checkpoint["count_mean"], dtype=float)
        self.count_std = np.asarray(checkpoint["count_std"], dtype=float)
        self.mean_outflow = float(checkpoint.get("mean_outflow", self.mean_outflow))
