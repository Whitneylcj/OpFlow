"""Pairwise feature MLP baseline for allocation-first OD prediction."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from opflow_benchmark.baselines.common import (
    FeatureStats,
    compose_prediction,
    timed_evaluate_area_allocation,
    masked_share_loss,
    mean_scale_prediction,
    oracle_scale_prediction,
    pair_node_features,
    positive_scale_from_log,
    resolve_device,
    tensor_area,
    profile_record,
    timed_evaluate_area_predictions,
)


class PairMLPNet(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        pair_dim = node_dim * 4 + 2
        self.utility = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.outflow = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, distance: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_x = pair_node_features(x)
        edge_x = torch.cat([pair_x, distance[..., None], adjacency[..., None]], dim=-1)
        logits = self.utility(edge_x).squeeze(-1)
        outflow_hat = positive_scale_from_log(self.outflow(x).squeeze(-1))
        return logits, outflow_hat


class PairMLPBaseline:
    """Scale-allocation pairwise MLP baseline."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 50,
        weight_decay: float = 0.0,
        lambda_count: float = 0.1,
        lambda_scale: float = 1.0,
        hidden_dim: int = 64,
        device: str = "auto",
        patience: int = 20,
        grad_clip: float = 5.0,
        max_pair_mlp_train_areas: int | None = None,
        report_name: str = "pair_mlp",
        **_: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.lambda_count = lambda_count
        self.lambda_scale = lambda_scale
        self.hidden_dim = hidden_dim
        self.patience = patience
        self.grad_clip = grad_clip
        self.max_pair_mlp_train_areas = max_pair_mlp_train_areas
        self.report_name = report_name
        self.mean_outflow = 1.0
        self.stats = FeatureStats()
        self.model: PairMLPNet | None = None
        self.profile: dict[str, float] = {}

    def _ensure_model(self, node_dim: int) -> None:
        if self.model is None:
            self.model = PairMLPNet(node_dim=node_dim, hidden_dim=self.hidden_dim).to(self.device)

    def _loss_area(self, area: dict[str, Any]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        batch = tensor_area(area, self.device, self.stats)
        logits, outflow_hat = self.model(batch["x"], batch["distance"], batch["adjacency"])
        y_hat, _ = compose_prediction(logits, outflow_hat)
        share = masked_share_loss(batch["p"], logits, batch["valid"])
        count = F.mse_loss(torch.log1p(y_hat), torch.log1p(batch["y"]))
        scale = F.mse_loss(torch.log1p(outflow_hat), torch.log1p(batch["outflow"]))
        return share + self.lambda_scale * scale + self.lambda_count * count

    def _eval_loss(self, areas: list[dict[str, Any]]) -> float:
        if not areas:
            return float("inf")
        with torch.no_grad():
            losses = [self._loss_area(area).detach() for area in areas]
        return float(torch.stack(losses).mean().cpu())

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        if not train_areas:
            raise ValueError("No training areas available")
        if self.max_pair_mlp_train_areas is not None and len(train_areas) > int(self.max_pair_mlp_train_areas):
            indices = torch.linspace(0, len(train_areas) - 1, steps=int(self.max_pair_mlp_train_areas)).long().tolist()
            train_areas = [train_areas[int(index)] for index in indices]
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            self.mean_outflow = float(torch.cat(valid_outflows).mean().item())
        self.stats.fit(train_areas)
        first = tensor_area(train_areas[0], self.device, self.stats)
        self._ensure_model(first["x"].shape[1])
        assert self.model is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        history = {"train_loss": [], "val_loss": []}
        best_value = float("inf")
        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        no_improve = 0
        val_ref = val_areas if val_areas else train_areas
        for _ in range(self.epochs):
            self.model.train()
            total = 0.0
            count = 0
            for area in train_areas:
                optimizer.zero_grad()
                loss = self._loss_area(area)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
            self.model.eval()
            train_loss = total / max(count, 1)
            val_loss = self._eval_loss(val_ref)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_value - 1e-6:
                best_value = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return history

    def predict_area(self, area: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.eval()
        with torch.no_grad():
            batch = tensor_area(area, self.device, self.stats)
            logits, outflow_hat = self.model(batch["x"], batch["distance"], batch["adjacency"])
            y_hat, p_hat = compose_prediction(logits, outflow_hat)
        return y_hat, p_hat, {"mean_pred_outflow": float(outflow_hat.mean().detach().cpu())}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload: dict[str, Any] = {
            f"{self.report_name}_learned_scale": {},
            f"{self.report_name}_oracle_scale": {},
            f"{self.report_name}_mean_scale": {},
        }
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
                y_oracle = oracle_scale_prediction(area, p_hat)
                oracle_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        y_oracle,
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
            payload[f"{self.report_name}_learned_scale"][split] = learned_rows
            payload[f"{self.report_name}_oracle_scale"][split] = oracle_rows
            payload[f"{self.report_name}_mean_scale"][split] = mean_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return {
            "model_type": "pair_mlp",
            "report_name": self.report_name,
            "model_state": self.model.state_dict(),
            "stats": self.stats.state_dict(),
            "node_dim": self.model.outflow[0].in_features,
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "lambda_count": self.lambda_count,
                "lambda_scale": self.lambda_scale,
                "hidden_dim": self.hidden_dim,
                "patience": self.patience,
                "grad_clip": self.grad_clip,
                "max_pair_mlp_train_areas": self.max_pair_mlp_train_areas,
                "mean_outflow": self.mean_outflow,
            },
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.stats = FeatureStats.from_state_dict(checkpoint["stats"])
        self.mean_outflow = float(checkpoint.get("config", {}).get("mean_outflow", self.mean_outflow))
        self._ensure_model(int(checkpoint["node_dim"]))
        assert self.model is not None
        self.model.load_state_dict(checkpoint["model_state"])
