"""Distance-only allocation baseline."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from opflow_benchmark.baselines.common import (
    EPS,
    timed_evaluate_area_allocation,
    profile_record,
    resolve_device,
    tensor_area,
    timed_evaluate_area_predictions,
)


class DistanceOnlyBaseline:
    """Fit a softmax distance-decay utility and two scale variants."""

    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 50,
        weight_decay: float = 0.0,
        device: str = "auto",
        patience: int = 20,
        **_: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.log_beta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.mean_outflow = 1.0
        self.profile: dict[str, float] = {}

    def _predict_share(self, area: dict[str, Any]) -> torch.Tensor:
        batch = tensor_area(area, self.device)
        beta = F.softplus(self.log_beta) + EPS
        logits = -beta * torch.log1p(torch.clamp(batch["distance_raw"], min=0.0))
        return F.softmax(logits, dim=1)

    def _loss(self, areas: list[dict[str, Any]]) -> torch.Tensor:
        losses = []
        for area in areas:
            batch = tensor_area(area, self.device)
            beta = F.softplus(self.log_beta) + EPS
            logits = -beta * torch.log1p(torch.clamp(batch["distance_raw"], min=0.0))
            if batch["valid"].sum() > 0:
                logp = F.log_softmax(logits[batch["valid"]], dim=1)
                losses.append(-(batch["p"][batch["valid"]] * logp).sum(dim=1).mean())
        if not losses:
            return self.log_beta.new_tensor(0.0)
        return torch.stack(losses).mean()

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        valid_outflows = []
        for area in train_areas:
            valid = area["valid_origin"]
            if valid.any():
                valid_outflows.append(area["outflow"][valid].float())
        if valid_outflows:
            self.mean_outflow = float(torch.cat(valid_outflows).mean().item())

        optimizer = torch.optim.Adam([self.log_beta], lr=self.lr, weight_decay=self.weight_decay)
        history = {"train_loss": [], "val_loss": []}
        best_value = float("inf")
        best_beta = self.log_beta.detach().clone()
        no_improve = 0
        val_ref = val_areas if val_areas else train_areas
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = self._loss(train_areas)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss = float(self._loss(train_areas).detach().cpu())
                val_loss = float(self._loss(val_ref).detach().cpu())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_value - 1e-7:
                best_value = val_loss
                best_beta = self.log_beta.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.log_beta.data.copy_(best_beta)
        return history

    def predict_area(self, area: dict[str, Any], variant: str = "mean_scale") -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        p_hat = self._predict_share(area)
        if variant == "distance_only_oracle_scale":
            outflow = area["outflow"].to(self.device)
        elif variant == "distance_only_learned_scale":
            outflow = torch.full((p_hat.shape[0],), self.mean_outflow, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown distance-only variant: {variant}")
        y_hat = outflow[:, None] * p_hat
        extra = {"distance_beta": float((F.softplus(self.log_beta) + EPS).detach().cpu()), "mean_outflow": self.mean_outflow}
        return y_hat, p_hat, extra

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload: dict[str, Any] = {"distance_only_learned_scale": {}, "distance_only_oracle_scale": {}}
        for split, areas in split_areas.items():
            learned_rows = []
            oracle_rows = []
            for area in areas:
                start = time.perf_counter()
                p_hat = self._predict_share(area)
                extra = {"distance_beta": float((F.softplus(self.log_beta) + EPS).detach().cpu()), "mean_outflow": self.mean_outflow}
                profile_record(self.profile, "predict_seconds", time.perf_counter() - start)
                self.profile["areas"] += 1.0
                allocation = timed_evaluate_area_allocation(self.profile, area, p_hat, topk=topk)
                learned_outflow = torch.full((p_hat.shape[0],), self.mean_outflow, dtype=torch.float32, device=self.device)
                learned_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        learned_outflow[:, None] * p_hat,
                        p_hat,
                        topk=topk,
                        extra=extra,
                        allocation=allocation,
                    )
                )
                oracle_outflow = area["outflow"].to(self.device)
                oracle_rows.append(
                    timed_evaluate_area_predictions(
                        self.profile,
                        area,
                        oracle_outflow[:, None] * p_hat,
                        p_hat,
                        topk=topk,
                        extra={**extra, "oracle_scale": 1.0},
                        allocation=allocation,
                    )
                )
            payload["distance_only_learned_scale"][split] = learned_rows
            payload["distance_only_oracle_scale"][split] = oracle_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": "distance_only",
            "log_beta": float(self.log_beta.detach().cpu()),
            "mean_outflow": self.mean_outflow,
            "config": {"lr": self.lr, "epochs": self.epochs, "weight_decay": self.weight_decay, "patience": self.patience},
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.log_beta.data.copy_(torch.tensor(float(checkpoint["log_beta"]), dtype=torch.float32, device=self.device))
        self.mean_outflow = float(checkpoint.get("mean_outflow", 1.0))
