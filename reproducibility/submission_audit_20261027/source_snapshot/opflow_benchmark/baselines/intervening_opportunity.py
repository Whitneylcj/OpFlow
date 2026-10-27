"""Classical intervening-opportunity exposure baseline."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
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
from opflow_benchmark.exposure import exogenous_opportunity_proxy, rank_prefix_exposure


class InterveningOpportunityBaseline:
    """Softmax utility from exogenous opportunity, rank exposure, and distance."""

    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 50,
        weight_decay: float = 0.0,
        device: str = "auto",
        patience: int = 20,
        io_cache_max_mb: float = 256.0,
        rank_prefix_row_chunk_size: int | None = None,
        progress_path: str | None = None,
        progress_prefix: str | None = None,
        **_: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.io_cache_max_bytes = max(int(float(io_cache_max_mb) * 1024 * 1024), 0)
        self.rank_prefix_row_chunk_size = rank_prefix_row_chunk_size
        self.progress_path = Path(progress_path) if progress_path else None
        self.progress_prefix = progress_prefix or "model=intervening_opportunity"
        self._feature_cache: OrderedDict[str, tuple[int, dict[str, torch.Tensor]]] = OrderedDict()
        self._feature_cache_bytes = 0
        self.raw_params = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self.device))
        self.mean_outflow = 1.0
        self.profile: dict[str, float] = {}

    def _format_seconds(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        return f"{minutes / 60.0:.2f}h"

    def _emit_progress(self, message: str) -> None:
        print(message, flush=True)
        if self.progress_path is None:
            return
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    def _cache_key(self, area: dict[str, Any]) -> str:
        return str(area.get("area_id", id(area)))

    def _feature_nbytes(self, features: dict[str, torch.Tensor]) -> int:
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in features.values())

    def _cache_features(self, key: str, features: dict[str, torch.Tensor]) -> None:
        if self.io_cache_max_bytes <= 0:
            return
        nbytes = self._feature_nbytes(features)
        if nbytes > self.io_cache_max_bytes:
            return
        if key in self._feature_cache:
            old_nbytes, _ = self._feature_cache.pop(key)
            self._feature_cache_bytes -= old_nbytes
        while self._feature_cache and self._feature_cache_bytes + nbytes > self.io_cache_max_bytes:
            _, (old_nbytes, _) = self._feature_cache.popitem(last=False)
            self._feature_cache_bytes -= old_nbytes
        self._feature_cache[key] = (nbytes, features)
        self._feature_cache_bytes += nbytes

    def _area_features(self, area: dict[str, Any], batch: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
        key = self._cache_key(area)
        cached = self._feature_cache.get(key)
        if cached is not None:
            self._feature_cache.move_to_end(key)
            return cached[1]
        if batch is None:
            batch = tensor_area(area, self.device)
        opp = exogenous_opportunity_proxy(batch["x"])
        prefix = rank_prefix_exposure(opp, batch["distance_raw"], row_chunk_size=self.rank_prefix_row_chunk_size)
        features = {
            "log_opp": torch.log1p(opp).detach(),
            "log_prefix": torch.log1p(prefix).detach(),
            "log_distance": torch.log1p(torch.clamp(batch["distance_raw"], min=0.0)).detach(),
        }
        self._cache_features(key, features)
        return features

    def _area_logits(self, area: dict[str, Any], batch: dict[str, Any] | None = None) -> torch.Tensor:
        features = self._area_features(area, batch=batch)
        alpha, gamma, beta = F.softplus(self.raw_params) + EPS
        return alpha * features["log_opp"][None, :] - gamma * features["log_prefix"] - beta * features["log_distance"]

    def _loss(self, areas: list[dict[str, Any]]) -> torch.Tensor:
        losses = []
        for area in areas:
            batch = tensor_area(area, self.device)
            logits = self._area_logits(area, batch=batch)
            if bool(batch["valid"].any()):
                logp = F.log_softmax(logits[batch["valid"]], dim=1)
                losses.append(-(batch["p"][batch["valid"]] * logp).sum(dim=1).mean())
        return torch.stack(losses).mean() if losses else self.raw_params.new_tensor(0.0)

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            self.mean_outflow = float(torch.cat(valid_outflows).mean().item())
        optimizer = torch.optim.Adam([self.raw_params], lr=self.lr, weight_decay=self.weight_decay)
        history = {"train_loss": [], "val_loss": []}
        best_value = float("inf")
        best_params = self.raw_params.detach().clone()
        no_improve = 0
        val_ref = val_areas if val_areas else train_areas
        start_time = time.perf_counter()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self._loss(train_areas)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss = float(self._loss(train_areas).detach().cpu())
                val_loss = float(self._loss(val_ref).detach().cpu())
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            completed = epoch + 1
            elapsed = time.perf_counter() - start_time
            avg = elapsed / completed
            eta = avg * (self.epochs - completed)
            self._emit_progress(
                "[real-world][epoch] "
                f"{self.progress_prefix} epoch={completed}/{self.epochs} "
                f"elapsed={self._format_seconds(elapsed)} eta={self._format_seconds(eta)} "
                f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
                f"cache_mb={self._feature_cache_bytes / (1024 * 1024):.1f}"
            )
            if val_loss < best_value - 1e-7:
                best_value = val_loss
                best_params = self.raw_params.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.raw_params.data.copy_(best_params)
        return history

    def predict_area(self, area: dict[str, Any], variant: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        logits = self._area_logits(area)
        p_hat = F.softmax(logits, dim=1)
        if variant == "intervening_opportunity_oracle_scale":
            outflow = area["outflow"].to(self.device)
        elif variant == "intervening_opportunity_learned_scale":
            outflow = torch.full((p_hat.shape[0],), self.mean_outflow, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown intervening-opportunity variant: {variant}")
        y_hat = outflow[:, None] * p_hat
        alpha, gamma, beta = (F.softplus(self.raw_params) + EPS).detach().cpu().tolist()
        return y_hat, p_hat, {"io_alpha": alpha, "io_gamma": gamma, "io_beta": beta, "mean_outflow": self.mean_outflow}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload: dict[str, Any] = {}
        variants = ("intervening_opportunity_learned_scale", "intervening_opportunity_oracle_scale")
        for variant in variants:
            payload[variant] = {}
        for split, areas in split_areas.items():
            learned_rows = []
            oracle_rows = []
            for area in areas:
                start = time.perf_counter()
                y_hat, p_hat, extra = self.predict_area(area, "intervening_opportunity_learned_scale")
                profile_record(self.profile, "predict_seconds", time.perf_counter() - start)
                self.profile["areas"] += 1.0
                allocation = timed_evaluate_area_allocation(self.profile, area, p_hat, topk=topk)
                learned_rows.append(timed_evaluate_area_predictions(self.profile, area, y_hat, p_hat, topk=topk, extra=extra, allocation=allocation))
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
            payload["intervening_opportunity_learned_scale"][split] = learned_rows
            payload["intervening_opportunity_oracle_scale"][split] = oracle_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": "intervening_opportunity",
            "raw_params": self.raw_params.detach().cpu().tolist(),
            "mean_outflow": self.mean_outflow,
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "io_cache_max_mb": self.io_cache_max_bytes / (1024 * 1024),
                "rank_prefix_row_chunk_size": self.rank_prefix_row_chunk_size,
            },
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.raw_params.data.copy_(torch.as_tensor(checkpoint["raw_params"], dtype=torch.float32, device=self.device))
        self.mean_outflow = float(checkpoint.get("mean_outflow", 1.0))
