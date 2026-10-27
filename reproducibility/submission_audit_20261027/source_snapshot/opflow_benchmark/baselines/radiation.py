"""Classical radiation-style spatial interaction baseline."""

from __future__ import annotations

import time
from typing import Any

import torch

from opflow_benchmark.baselines.common import (
    EPS,
    mean_scale_prediction,
    oracle_scale_prediction,
    profile_record,
    resolve_device,
    tensor_area,
    timed_evaluate_area_allocation,
    timed_evaluate_area_predictions,
)
from opflow_benchmark.exposure import exogenous_opportunity_proxy, rank_prefix_exposure


class RadiationBaseline:
    """Radiation allocation using deployment-observable node-feature mass."""

    def __init__(
        self,
        device: str = "auto",
        rank_prefix_row_chunk_size: int | None = None,
        **_: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.rank_prefix_row_chunk_size = rank_prefix_row_chunk_size
        self.mean_outflow = 1.0
        self.profile: dict[str, float] = {}

    def fit(self, train_areas: list[dict[str, Any]], val_areas: list[dict[str, Any]]) -> dict[str, list[float]]:
        valid_outflows = [area["outflow"][area["valid_origin"]].float() for area in train_areas if bool(area["valid_origin"].any())]
        if valid_outflows:
            self.mean_outflow = float(torch.cat(valid_outflows).mean().item())
        return {"train_loss": [], "val_loss": []}

    def _allocation(self, area: dict[str, Any]) -> torch.Tensor:
        batch = tensor_area(area, self.device)
        mass = exogenous_opportunity_proxy(batch["x"])
        prefix = rank_prefix_exposure(mass, batch["distance_raw"], row_chunk_size=self.rank_prefix_row_chunk_size)
        origin_mass = mass[:, None]
        dest_mass = mass[None, :]
        numer = origin_mass * dest_mass
        denom = torch.clamp((origin_mass + prefix) * (origin_mass + dest_mass + prefix), min=EPS)
        score = torch.clamp(numer / denom, min=EPS)
        return score / torch.clamp(score.sum(dim=1, keepdim=True), min=EPS)

    def predict_area(self, area: dict[str, Any], variant: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        p_hat = self._allocation(area)
        if variant == "radiation_oracle_scale":
            outflow = area["outflow"].to(self.device)
        elif variant in {"radiation_learned_scale", "radiation_mean_scale"}:
            outflow = torch.full((p_hat.shape[0],), self.mean_outflow, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown radiation variant: {variant}")
        y_hat = outflow[:, None] * p_hat
        return y_hat, p_hat, {"mean_outflow": self.mean_outflow}

    def evaluate(self, split_areas: dict[str, list[dict[str, Any]]], topk: int = 10) -> dict[str, Any]:
        self.profile = {"predict_seconds": 0.0, "metric_seconds": 0.0, "areas": 0.0, "metric_rows": 0.0}
        payload = {"radiation_learned_scale": {}, "radiation_oracle_scale": {}, "radiation_mean_scale": {}}
        for split, areas in split_areas.items():
            learned_rows = []
            oracle_rows = []
            mean_rows = []
            for area in areas:
                start = time.perf_counter()
                y_hat, p_hat, extra = self.predict_area(area, "radiation_learned_scale")
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
            payload["radiation_learned_scale"][split] = learned_rows
            payload["radiation_oracle_scale"][split] = oracle_rows
            payload["radiation_mean_scale"][split] = mean_rows
        return payload

    def checkpoint(self) -> dict[str, Any]:
        return {
            "model_type": "radiation",
            "mean_outflow": self.mean_outflow,
            "config": {"rank_prefix_row_chunk_size": self.rank_prefix_row_chunk_size},
        }
