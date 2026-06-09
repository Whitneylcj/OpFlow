"""Exposure-potential feature helpers for real-world OpFlow."""

from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8


def row_center(values: torch.Tensor) -> torch.Tensor:
    """Center utilities within each origin row."""
    return values - values.mean(dim=1, keepdim=True)


def rank_prefix_exposure(opportunity: torch.Tensor, distance: torch.Tensor, row_chunk_size: int | None = None) -> torch.Tensor:
    """Strict rank-prefix exposure, excluding the destination itself.

    For every origin i and destination j, returns the sum of opportunities at
    destinations k with D_ik < D_ij. Tied distances share the same prefix mass.
    """
    if opportunity.ndim != 1:
        raise ValueError("opportunity must be a 1-D tensor")
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    n = int(opportunity.shape[0])
    if distance.shape[0] != n:
        raise ValueError("opportunity and distance dimensions do not match")

    result = torch.zeros((n, n), dtype=opportunity.dtype, device=opportunity.device)
    clean_distance = torch.nan_to_num(distance, nan=float("inf"), posinf=float("inf"), neginf=0.0)
    if n == 0:
        return result

    chunk_size = int(row_chunk_size or min(max(64, 262144 // max(n, 1)), n))
    chunk_size = max(min(chunk_size, n), 1)
    positions = torch.arange(n, device=opportunity.device, dtype=torch.long)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distance_chunk = clean_distance[start:end]
        order = torch.argsort(distance_chunk, dim=1, stable=True)
        sorted_distance = torch.gather(distance_chunk, 1, order)
        row_ids = torch.arange(start, end, device=opportunity.device, dtype=torch.long)[:, None]
        sorted_opp = opportunity[order].masked_fill(order == row_ids, 0.0)
        exclusive_prefix = sorted_opp.cumsum(dim=1) - sorted_opp

        group_start = torch.ones_like(sorted_distance, dtype=torch.bool)
        group_start[:, 1:] = sorted_distance[:, 1:] != sorted_distance[:, :-1]
        start_positions = torch.where(group_start, positions.expand(end - start, n), torch.zeros_like(order))
        group_start_positions = torch.cummax(start_positions, dim=1).values
        sorted_prefix = torch.gather(exclusive_prefix, 1, group_start_positions)
        result[start:end].scatter_(1, order, sorted_prefix)
    return result


def local_destination_exposure(
    opportunity: torch.Tensor,
    distance: torch.Tensor,
    omega: float,
    mode: str = "kernel_sum",
) -> torch.Tensor:
    """Destination-side local exposure, excluding self from local mass."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    if mode not in {"kernel_sum", "softmax"}:
        raise ValueError(f"Unknown local exposure mode: {mode}")
    n = int(opportunity.shape[0])
    clean_distance = torch.nan_to_num(distance, nan=float("inf"), posinf=float("inf"), neginf=0.0)
    self_mask = torch.eye(n, dtype=torch.bool, device=opportunity.device)
    logits = (-clean_distance / float(omega)).masked_fill(self_mask, -torch.inf)
    if mode == "softmax":
        weights = F.softmax(logits, dim=1)
    else:
        weights = torch.exp(logits)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights @ opportunity


def exogenous_opportunity_proxy(node_feat: torch.Tensor) -> torch.Tensor:
    """Non-label opportunity proxy used by classical exposure baselines."""
    proxy = torch.log1p(torch.clamp(node_feat, min=0.0)).sum(dim=1)
    return torch.clamp(proxy, min=0.0) + EPS
