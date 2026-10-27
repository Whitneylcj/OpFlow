"""Baseline models for the OpFlow real-world benchmark."""

from __future__ import annotations

from opflow_benchmark.baselines.distance_only import DistanceOnlyBaseline
from opflow_benchmark.baselines.gravity import GravityBaseline
from opflow_benchmark.baselines.intervening_opportunity import InterveningOpportunityBaseline
from opflow_benchmark.baselines.opflow_lite import OpFlowLiteBaseline
from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline
from opflow_benchmark.baselines.opportunity_only import OpportunityOnlyBaseline
from opflow_benchmark.baselines.pair_mlp import PairMLPBaseline
from opflow_benchmark.baselines.radiation import RadiationBaseline
from opflow_benchmark.baselines.raw_count_mlp import RawCountMLPBaseline
from opflow_benchmark.baselines.scale_aware_mlp import ScaleAwareMLPBaseline


OPFLOW_REAL_VARIANTS = {
    "opflow_full": "opflow_full",
    "opflow_no_rank_exposure": "opflow_no_rank_exposure",
    "opflow_no_competition": "opflow_no_competition",
    "opflow_no_opportunity_field": "opflow_no_opportunity_field",
    "opflow_no_scale_detach": "opflow_no_scale_detach",
    "opflow_no_scale_branch": "opflow_no_scale_branch",
    "opflow_no_exposure_ops": "opflow_no_exposure_ops",
    "opflow_no_row_centering": "opflow_no_row_centering",
    "opflow_full_softmax_local": "opflow_full_softmax_local",
    "opflow_generic_potential_mlp": "opflow_generic_potential_mlp",
    "opflow_no_groupdro": "opflow_no_groupdro",
    "opflow_no_vrex": "opflow_no_vrex",
    "opflow_no_monotone_penalty": "opflow_no_monotone_penalty",
}


def build_model(model_name: str, **kwargs):
    if model_name == "gravity":
        return GravityBaseline(**kwargs)
    if model_name == "intervening_opportunity":
        return InterveningOpportunityBaseline(**kwargs)
    if model_name == "distance_only":
        return DistanceOnlyBaseline(**kwargs)
    if model_name == "opportunity_only":
        return OpportunityOnlyBaseline(**kwargs)
    if model_name == "pair_mlp":
        return PairMLPBaseline(**kwargs)
    if model_name == "pair_mlp_sampled":
        return PairMLPBaseline(report_name="pair_mlp_sampled", **kwargs)
    if model_name == "raw_count_mlp":
        return RawCountMLPBaseline(**kwargs)
    if model_name == "radiation":
        return RadiationBaseline(**kwargs)
    if model_name == "scale_aware_mlp":
        return ScaleAwareMLPBaseline(**kwargs)
    if model_name == "opflow_lite":
        return OpFlowLiteBaseline(**kwargs)
    if model_name == "opflow_real":
        return OpFlowRealBaseline(model_variant="opflow_full", report_name=model_name, **kwargs)
    if model_name in OPFLOW_REAL_VARIANTS:
        return OpFlowRealBaseline(model_variant=OPFLOW_REAL_VARIANTS[model_name], report_name=model_name, **kwargs)
    if model_name.startswith("official_"):
        from opflow_benchmark.baselines.official import build_official_model

        return build_official_model(model_name, **kwargs)
    raise ValueError(f"Unknown model: {model_name}")
