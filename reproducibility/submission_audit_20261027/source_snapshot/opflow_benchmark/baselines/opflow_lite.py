"""Compatibility alias for the P-only real-world OpFlow baseline."""

from __future__ import annotations

from typing import Any

from opflow_benchmark.baselines.opflow_real import OpFlowRealBaseline


class OpFlowLiteBaseline(OpFlowRealBaseline):
    """Preserve the historical model name while using the canonical P-only OpFlow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("lambda_chal", None)
        kwargs["model_variant"] = kwargs.pop("model_variant", "opflow_full")
        kwargs["report_name"] = kwargs.pop("report_name", "opflow_lite")
        super().__init__(*args, **kwargs)
