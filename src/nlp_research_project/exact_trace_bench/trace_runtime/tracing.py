"""Canonical per-step trace execution."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from circuit_tracer import trace_one

from .request import TracePolicy


def extract_compact_chunked_attribution(
    model: Any,
    prompt: str | torch.Tensor | list[int],
    *,
    policy: TracePolicy,
    telemetry_jsonl_path: str | Path | None = None,
    telemetry_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one canonical typed trace and return its compact graph payload."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = trace_one(
        policy.request(
            model=model,
            prompt=prompt,
            telemetry_jsonl_path=telemetry_jsonl_path,
            telemetry_context=telemetry_context,
        ),
        resources=policy.resources,
        provider_profile=policy.provider_profile,
    )
    if not isinstance(result.output, dict):
        raise TypeError(
            "compact exact tracing must return a dictionary payload; "
            f"got {type(result.output).__name__}"
        )
    output = dict(result.output)
    output.setdefault("semantic_fingerprint", result.semantic_fingerprint)
    output.setdefault("execution_fingerprint", result.execution_fingerprint)
    output.setdefault("telemetry_summary", dict(result.telemetry_summary))
    return output
