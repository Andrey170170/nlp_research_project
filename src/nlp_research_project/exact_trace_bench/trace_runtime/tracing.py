"""Canonical per-step trace execution."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from circuit_tracer import trace_one

from .request import TracePolicy


@dataclass(frozen=True)
class DiagnosticTraceCompletion:
    """Project-owned terminal diagnostic; intentionally contains no graph."""

    semantic_fingerprint: str
    execution_fingerprint: str
    telemetry_summary: Mapping[str, Any]
    telemetry_events: tuple[Mapping[str, Any], ...]
    admission_report: Mapping[str, Any] | None = None
    diagnostic_artifacts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def diagnostic_stop_mode(self) -> str | None:
        value = self.telemetry_summary.get("diagnostic_stop_mode")
        return value if isinstance(value, str) else None

    @property
    def phase4_batches_completed(self) -> int:
        value = self.telemetry_summary.get("phase4_batches_completed", 0)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def extract_compact_chunked_attribution(
    model: Any,
    prompt: str | torch.Tensor | list[int],
    *,
    policy: TracePolicy,
    telemetry_jsonl_path: str | Path | None = None,
    telemetry_context: dict[str, Any] | None = None,
) -> dict[str, Any] | DiagnosticTraceCompletion:
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
    if getattr(result.status, "value", result.status) == "refused":
        report = result.admission_report
        reasons = () if report is None else report.refusals
        detail = "; ".join(reasons) if reasons else "no refusal reason was recorded"
        raise RuntimeError(f"trace refused by memory governor: {detail}")
    if getattr(result.status, "value", result.status) == "probe_completed":
        diagnostic_artifacts = (
            dict(result.output) if isinstance(result.output, dict) else {}
        )
        return DiagnosticTraceCompletion(
            semantic_fingerprint=result.semantic_fingerprint,
            execution_fingerprint=result.execution_fingerprint,
            telemetry_summary=dict(result.telemetry_summary),
            telemetry_events=tuple(result.telemetry_events),
            admission_report=(
                None
                if result.admission_report is None
                else asdict(result.admission_report)
            ),
            diagnostic_artifacts=diagnostic_artifacts,
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
    output.setdefault("telemetry_events", list(result.telemetry_events))
    if result.admission_report is not None:
        output.setdefault("admission_report", asdict(result.admission_report))
    return output
