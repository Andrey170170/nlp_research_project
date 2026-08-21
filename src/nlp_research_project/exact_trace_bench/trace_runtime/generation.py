"""Completion orchestration over the canonical sibling tracing API."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import legacy_capture_artifact_status
from .completion_workspace import CompletionWorkspace
from .observations import CompletionObservations
from .request import TracePolicy
from .step_artifacts import StepArtifactWriter
from .support import (
    build_completion_timing_summary,
    capture_resource_snapshot,
    generate_next_token,
)
from .telemetry import normalize_telemetry_events
from .tracing import DiagnosticTraceCompletion, extract_compact_chunked_attribution


@dataclass(frozen=True)
class CompletionPlan:
    """Generation and project-artifact policy for one completion."""

    temperature: float = 0.7
    max_steps: int = 256
    max_edges: int = 10_000
    incremental_telemetry_jsonl: bool = False
    prompt_token_count: int | None = None
    prompt_source: str = "gsm8k"
    fixture_name: str | None = None
    fixture_kind: str | None = None


@dataclass(frozen=True)
class TokenGenerationPolicy:
    temperature: float
    stop_token_ids: frozenset[int]

    @classmethod
    def for_model(cls, model: Any, *, temperature: float) -> TokenGenerationPolicy:
        tokenizer = model.tokenizer
        candidates = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int) and end_of_turn != tokenizer.unk_token_id:
            candidates.append(end_of_turn)
        return cls(
            temperature=temperature,
            stop_token_ids=frozenset(
                value for value in candidates if value is not None
            ),
        )

    def generate(self, model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
        return generate_next_token(
            model,
            input_ids,
            temperature=self.temperature,
        )

    def should_stop(self, token_result: dict[str, Any]) -> bool:
        return int(token_result["token_id"]) in self.stop_token_ids


@dataclass(frozen=True)
class TraceStepResult:
    compact_result: dict[str, Any]
    token_result: dict[str, Any]
    attribution_seconds: float
    token_generation_seconds: float
    stop: bool


@dataclass(frozen=True)
class DiagnosticStepResult:
    diagnostic: DiagnosticTraceCompletion
    attribution_seconds: float


def run_trace_step(
    *,
    model: Any,
    input_ids: torch.Tensor,
    trace_policy: TracePolicy,
    token_policy: TokenGenerationPolicy,
    workspace: CompletionWorkspace,
    step_index: int,
    incremental_telemetry: bool,
) -> TraceStepResult | DiagnosticStepResult:
    """Run canonical attribution and sample the next token for one prefix."""

    attribution_started = time.perf_counter()
    compact_result = extract_compact_chunked_attribution(
        model,
        input_ids[0],
        policy=trace_policy,
        telemetry_jsonl_path=(
            workspace.live_telemetry_path if incremental_telemetry else None
        ),
        telemetry_context=(
            {
                "schema_version": 1,
                "prompt_id": workspace.prompt_id,
                "completion_id": workspace.completion_id,
                "trace_step_index": step_index,
            }
            if incremental_telemetry
            else None
        ),
    )
    attribution_seconds = time.perf_counter() - attribution_started
    if isinstance(compact_result, DiagnosticTraceCompletion):
        return DiagnosticStepResult(
            diagnostic=compact_result,
            attribution_seconds=attribution_seconds,
        )
    generation_started = time.perf_counter()
    token_result = token_policy.generate(model, input_ids)
    generation_seconds = time.perf_counter() - generation_started
    return TraceStepResult(
        compact_result=compact_result,
        token_result=token_result,
        attribution_seconds=attribution_seconds,
        token_generation_seconds=generation_seconds,
        stop=token_policy.should_stop(token_result),
    )


def trace_completion_compact_chunked(
    model: Any,
    prompt: str,
    *,
    output_dir: Path,
    prompt_idx: int,
    completion_idx: int,
    completion: CompletionPlan,
    trace_policy: TracePolicy,
) -> dict[str, Any]:
    """Generate and persist one completion; orchestration intentionally stays small."""

    workspace = CompletionWorkspace.create(
        output_dir,
        prompt_index=prompt_idx,
        completion_index=completion_idx,
    )
    token_policy = TokenGenerationPolicy.for_model(
        model,
        temperature=completion.temperature,
    )
    writer = StepArtifactWriter(
        workspace=workspace,
        trace_policy=trace_policy,
        model=model,
        max_edges=completion.max_edges,
    )
    observations = CompletionObservations()
    tokenizer = model.tokenizer
    input_ids = model.ensure_tokenized(prompt).unsqueeze(0)
    initial_input_token_count = int(input_ids.shape[1])
    prompt_token_count = completion.prompt_token_count or initial_input_token_count
    generated_token_ids: list[int] = []
    started_wall = time.time()
    started_perf = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        f"\n  [{workspace.prompt_id}/{workspace.completion_id}] "
        f"Starting trace (temp={completion.temperature})"
    )

    for step_index in range(completion.max_steps):
        step_started = time.perf_counter()
        step = run_trace_step(
            model=model,
            input_ids=input_ids,
            trace_policy=trace_policy,
            token_policy=token_policy,
            workspace=workspace,
            step_index=step_index,
            incremental_telemetry=completion.incremental_telemetry_jsonl,
        )
        if isinstance(step, DiagnosticStepResult):
            capture_artifacts = writer.write_diagnostic_sidecars(
                step_index, step.diagnostic.diagnostic_artifacts or {}
            )
            telemetry_records = [
                {
                    "prompt_id": workspace.prompt_id,
                    "completion_id": workspace.completion_id,
                    "trace_step_index": step_index,
                    "event_index": event_index,
                    **event,
                }
                for event_index, event in enumerate(
                    normalize_telemetry_events(list(step.diagnostic.telemetry_events))
                )
            ]
            workspace.append_jsonl(workspace.telemetry_path, telemetry_records)
            manifest = {
                "status": "probe_completed",
                "prompt_id": workspace.prompt_id,
                "completion_id": workspace.completion_id,
                "prompt": prompt,
                "prompt_source": completion.prompt_source,
                "fixture_name": completion.fixture_name,
                "fixture_kind": completion.fixture_kind,
                "completion_text": "",
                "duration_seconds": round(time.time() - started_wall, 2),
                "prompt_token_count": prompt_token_count,
                "initial_input_token_count": initial_input_token_count,
                "generated_token_count": 0,
                "n_steps_traced": 0,
                "temperature": completion.temperature,
                "max_edges": completion.max_edges,
                "semantic_fingerprint": step.diagnostic.semantic_fingerprint,
                "execution_fingerprint": step.diagnostic.execution_fingerprint,
                "graph_packaging_mode": "diagnostic_no_graph",
                "diagnostic_stop_mode": step.diagnostic.diagnostic_stop_mode,
                "phase4_batches_completed": (step.diagnostic.phase4_batches_completed),
                "telemetry_summary": dict(step.diagnostic.telemetry_summary),
                "telemetry_event_count": len(telemetry_records),
                "sidecar_status": legacy_capture_artifact_status(capture_artifacts),
                "capture_artifact_status": capture_artifacts,
                "admission_report": step.diagnostic.admission_report,
                "resource_snapshot": capture_resource_snapshot(),
                "timing_summary": build_completion_timing_summary(
                    completion_end_to_end_seconds=time.perf_counter() - started_perf,
                    step_records=[],
                ),
                "steps": [],
            }
            manifest_path = workspace.write_json("completion.json", manifest)
            print(
                "    Diagnostic probe completed "
                f"({step.diagnostic.diagnostic_stop_mode}); "
                f"saved manifest: {manifest_path}"
            )
            return manifest
        generated_token_ids.append(int(step.token_result["token_id"]))
        record = writer.write(
            step_index=step_index,
            prefix_token_count=int(input_ids.shape[1]),
            compact_result=step.compact_result,
            token_result=step.token_result,
            attribution_seconds=step.attribution_seconds,
            token_generation_seconds=step.token_generation_seconds,
            step_started=step_started,
            stop=step.stop,
        )
        observations.record_trace_result(record)
        if step_index % 10 == 0 or step.stop:
            print(
                f"    Step {step_index:03d}: tok={step.token_result['token_text']!r} "
                f"feat={record['n_active_features']} edges={record['n_edges_retained']}"
            )
        input_ids = step.token_result["next_input_ids"]
        del step
        gc.collect()
        if record["stop_reason"] is not None:
            print(f"    Stop token at step {step_index}")
            break

    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    summary = observations.final_summary()
    manifest = {
        "status": "success",
        "prompt_id": workspace.prompt_id,
        "completion_id": workspace.completion_id,
        "prompt": prompt,
        "prompt_source": completion.prompt_source,
        "fixture_name": completion.fixture_name,
        "fixture_kind": completion.fixture_kind,
        "completion_text": completion_text,
        **summary,
        "duration_seconds": round(time.time() - started_wall, 2),
        "prompt_token_count": prompt_token_count,
        "initial_input_token_count": initial_input_token_count,
        "generated_token_count": len(generated_token_ids),
        "temperature": completion.temperature,
        "max_edges": completion.max_edges,
        "semantic_fingerprint": _last(observations, "semantic_fingerprint"),
        "execution_fingerprint": _last(observations, "execution_fingerprint"),
        "graph_packaging_mode": "compact_chunked_no_full_graph",
        "resource_snapshot": capture_resource_snapshot(),
        "timing_summary": build_completion_timing_summary(
            completion_end_to_end_seconds=time.perf_counter() - started_perf,
            step_records=observations.step_records,
        ),
        "steps": observations.step_records,
    }
    manifest_path = workspace.write_json("completion.json", manifest)
    print(f"    Saved manifest: {manifest_path}")
    print(f"    Answer (first 200 chars): {completion_text[:200]}")
    return manifest


def _last(observations: CompletionObservations, key: str) -> Any:
    if not observations.step_records:
        return None
    return observations.step_records[-1].get(key)
