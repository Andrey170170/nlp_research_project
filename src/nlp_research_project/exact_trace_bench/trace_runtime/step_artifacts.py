"""Project artifact persistence for one canonical trace step."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nlp_research_project.exact_trace_bench.compact_io import save_compact

from .artifacts import (
    legacy_capture_artifact_status,
    require_complete_capture_artifacts,
    write_capture_artifacts,
)
from .compact_graph import compact_result_to_step_data
from .completion_workspace import CompletionWorkspace
from .request import TracePolicy
from .support import capture_resource_snapshot, capture_transcoder_diagnostics
from .telemetry import normalize_telemetry_events


@dataclass
class StepArtifactWriter:
    workspace: CompletionWorkspace
    trace_policy: TracePolicy
    model: Any
    max_edges: int
    _debug_written: bool = False

    def write(
        self,
        *,
        step_index: int,
        prefix_token_count: int,
        compact_result: dict[str, Any],
        token_result: dict[str, Any],
        attribution_seconds: float,
        token_generation_seconds: float,
        step_started: float,
        stop: bool,
    ) -> dict[str, Any]:
        step_data = compact_result_to_step_data(
            compact_result,
            step_index,
            token_text=token_result["token_text"],
            logprob=token_result["token_logprob"],
            max_edges=self.max_edges,
        )
        artifact_started = time.perf_counter()
        save_compact(step_data, self.workspace.step_path(step_index))
        artifact_seconds = time.perf_counter() - artifact_started

        telemetry = self._write_telemetry(step_index, compact_result)
        capture_artifacts = self._write_sidecars(step_index, compact_result)
        self._persist_capture_artifact_status(step_index, capture_artifacts)
        require_complete_capture_artifacts(capture_artifacts)
        self._write_debug_artifacts(step_index, compact_result)
        runtime = jsonable_runtime_metadata(compact_result)
        return {
            **runtime,
            "step_index": step_index,
            "prefix_token_count": prefix_token_count,
            "generated_token_count": step_index + 1,
            "next_token_id": int(token_result["token_id"]),
            "next_token_text": str(token_result["token_text"]),
            "next_token_logprob": token_result["token_logprob"],
            "n_active_features": int(step_data.n_features),
            "n_edges_retained": len(step_data.weights),
            "stop_reason": "eos" if stop else None,
            "step_end_to_end_seconds": round(time.perf_counter() - step_started, 6),
            "attribution_seconds": round(attribution_seconds, 6),
            "token_generation_seconds": round(token_generation_seconds, 6),
            "artifact_save_seconds": round(artifact_seconds, 6),
            "telemetry_event_count": telemetry,
            "sidecar_status": legacy_capture_artifact_status(capture_artifacts),
            "capture_artifact_status": capture_artifacts,
            "resource_snapshot": capture_resource_snapshot(),
            "transcoder_diagnostics": capture_transcoder_diagnostics(self.model),
        }

    def _write_telemetry(self, step_index: int, result: dict[str, Any]) -> int:
        records = []
        for event_index, event in enumerate(
            normalize_telemetry_events(result.get("telemetry_events"))
        ):
            records.append(
                {
                    "prompt_id": self.workspace.prompt_id,
                    "completion_id": self.workspace.completion_id,
                    "trace_step_index": step_index,
                    "event_index": event_index,
                    **event,
                }
            )
        return self.workspace.append_jsonl(self.workspace.telemetry_path, records)

    def _write_sidecars(
        self, step_index: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        enabled = self.trace_policy.execution.observability
        requested = {
            "phase0_donor_bundle": enabled.capture_phase0_donor_bundle,
            "phase3_seed_bundle": enabled.capture_phase3_seed_bundle,
            "phase3_gradient_bundle": enabled.capture_phase3_gradient_bundle,
            "phase3_row_bundle": enabled.capture_phase3_row_bundle,
            "feature_semantic_descriptors": enabled.capture_feature_semantic_descriptors,
        }
        return write_capture_artifacts(
            requested=requested,
            payloads=result,
            path_for=lambda name: self.workspace.sidecar_path(step_index, name),
        )

    def write_diagnostic_sidecars(
        self, step_index: int, artifacts: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Persist bounded captures returned by a graph-free diagnostic probe."""

        report = self._write_sidecars(step_index, dict(artifacts))
        self._persist_capture_artifact_status(step_index, report)
        require_complete_capture_artifacts(report)
        return report

    def _persist_capture_artifact_status(
        self, step_index: int, report: Mapping[str, Any]
    ) -> None:
        self.workspace.write_json(
            f"step_{step_index:03d}_capture_artifacts.json", dict(report)
        )

    def _write_debug_artifacts(self, step_index: int, result: dict[str, Any]) -> None:
        anomaly = result.get("phase4_anomaly_debug")
        if isinstance(anomaly, dict):
            self.workspace.write_json("phase4_anomaly_debug.json", anomaly)
        if self._debug_written:
            return
        summary = result.get("cross_cluster_debug_summary")
        if isinstance(summary, dict):
            self.workspace.write_json("cross_cluster_debug.json", summary)
        for key, filename in (
            (
                "cross_cluster_debug_checkpoints",
                "cross_cluster_debug_checkpoints.jsonl",
            ),
            ("cross_cluster_debug_batches", "cross_cluster_debug_batches.jsonl"),
        ):
            payload = result.get(key)
            if isinstance(payload, list):
                records = [
                    {
                        "prompt_id": self.workspace.prompt_id,
                        "completion_id": self.workspace.completion_id,
                        "trace_step_index": step_index,
                        **record,
                    }
                    for record in payload
                    if isinstance(record, dict)
                ]
                self.workspace.append_jsonl(self.workspace.root / filename, records)
        self._debug_written = any(
            result.get(key) is not None
            for key in (
                "cross_cluster_debug_summary",
                "cross_cluster_debug_checkpoints",
                "cross_cluster_debug_batches",
            )
        )


def jsonable_runtime_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Keep detailed scalar/runtime diagnostics while excluding graph payloads."""

    excluded = {
        "feature_feature_edges",
        "logit_feature_edges",
        "feature_row_node_indices",
        "selected_features",
        "active_features",
        "phase0_donor_bundle",
        "phase3_seed_bundle",
        "phase3_gradient_bundle",
        "phase3_row_bundle",
        "feature_semantic_descriptors",
        "telemetry_events",
        "cross_cluster_debug_checkpoints",
        "cross_cluster_debug_batches",
    }
    return {
        key: converted
        for key, value in result.items()
        if key not in excluded and (converted := _jsonable(value)) is not None
    }


def _jsonable(value: Any) -> Any | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): converted
            for key, item in value.items()
            if (converted := _jsonable(item)) is not None
        }
    if isinstance(value, (list, tuple)) and len(value) <= 4096:
        converted = [_jsonable(item) for item in value]
        return converted if all(item is not None for item in converted) else None
    if isinstance(value, torch.Tensor) and value.numel() <= 64:
        return value.detach().cpu().tolist()
    return None
