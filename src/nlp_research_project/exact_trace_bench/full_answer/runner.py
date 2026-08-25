from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        TypedCompactGraph,
    )

from ..backward_selection import build_backward_plan
from ..io_utils import ensure_dir, write_json, write_jsonl
from ..trace_runtime.artifacts import (
    legacy_capture_artifact_status,
    require_complete_capture_artifacts,
    write_capture_artifacts,
)
from ..transcoder_config import (
    PUBLIC_TRANSCODER_KNOB_KEYS,
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)
from .execution_control import (
    RuntimeResourcePolicy,
    execution_resolution,
    resource_sample,
    selected_execution_record,
    validate_resource_policy,
)
from .schemas import TraceSpec, load_shards, load_trace_specs, load_trajectory


class _DiagnosticProbeCompleted(Exception):
    """Internal non-error control flow after a terminal diagnostic trace."""


def _shard_dir(output_root: Path, shard_id: int) -> Path:
    return output_root / "shards" / f"shard_{shard_id:03d}"


def select_shard(shards: Mapping[str, Any], shard_id: int) -> dict[str, Any]:
    shard_rows = shards.get("shards")
    if not isinstance(shard_rows, list):
        raise ValueError("shards payload missing shards list")
    for shard in shard_rows:
        if isinstance(shard, dict) and shard.get("shard_id") == shard_id:
            return cast(dict[str, Any], shard)
    raise ValueError(f"shard_id not found: {shard_id}")


def assigned_specs(specs: list[TraceSpec], shard: Mapping[str, Any]) -> list[TraceSpec]:
    selected: list[TraceSpec] = []
    for index in shard.get("spec_indices", []):
        if not isinstance(index, int) or index < 0 or index >= len(specs):
            raise ValueError(f"spec index out of bounds for shard: {index}")
        selected.append(specs[index])
    return selected


def load_shard_inputs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> tuple[dict[str, Any], list[TraceSpec], dict[str, Any]]:
    trajectory = load_trajectory(trajectory_path)
    specs = load_trace_specs(trace_specs_path)
    shards = load_shards(shards_path)
    shard = select_shard(shards, shard_id)
    selected = assigned_specs(specs, shard)
    trajectory_id = trajectory["trajectory_id"]
    for spec in selected:
        if spec["trajectory_id"] != trajectory_id:
            raise ValueError("trace spec trajectory_id does not match trajectory")
        validate_trace_spec_against_trajectory(trajectory, spec)
    return cast(dict[str, Any], trajectory), selected, shard


def validate_trace_spec_against_trajectory(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> None:
    """Fail before attribution if position metadata diverges from the prefix."""
    prompt_token_count = trajectory.get("prompt_token_count")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_count, int) or prompt_token_count < 0:
        raise ValueError("trajectory prompt_token_count must be a non-negative int")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated_index = spec["generated_index"]
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    expected_position = prompt_token_count + generated_index
    if spec["target_position"] != expected_position:
        raise ValueError(
            "trace spec target_position does not match "
            "prompt_token_count + generated_index "
            f"({spec['target_position']} != {expected_position})"
        )
    token = generated_tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    if token.get("absolute_token_position") != expected_position:
        raise ValueError(
            "trajectory generated token absolute_token_position does not match "
            "prompt_token_count + generated_index"
        )


def list_shard_specs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> list[dict[str, Any]]:
    _, specs, _ = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    return [_summary_row(spec, shard_id=shard_id) for spec in specs]


def print_shard_specs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"shard={row['shard_id']} generated_index={row['generated_index']} "
            f"target_position={row['target_position']} trace_id={row['trace_id']} "
            f"target={row['target_token_id']} "
            f"text={row['target_token_text']!r} cost={row['estimated_cost']}"
        )


def dry_run_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
) -> dict[str, Any]:
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    selected_execution = selected_execution_record(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        specs=specs,
        shard=shard,
        metadata=_runtime_metadata(
            run_id=None,
            run_name="dry run",
            run_description=None,
            run_goal=None,
        ),
    )
    write_json(root / "selected_execution.json", selected_execution)
    print(
        json.dumps(
            {
                "event": "selected_execution",
                "selection_fingerprint": selected_execution["selection_fingerprint"],
                "feature_row_influence": selected_execution["mechanism_selection"],
                "backward_engine_mode": (
                    selected_execution["mechanism_selection"].get(
                        "backward_engine_mode"
                    )
                    if isinstance(selected_execution["mechanism_selection"], Mapping)
                    else None
                ),
                "resources": selected_execution["resources"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    shard_payload = {
        "schema_version": 1,
        "status": "dry_run",
        "shard": shard,
        "trajectory_id": trajectory["trajectory_id"],
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": len(specs),
        "target_positions": [spec["target_position"] for spec in specs],
    }
    write_json(root / "shard.json", shard_payload)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        trace = _trace_payload(spec, shard_id=shard_id)
        token_dir = root / f"token_{spec['generated_index']:06d}"
        write_json(token_dir / "trace.json", trace)
        rows.append(trace)
    write_jsonl(root / "trace_results.jsonl", rows)
    return {"shard_dir": str(root), "token_count": len(rows), "status": "dry_run"}


def reconstruct_prefix_token_ids(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> list[int]:
    """Return frozen prompt + generated prefix immediately before target token."""
    generated_index = spec["generated_index"]
    prompt_token_ids = trajectory.get("prompt_token_ids")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("trajectory prompt_token_ids must be a list of ints")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    prefix_generated = [
        int(token["token_id"])
        for token in generated_tokens[:generated_index]
        if isinstance(token, Mapping) and isinstance(token.get("token_id"), int)
    ]
    if len(prefix_generated) != generated_index:
        raise ValueError("generated prefix contains malformed token rows")
    prefix = [int(token_id) for token_id in prompt_token_ids] + prefix_generated
    if spec["prefix_token_count"] != len(prefix):
        raise ValueError(
            "trace spec prefix_token_count does not match reconstructed prefix "
            f"({spec['prefix_token_count']} != {len(prefix)})"
        )
    if spec["target_position"] != len(prefix):
        raise ValueError(
            "trace spec target_position does not match reconstructed target "
            f"position ({spec['target_position']} != {len(prefix)})"
        )
    return prefix


def reconstruct_full_sequence_token_ids(trajectory: Mapping[str, Any]) -> list[int]:
    prompt_token_ids = trajectory.get("prompt_token_ids")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("trajectory prompt_token_ids must be a list of ints")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated = [
        int(token["token_id"])
        for token in generated_tokens
        if isinstance(token, Mapping) and isinstance(token.get("token_id"), int)
    ]
    if len(generated) != len(generated_tokens):
        raise ValueError("generated sequence contains malformed token rows")
    return [int(token_id) for token_id in prompt_token_ids] + generated


def _hash_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prefix_view_metadata(
    trajectory: Mapping[str, Any],
    spec: TraceSpec,
    prefix_token_ids: list[int],
    *,
    full_sequence_token_ids: list[int] | None = None,
    full_sequence_token_ids_sha256: str | None = None,
) -> dict[str, Any]:
    knobs = spec.get("graph_knobs", {})
    input_context_mode = (
        knobs.get("input_context_mode") if isinstance(knobs, Mapping) else None
    )
    metadata = {
        "schema_version": 1,
        "mode": "full_sequence_target_position"
        if input_context_mode == "full_sequence"
        else "independent_prefix",
        "trajectory_id": trajectory["trajectory_id"],
        "trace_id": spec["trace_id"],
        "target_position": spec["target_position"],
        "prefix_token_count": spec["prefix_token_count"],
        "target_token_ids": [spec["target_token_id"]],
        "prefix_token_ids_sha256": _hash_token_ids(prefix_token_ids),
    }
    if input_context_mode == "full_sequence":
        full_sequence = full_sequence_token_ids or reconstruct_full_sequence_token_ids(
            trajectory
        )
        full_sequence_hash = full_sequence_token_ids_sha256 or _hash_token_ids(
            full_sequence
        )
        metadata["output_position"] = spec["target_position"] - 1
        metadata["input_token_count"] = len(full_sequence)
        metadata["full_sequence_token_count"] = len(full_sequence)
        metadata["input_token_ids_sha256"] = full_sequence_hash
    return metadata


def prepare_full_sequence_cache(
    trajectory: Mapping[str, Any], specs: list[TraceSpec]
) -> dict[str, Any] | None:
    """Reconstruct full-sequence ids/hash once when any shard spec requests it."""
    if not any(
        spec.get("graph_knobs", {}).get("input_context_mode") == "full_sequence"
        for spec in specs
    ):
        return None
    token_ids = reconstruct_full_sequence_token_ids(trajectory)
    return {"token_ids": token_ids, "token_ids_sha256": _hash_token_ids(token_ids)}


def session_reuse_metadata(spec: TraceSpec) -> dict[str, Any]:
    knobs = spec.get("graph_knobs", {})
    mode = knobs.get("trajectory_session_mode", "per_token")
    requested = mode not in (None, "per_token")
    return {
        "trajectory_session_mode": mode,
        "session_reuse_requested": bool(requested),
        "session_reuse_effective": False,
        "session_reuse_fallback": "per_token_path" if requested else None,
        "reuse_phase0_window_state_requested": bool(
            knobs.get("reuse_phase0_window_state", False)
        ),
        "reuse_phase0_window_state_effective": False,
        "reuse_target_logits_requested": bool(knobs.get("reuse_target_logits", False)),
        "reuse_target_logits_effective": False,
        "target_logit_source": "per_target_trace",
    }


def _shard_windows(
    specs: list[TraceSpec], shard: Mapping[str, Any]
) -> list[dict[str, Any]]:
    original_indices = shard.get("spec_indices", [])
    original_to_spec: dict[int, TraceSpec] = {}
    if isinstance(original_indices, list) and len(original_indices) == len(specs):
        for original_index, spec in zip(original_indices, specs):
            if isinstance(original_index, int):
                original_to_spec[original_index] = spec
    windows = shard.get("windows")
    result: list[dict[str, Any]] = []
    if isinstance(windows, list) and original_to_spec:
        for window_id, window in enumerate(windows):
            if not isinstance(window, Mapping):
                continue
            window_mapping = cast(Mapping[str, Any], window)
            window_spec_indices = window_mapping.get("spec_indices")
            if not isinstance(window_spec_indices, list):
                continue
            window_specs = [
                original_to_spec[index]
                for index in window_spec_indices
                if isinstance(index, int) and index in original_to_spec
            ]
            if window_specs:
                payload = dict(window_mapping)
                payload["window_id"] = window_id
                payload["specs"] = window_specs
                result.append(payload)
    if not result:
        if not specs:
            return []
        result = [
            {
                "window_id": 0,
                "window_start_generated_index": min(
                    int(spec["generated_index"]) for spec in specs
                ),
                "window_end_generated_index": max(
                    int(spec["generated_index"]) for spec in specs
                ),
                "window_max_target_position": max(
                    int(spec["target_position"]) for spec in specs
                ),
                "target_positions": [int(spec["target_position"]) for spec in specs],
                "specs": specs,
            }
        ]
    return _bound_correctness_windows(result)


def _bound_correctness_windows(
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep live correctness evidence bounded to one completed trace at a time."""

    bounded: list[dict[str, Any]] = []
    for window in windows:
        window_specs = cast(list[TraceSpec], window["specs"])
        correctness_enabled = any(
            spec.get("graph_knobs", {}).get("correctness_probe_mode", "off")
            != "off"
            for spec in window_specs
        )
        if not correctness_enabled:
            bounded.append(window)
            continue
        source_window_id = window.get("window_id")
        for spec in window_specs:
            generated_index = int(spec["generated_index"])
            target_position = int(spec["target_position"])
            payload = dict(window)
            payload.update(
                {
                    "window_id": len(bounded),
                    "source_window_id": source_window_id,
                    "window_start_generated_index": generated_index,
                    "window_end_generated_index": generated_index,
                    "window_max_target_position": target_position,
                    "target_positions": [target_position],
                    "specs": [spec],
                    "correctness_single_trace_window": True,
                }
            )
            bounded.append(payload)
    return bounded


def _validate_window_session_specs(window_specs: list[TraceSpec]) -> None:
    window_modes = {
        spec.get("graph_knobs", {}).get("trajectory_session_mode", "per_token")
        for spec in window_specs
    }
    if "window_reuse_v1" not in window_modes:
        return
    for spec in window_specs:
        knobs = spec.get("graph_knobs", {})
        if knobs.get("trajectory_session_mode") != "window_reuse_v1":
            continue
        if knobs.get("input_context_mode") != "full_sequence":
            raise ValueError(
                "window_reuse_v1 requires input_context_mode='full_sequence'"
            )
    shared_keys = (
        "reuse_phase0_window_state",
        "reuse_target_logits",
        "phase0_window_scope",
        "phase0_window_max_prefix_policy",
        "phase0_window_reference_checks",
    )
    first_knobs = next(
        spec["graph_knobs"]
        for spec in window_specs
        if spec.get("graph_knobs", {}).get("trajectory_session_mode")
        == "window_reuse_v1"
    )
    for spec in window_specs:
        knobs = spec.get("graph_knobs", {})
        if knobs.get("trajectory_session_mode") != "window_reuse_v1":
            continue
        for key in shared_keys:
            if knobs.get(key) != first_knobs.get(key):
                raise ValueError(f"window_reuse_v1 specs in a window must share {key}")


def _normalize_target_logit_source(source: Any) -> str:
    if source in (None, "context", "override"):
        return "per_target_trace"
    return str(source)


def forced_target_payload(spec: TraceSpec) -> dict[str, Any]:
    return {
        "token_id": spec["target_token_id"],
        "token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "attribution_targets": [spec["target_token_id"]],
    }


def _runtime_metadata(
    *,
    run_id: str | None,
    run_name: str | None,
    run_description: str | None,
    run_goal: str | None,
) -> dict[str, Any]:
    workspace_root = os.environ.get("WORKSPACE_ROOT")
    library_workspace_root = os.environ.get("LIB_WORKSPACE_ROOT")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return {
        "run_id": run_id or os.environ.get("RUN_ID") or None,
        "run_name": run_name or os.environ.get("RUN_NAME") or None,
        "run_description": run_description or os.environ.get("RUN_DESCRIPTION") or None,
        "run_goal": run_goal or os.environ.get("RUN_GOAL") or None,
        "workspace_root": workspace_root,
        "library_workspace_root": library_workspace_root,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
    }


def _shard_record(
    *,
    status: str,
    shard: Mapping[str, Any],
    trajectory_id: str,
    trace_specs_path: Path,
    shards_path: Path,
    token_count: int,
    metadata: Mapping[str, Any],
    target_positions: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "shard": dict(shard),
        "trajectory_id": trajectory_id,
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": token_count,
        "target_positions": target_positions or [],
    }
    payload.update(metadata)
    return payload


def _trace_runtime_payload(
    *,
    spec: TraceSpec,
    shard_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _trace_payload(spec, shard_id=shard_id)
    payload.update(metadata)
    return payload


def _target_token_metadata(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> dict[str, Any]:
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated_index = spec["generated_index"]
    token = generated_tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    return {
        "target_logprob": token.get("logprob"),
        "target_probability": token.get("probability"),
        "target_rank": token.get("rank"),
    }


def _graph_summary(
    step: Mapping[str, Any] | Any, graph_path: Path | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {"saved": graph_path is not None}
    if graph_path is not None:
        summary["graph_path"] = str(graph_path)
    if isinstance(step, Mapping):
        for key in (
            "node_count",
            "edge_count",
            "feature_node_count",
            "max_feature_nodes",
            "active_feature_count",
        ):
            if key in step:
                summary[key] = step[key]
        nested = step.get("summary")
        if isinstance(nested, Mapping):
            for key in ("node_count", "edge_count", "feature_node_count"):
                if key in nested and key not in summary:
                    summary[key] = nested[key]
    return summary


def _model_load_knobs(specs: list[TraceSpec]) -> dict[str, Any]:
    resolved: dict[str, Any] | None = None
    for spec in specs:
        knobs = spec["graph_knobs"]
        if not isinstance(knobs, Mapping):
            raise ValueError("trace spec graph_knobs must be an object")
        config = transcoder_config_to_json(
            resolve_transcoder_load_config(knobs, preserve_default_values=True)
        )
        if config["decoder_chunk_size"] <= 0:
            raise ValueError("decoder_chunk_size must be a positive int")
        if config["cross_batch_decoder_cache_bytes"] < 0:
            raise ValueError(
                "cross_batch_decoder_cache_bytes must be a non-negative int"
            )
        active_row_residency = knobs.get("decoder_active_row_residency", False)
        if not isinstance(active_row_residency, bool):
            raise ValueError("decoder_active_row_residency must be a bool")
        phase0_decoder_row_ranges = knobs.get("phase0_decoder_row_ranges", False)
        if not isinstance(phase0_decoder_row_ranges, bool):
            raise ValueError("phase0_decoder_row_ranges must be a bool")
        active_row_max_bytes = knobs.get("decoder_active_row_max_bytes", 0)
        if (
            isinstance(active_row_max_bytes, bool)
            or not isinstance(active_row_max_bytes, int)
            or active_row_max_bytes < 0
        ):
            raise ValueError("decoder_active_row_max_bytes must be a non-negative int")
        active_row_safety_margin_bytes = knobs.get(
            "decoder_active_row_safety_margin_bytes", 0
        )
        if (
            isinstance(active_row_safety_margin_bytes, bool)
            or not isinstance(active_row_safety_margin_bytes, int)
            or active_row_safety_margin_bytes < 0
        ):
            raise ValueError(
                "decoder_active_row_safety_margin_bytes must be a non-negative int"
            )
        active_row_requirement = knobs.get(
            "decoder_active_row_residency_requirement", "preferred"
        )
        if active_row_requirement not in {"preferred", "required"}:
            raise ValueError(
                "decoder_active_row_residency_requirement must be preferred or required"
            )
        if active_row_requirement == "required" and not active_row_residency:
            raise ValueError(
                "required decoder active-row residency requires "
                "decoder_active_row_residency=true"
            )
        if (
            active_row_residency
            and active_row_max_bytes == 0
            and active_row_safety_margin_bytes == 0
        ):
            raise ValueError(
                "dynamic decoder active-row residency requires a positive "
                "decoder_active_row_safety_margin_bytes"
            )
        if phase0_decoder_row_ranges:
            if config["transcoder_architecture"] != "plt":
                raise ValueError(
                    "phase0_decoder_row_ranges requires a PLT-compatible provider"
                )
            if not active_row_residency:
                raise ValueError(
                    "phase0_decoder_row_ranges requires "
                    "decoder_active_row_residency=true"
                )
            if knobs.get("reuse_phase0_window_state", False):
                raise ValueError(
                    "phase0_decoder_row_ranges is incompatible with "
                    "reuse_phase0_window_state until forward-session policy is shared"
                )
        if resolved is not None and any(
            config[k] != resolved[k] for k in PUBLIC_TRANSCODER_KNOB_KEYS
        ):
            raise ValueError(
                "all specs in a shard must share transcoder provider/load config"
            )
        resolved = config
    return resolved or transcoder_config_to_json(resolve_transcoder_load_config())


@dataclass(frozen=True)
class FullAnswerTypedGraphRequest:
    compact_result: Mapping[str, Any]
    step_idx: int
    token_text: str
    logprob: float | None
    retention_policy_id: str
    provider_identity: Mapping[str, Any]
    trace_identity: Mapping[str, Any]
    target_identity: Mapping[str, Any]
    step_identity: Mapping[str, Any]


def _build_typed_compact_graph(
    request: FullAnswerTypedGraphRequest,
) -> TypedCompactGraph:
    """Load the canonical artifact module only on the SLURM execution path."""
    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        ArtifactProvenance,
        build_typed_compact_graph,
    )

    bound_result = dict(request.compact_result)
    for key in ("semantic_fingerprint", "execution_fingerprint"):
        authoritative = request.trace_identity.get(key)
        if not isinstance(authoritative, str) or not authoritative:
            raise ValueError(
                f"Full Answer TraceResult.{key} is required for graph provenance"
            )
        if key in bound_result and bound_result[key] != authoritative:
            raise ValueError(
                f"compact output {key} conflicts with authoritative TraceResult"
            )
        bound_result[key] = authoritative

    return build_typed_compact_graph(
        bound_result,
        request.step_idx,
        provenance=ArtifactProvenance(
            provider=request.provider_identity,
            trace=request.trace_identity,
            target=request.target_identity,
            step=request.step_identity,
        ),
        token_text=request.token_text,
        logprob=request.logprob,
        retention_policy_id=request.retention_policy_id,
    )


def _save_typed_compact_graph(graph: TypedCompactGraph, path: Path) -> None:
    from nlp_research_project.exact_trace_bench.typed_compact_graph import (
        save_typed_compact_graph,
    )

    save_typed_compact_graph(graph, path)


def _trace_request(
    *,
    model: Any,
    prompt_token_ids: list[int],
    spec: TraceSpec,
    prefix_metadata: Mapping[str, Any],
    full_sequence_mode: bool,
    telemetry_jsonl_path: Path | None = None,
) -> Any:
    """Translate one harness spec into canonical, subsystem-owned policies."""
    import torch
    from circuit_tracer import (
        AttributionProblem,
        DecoderCachePolicy,
        DiagnosticStopPolicy,
        ExecutionConstraints,
        FrontierExpansionPlan,
        FrontierSemantics,
        ObservabilityPolicy,
        PrefixViewTarget,
        ReplayPlan,
        RowStoragePlan,
        SessionPlan,
        TraceEvidence,
        TraceRequest,
        TraceSemantics,
    )

    knobs = spec["graph_knobs"]
    correctness_capture = knobs.get("correctness_probe_mode", "off") != "off"
    semantic_descriptor_top_k = int(knobs.get("semantic_descriptor_top_k", 2048))
    if correctness_capture:
        from ..correctness.behavioral import MAX_UNSELECTED_CONTROLS

        semantic_descriptor_top_k = max(
            semantic_descriptor_top_k,
            int(knobs.get("max_feature_nodes", 8192)) + MAX_UNSELECTED_CONTROLS,
        )
    semantics = TraceSemantics(
        source_batch_size=int(knobs.get("attribution_batch_size", 256)),
        feature_batch_size=knobs.get("feature_batch_size"),
        logit_batch_size=knobs.get("logit_batch_size"),
        max_feature_nodes=int(knobs.get("max_feature_nodes", 8192)),
        diagnostic_feature_cap=knobs.get("diagnostic_feature_cap"),
        update_interval=int(knobs.get("attribution_update_interval", 4)),
        exact_trace_internal_dtype=str(knobs.get("exact_trace_internal_dtype", "fp32")),
        phase0_activation_threshold_compare_mode=str(
            knobs.get("phase0_activation_threshold_compare_mode", "baseline")
        ),
        frontier=FrontierSemantics(
            scheduler=str(knobs.get("phase4_scheduler_mode", "locality")),
            refresh_policy=str(knobs.get("phase4_refresh_policy", "standard")),
            refresh_interval_multiplier=int(
                knobs.get("phase4_refresh_interval_multiplier", 1)
            ),
            ranker=str(knobs.get("phase4_ranker", "argsort")),
            phase3_buffer_relative_epsilon=knobs.get(
                "phase3_frontier_buffer_relative_epsilon"
            ),
            phase3_buffer_max_extra=int(
                knobs.get("phase3_frontier_buffer_max_extra", 0)
            ),
            phase4_buffer_relative_epsilon=knobs.get(
                "phase4_frontier_buffer_relative_epsilon"
            ),
            phase4_buffer_max_extra_per_refresh=int(
                knobs.get("phase4_frontier_buffer_max_extra_per_refresh", 0)
            ),
            phase4_buffer_max_extra_total=int(
                knobs.get("phase4_frontier_buffer_max_extra_total", 0)
            ),
        ),
    )
    decoder_cache_bytes = int(knobs.get("cross_batch_decoder_cache_bytes", 0))
    execution = ExecutionConstraints(
        session=SessionPlan(
            capacity=knobs.get("nnsight_session_capacity"),
            phase3_microbatch_max_rows=knobs.get("phase3_compute_microbatch_max_rows"),
            phase4_execution_batch_max_rows=(
                knobs["phase4_execution_batch_max_rows"]
                if knobs.get("phase4_execution_batch_max_rows") is not None
                else knobs.get("phase4_compute_microbatch_max_rows")
            ),
            phase1_trace_batch_policy=str(
                knobs.get("phase1_trace_batch_policy", "legacy")
            ),
            phase1_trace_batch_size_max=knobs.get("phase1_trace_batch_size_max"),
            decoder_cache=DecoderCachePolicy(
                enabled=decoder_cache_bytes > 0,
                max_bytes=decoder_cache_bytes or None,
            ),
        ),
        backward=build_backward_plan(knobs),
        storage=RowStoragePlan(
            retention=str(knobs.get("feature_row_retention", "full_file")),
            full_retention_backend=str(
                knobs.get("full_retention_backend", "full_file")
            ),
            feature_column_tile_size=int(
                knobs.get("feature_row_column_tile_size", 2048)
            ),
            influence_row_tile_size=int(knobs.get("influence_row_tile_size", 4096)),
            influence_column_tile_size=int(
                knobs.get("influence_column_tile_size", 2048)
            ),
            cache_control=str(knobs.get("row_store_cache_control", "off")),
            temp_root_policy=str(knobs.get("row_store_temp_root_policy", "default")),
            temp_root=knobs.get("row_store_temp_root"),
            preallocate=bool(knobs.get("row_store_preallocate", True)),
            replay_tile_cache_bytes=knobs.get("replay_tile_cache_bytes"),
            feature_row_influence_mode=str(
                knobs.get("feature_row_influence_mode", "cpu_exact")
            ),
            feature_row_influence_requirement=cast(
                Literal["preferred", "required"],
                str(knobs.get("feature_row_influence_requirement", "preferred")),
            ),
            gpu_resident_max_bytes=int(
                knobs.get("feature_row_gpu_resident_max_bytes", 0)
            ),
            gpu_window_max_bytes=int(knobs.get("feature_row_gpu_window_max_bytes", 0)),
            gpu_resident_safety_margin_bytes=int(
                knobs.get("feature_row_gpu_resident_safety_margin_bytes", 0)
            ),
            exact_encoder_residency=str(knobs.get("exact_encoder_residency", "lazy")),
        ),
        replay=ReplayPlan(
            feature_window=int(knobs.get("chunked_feature_replay_window") or 4),
            error_vector_prefetch_lookahead=int(
                knobs.get("error_vector_prefetch_lookahead") or 2
            ),
            stage_encoder_vecs_on_cpu=knobs.get("stage_encoder_vecs_on_cpu"),
            stage_error_vectors_on_cpu=knobs.get("stage_error_vectors_on_cpu"),
            decoder_contraction_tile=knobs.get("row_subchunk_size"),
            phase0_donor_bundle=knobs.get("phase0_donor_bundle"),
            phase0_mode=str(knobs.get("phase0_replay_mode", "disabled")),
            phase0_donor_context_policy=str(
                knobs.get("phase0_donor_context_policy", "strict")
            ),
            phase3_gradient_donor_bundle=knobs.get("phase3_gradient_donor_bundle"),
            phase3_gradient_mode=str(
                knobs.get("phase3_gradient_replay_mode", "disabled")
            ),
            phase3_row_donor_bundle=knobs.get("phase3_row_donor_bundle"),
            phase3_row_mode=str(knobs.get("phase3_row_replay_mode", "disabled")),
            phase3_validation_policy=str(
                knobs.get("phase3_replay_validation_policy", "strict")
            ),
        ),
        frontier=FrontierExpansionPlan(
            scheduler_debug=bool(knobs.get("phase4_scheduler_debug", False)),
            scheduler_telemetry_detail=str(
                knobs.get("phase4_scheduler_telemetry_detail", "normal")
            ),
            refresh_optimization=str(knobs.get("phase4_refresh_optimization", "v1")),
            refresh_prepared_chunk_cache_bytes=int(
                knobs.get("phase4_refresh_prepared_chunk_cache_bytes", 0)
            ),
            refresh_active_row_accumulation=str(
                knobs.get("phase4_refresh_active_row_accumulation", "direct_v1")
            ),
            row_executor=str(knobs.get("phase4_row_executor", "batched")),
            row_reduction=str(knobs.get("phase4_row_reduction", "gpu_v1")),
            feature_batch_planning=bool(knobs.get("plan_feature_batch_size", False)),
            feature_batch_size_max=knobs.get("feature_batch_size_max"),
            feature_batch_target_reserved_fraction=float(
                knobs.get("feature_batch_target_reserved_fraction", 0.9)
            ),
            feature_batch_min_free_fraction=float(
                knobs.get("feature_batch_min_free_fraction", 0.05)
            ),
            feature_batch_probe_batches=int(
                knobs.get("feature_batch_probe_batches", 1)
            ),
            feature_vjp_tape_batch_window=int(
                knobs.get("feature_vjp_tape_batch_window", 1)
            ),
            feature_vjp_tape_max_bytes=int(knobs.get("feature_vjp_tape_max_bytes", 0)),
            decoder_page_prefetch_depth=int(
                knobs.get("decoder_page_prefetch_depth", 0)
            ),
            decoder_active_row_residency=bool(
                knobs.get("decoder_active_row_residency", False)
            ),
            decoder_active_row_residency_requirement=str(
                knobs.get("decoder_active_row_residency_requirement", "preferred")
            ),
            decoder_active_row_max_bytes=int(
                knobs.get("decoder_active_row_max_bytes", 0)
            ),
            decoder_active_row_safety_margin_bytes=int(
                knobs.get("decoder_active_row_safety_margin_bytes", 0)
            ),
            phase0_decoder_row_ranges=bool(
                knobs.get("phase0_decoder_row_ranges", False)
            ),
        ),
        observability=ObservabilityPolicy(
            verbose=bool(knobs.get("verbose_attribution", True)),
            profile=bool(knobs.get("profile_attribution", True)),
            profile_log_interval=int(knobs.get("profile_log_interval", 1)),
            telemetry_max_events=knobs.get("telemetry_max_events"),
            telemetry_jsonl_path=telemetry_jsonl_path,
            telemetry_context={
                key: (str(value) if key in {"trace_id", "trajectory_id"} else int(value))
                for key in (
                    "trace_id",
                    "trajectory_id",
                    "generated_index",
                    "target_position",
                    "target_token_id",
                )
                if (value := spec.get(key)) is not None
            },
            phase4_anomaly_debug=bool(knobs.get("phase4_anomaly_debug", False)),
            cross_cluster_debug=bool(knobs.get("cross_cluster_debug", False)),
            capture_phase0_donor_bundle=bool(
                knobs.get("capture_phase0_donor_bundle", False)
            ),
            capture_phase3_seed_bundle=bool(
                knobs.get("capture_phase3_seed_bundle", False)
            ),
            capture_phase3_gradient_bundle=bool(
                knobs.get("capture_phase3_gradient_bundle", False)
            ),
            capture_phase3_row_bundle=bool(
                knobs.get("capture_phase3_row_bundle", False)
            ),
            capture_feature_semantic_descriptors=bool(
                knobs.get("capture_feature_semantic_descriptors", False)
                or correctness_capture
            ),
            semantic_descriptor_top_k=semantic_descriptor_top_k,
            semantic_descriptor_dim=int(knobs.get("semantic_descriptor_dim", 64)),
        ),
        diagnostic_stop=DiagnosticStopPolicy(
            mode=str(knobs.get("diagnostic_stop_mode", "none")),
            phase4_batches=knobs.get("diagnostic_stop_phase4_batches"),
        ),
        offload="cpu",
        compact_output=not bool(knobs.get("save_raw_graph", False)),
    )
    prefix_evidence = {
        key: value
        for key, value in prefix_metadata.items()
        if key not in {"mode", "target_position"}
    }
    return TraceRequest(
        problem=AttributionProblem(
            model=model,
            prompt=torch.tensor(prompt_token_ids, dtype=torch.long),
            targets=torch.tensor([spec["target_token_id"]], dtype=torch.long),
            max_n_logits=1,
            desired_logit_prob=1.0,
            output_position=spec["target_position"] - 1 if full_sequence_mode else None,
            prefix_view=PrefixViewTarget(
                mode=(
                    "full_sequence_target_position"
                    if full_sequence_mode
                    else "independent_prefix"
                ),
                target_position=int(spec["target_position"]),
            ),
        ),
        semantics=semantics,
        execution=execution,
        evidence=TraceEvidence(
            name="full_answer_runner",
            version="1",
            metadata={"prefix_view_metadata": prefix_evidence},
        ),
    )


def _json_ready(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _save_compact_debug_sidecars(
    token_dir: Path, compact_result: Mapping[str, Any]
) -> dict[str, str]:
    sidecar_keys = (
        "cross_cluster_debug_summary",
        "cross_cluster_debug_checkpoints",
        "cross_cluster_debug_batches",
    )
    sidecars: dict[str, str] = {}
    for key in sidecar_keys:
        payload = compact_result.get(key)
        if payload is None:
            continue
        ensure_dir(token_dir)
        path = token_dir / f"{key}.json"
        write_json(path, _json_ready(payload))
        sidecars[key] = str(path)
    return sidecars


def _persist_compact_telemetry_events(
    *,
    token_dir: Path,
    compact_result: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    events = compact_result.get("telemetry_events")
    if not events:
        return {"telemetry_event_count": 0, "telemetry_events_path": None}
    if not isinstance(events, list):
        events = [events]

    path = token_dir / "telemetry.jsonl"
    context = {
        "run_id": trace.get("run_id"),
        "shard_id": trace.get("shard_id"),
        "trace_id": trace.get("trace_id"),
        "trajectory_id": trace.get("trajectory_id"),
        "generated_index": trace.get("generated_index"),
        "target_position": trace.get("target_position"),
        "target_token_id": trace.get("target_token_id"),
        "target_token_text": trace.get("target_token_text"),
    }
    rows = []
    for index, event in enumerate(events):
        rows.append(
            {
                **context,
                "telemetry_event_index": index,
                "event": _json_ready(event),
            }
        )
    write_jsonl(path, rows)
    return {"telemetry_event_count": len(rows), "telemetry_events_path": str(path)}


_CAPTURE_SIDECAR_KNOBS = {
    "phase0_donor_bundle": "capture_phase0_donor_bundle",
    "phase3_seed_bundle": "capture_phase3_seed_bundle",
    "phase3_gradient_bundle": "capture_phase3_gradient_bundle",
    "phase3_row_bundle": "capture_phase3_row_bundle",
    "feature_semantic_descriptors": "capture_feature_semantic_descriptors",
}


def _persist_successful_result_artifacts(
    *,
    token_dir: Path,
    trace_result: Any,
    selected_config: Mapping[str, Any],
    trace: dict[str, Any],
) -> Mapping[str, Any] | None:
    """Persist result evidence before branching on full versus probe completion.

    A successful trace result has one artifact contract regardless of whether it
    reached graph assembly.  In particular, a transition probe returns its
    bounded captures through ``TraceResult.output`` even though it has no graph.
    """
    output = getattr(trace_result, "output", None)
    compact_output = output if isinstance(output, Mapping) else None

    telemetry_events = getattr(trace_result, "telemetry_events", None)
    if not telemetry_events and compact_output is not None:
        telemetry_events = compact_output.get("telemetry_events")
    if not telemetry_events:
        normalized_events: list[Any] = []
    elif isinstance(telemetry_events, Mapping):
        normalized_events = [telemetry_events]
    else:
        normalized_events = list(telemetry_events)
    telemetry_payload = {"telemetry_events": normalized_events}
    trace.update(
        _persist_compact_telemetry_events(
            token_dir=token_dir,
            compact_result=telemetry_payload,
            trace=trace,
        )
    )

    debug_sidecars = (
        _save_compact_debug_sidecars(token_dir, compact_output)
        if compact_output is not None
        else {}
    )
    capture_artifacts = write_capture_artifacts(
        requested={
            sidecar_key: bool(selected_config.get(knob_key, False))
            or (
                sidecar_key == "feature_semantic_descriptors"
                and selected_config.get("correctness_probe_mode", "off") != "off"
            )
            for sidecar_key, knob_key in _CAPTURE_SIDECAR_KNOBS.items()
        },
        payloads=compact_output or {},
        path_for=lambda name: token_dir / f"{name}.npz",
    )
    trace["capture_artifact_status"] = capture_artifacts
    trace["sidecar_status"] = legacy_capture_artifact_status(capture_artifacts)
    debug_sidecars.update(capture_artifacts["paths"])
    if debug_sidecars:
        trace["debug_sidecars"] = debug_sidecars
    if compact_output is not None and "decoder_active_row_residency" in compact_output:
        trace["decoder_active_row_residency"] = _json_ready(
            compact_output["decoder_active_row_residency"]
        )
    require_complete_capture_artifacts(capture_artifacts)
    return compact_output


def _execution_result_payload(
    trace_result: Any,
    *,
    selected_config: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = getattr(trace_result, "effective_execution", None)
    if descriptor is None:
        effective = None
    elif isinstance(descriptor, Mapping):
        effective = dict(descriptor)
    else:
        effective = descriptor.to_dict()
    return {
        "requested_execution_fingerprint": getattr(
            trace_result, "requested_execution_fingerprint", None
        ),
        "effective_execution_fingerprint": getattr(
            trace_result, "effective_execution_fingerprint", None
        ),
        "execution_fingerprint": getattr(trace_result, "execution_fingerprint", None),
        "effective_execution": effective,
        "execution_resolution": execution_resolution(
            selected_config=selected_config,
            effective_execution=effective,
        ),
    }


_TELEMETRY_EXCEPTION_SUMMARY_ATTR = "circuit_tracer_telemetry_summary"
_TELEMETRY_EXCEPTION_EVENTS_ATTR = "circuit_tracer_telemetry_events"


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.extend(
            [
                getattr(current, "original", None),
                current.__cause__,
                current.__context__,
            ]
        )
    return chain


def _exception_telemetry_payload(exc: BaseException) -> dict[str, Any] | None:
    for candidate in _iter_exception_chain(exc):
        summary = getattr(candidate, _TELEMETRY_EXCEPTION_SUMMARY_ATTR, None)
        events = getattr(candidate, _TELEMETRY_EXCEPTION_EVENTS_ATTR, None)
        if summary is None and events is None:
            continue
        if events is None:
            events = []
        elif not isinstance(events, list):
            events = [events]
        return {
            "telemetry_summary": _json_ready(summary),
            "telemetry_events": events,
            "telemetry_exception_type": type(candidate).__name__,
        }
    return None


def _apply_correctness_requirement(
    trace: dict[str, Any],
    *,
    mode: str,
) -> None:
    correctness = trace.get("correctness")
    required_policy_satisfied = (
        correctness.get("required_policy_satisfied")
        if isinstance(correctness, Mapping)
        else False
    )
    if mode != "required" or required_policy_satisfied is True:
        return
    error = RuntimeError("required correctness policy did not fully qualify the trace")
    trace.update(
        {
            "status": "error",
            "error": repr(error),
            "error_type": type(error).__name__,
        }
    )


def run_real_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
    run_id: str | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
) -> dict[str, Any]:
    """SLURM-only real tracing path; imports model stack only inside this function."""
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    metadata = _runtime_metadata(
        run_id=run_id,
        run_name=run_name,
        run_description=run_description,
        run_goal=run_goal,
    )
    if (
        os.environ.get("SLURM_JOB_ID") is None
        and os.environ.get("EXACT_TRACE_ALLOW_LOCAL_GPU") != "1"
    ):
        raise RuntimeError(
            "run-full-answer-shard requires SLURM_JOB_ID; set EXACT_TRACE_ALLOW_LOCAL_GPU=1 "
            "to override for local GPU debugging before importing model code"
        )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    trace_results_path = root / "trace_results.jsonl"
    trace_results_path.write_text("", encoding="utf-8")
    selected_execution = selected_execution_record(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        specs=specs,
        shard=shard,
        metadata=metadata,
    )
    write_json(root / "selected_execution.json", selected_execution)
    resource_payload = cast(dict[str, Any], selected_execution["resources"])
    print(
        json.dumps(
            {
                "event": "selected_execution",
                "selection_fingerprint": selected_execution["selection_fingerprint"],
                "feature_row_influence": selected_execution["mechanism_selection"],
                "backward_engine_mode": (
                    selected_execution["mechanism_selection"].get(
                        "backward_engine_mode"
                    )
                    if isinstance(selected_execution["mechanism_selection"], Mapping)
                    else None
                ),
                "resources": resource_payload,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resource_policy = RuntimeResourcePolicy(resource_payload["runtime_policy"])
    planning_envelope = cast(dict[str, Any], resource_payload["planning_envelope"])
    resource_started = time.monotonic()
    resource_samples_path = root / "resource_samples.jsonl"
    if resource_policy is not RuntimeResourcePolicy.OFF:
        initial_sample = resource_sample(
            boundary="shard_start", started_monotonic=resource_started
        )
        validate_resource_policy(
            policy=resource_policy,
            envelope=planning_envelope,
            sample=initial_sample,
        )
        resource_samples_path.write_text(
            json.dumps(initial_sample, sort_keys=True) + "\n", encoding="utf-8"
        )
    write_json(
        root / "shard.json",
        _shard_record(
            status="running",
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata=metadata,
            target_positions=[spec["target_position"] for spec in specs],
        ),
    )

    from circuit_tracer import SessionWindow, open_session, trace_one

    from nlp_research_project.exact_trace_bench.trace_runtime.provider import (
        get_model_transcoder_metadata,
        load_model,
    )

    model_load_knobs = _model_load_knobs(specs)
    model = load_model(
        exact_chunked_decoder=True,
        **model_load_knobs,
    )
    transcoder_metadata = get_model_transcoder_metadata(model) or {
        "requested": model_load_knobs
    }
    metadata = {**metadata, "transcoder": transcoder_metadata}
    if resource_policy is not RuntimeResourcePolicy.OFF:
        loaded_sample = resource_sample(
            boundary="model_loaded", started_monotonic=resource_started
        )
        validate_resource_policy(
            policy=resource_policy,
            envelope=planning_envelope,
            sample=loaded_sample,
        )
        with resource_samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(loaded_sample, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    full_sequence_cache = prepare_full_sequence_cache(trajectory, specs)
    for window in _shard_windows(specs, shard):
        window_specs = cast(list[TraceSpec], window["specs"])
        _validate_window_session_specs(window_specs)
        window_session = None
        pending_records: list[
            tuple[Path, dict[str, Any], dict[str, Any] | None]
        ] = []
        if any(
            spec.get("graph_knobs", {}).get("trajectory_session_mode")
            == "window_reuse_v1"
            for spec in window_specs
        ):
            if full_sequence_cache is None:
                raise ValueError("window_reuse_v1 requires full_sequence cache")
            window_session_knobs = next(
                spec["graph_knobs"]
                for spec in window_specs
                if spec.get("graph_knobs", {}).get("trajectory_session_mode")
                == "window_reuse_v1"
            )
            if bool(
                window_session_knobs.get("reuse_phase0_window_state", False)
            ) != bool(window_session_knobs.get("reuse_target_logits", False)):
                raise ValueError(
                    "canonical window sessions require phase-0 and target-logit reuse "
                    "to be enabled or disabled together"
                )
        try:
            for spec in window_specs:
                token_dir = root / f"token_{spec['generated_index']:06d}"
                ensure_dir(token_dir)
                graph_path = token_dir / "graph.npz"
                trace = _trace_runtime_payload(
                    spec=spec, shard_id=shard_id, metadata=metadata
                )
                trace["status"] = "running"
                trace["forced_target"] = forced_target_payload(spec)
                trace.update(_target_token_metadata(trajectory, spec))
                trace["window_session"] = {
                    key: value for key, value in window.items() if key != "specs"
                }
                correctness_context: dict[str, Any] | None = None
                started = time.perf_counter()
                try:
                    prefix_token_ids = reconstruct_prefix_token_ids(trajectory, spec)
                    prefix_metadata = prefix_view_metadata(
                        trajectory,
                        spec,
                        prefix_token_ids,
                        full_sequence_token_ids=full_sequence_cache["token_ids"]
                        if full_sequence_cache
                        else None,
                        full_sequence_token_ids_sha256=full_sequence_cache[
                            "token_ids_sha256"
                        ]
                        if full_sequence_cache
                        else None,
                    )
                    trace["prefix_view_metadata"] = prefix_metadata
                    mode = spec.get("graph_knobs", {}).get(
                        "trajectory_session_mode", "per_token"
                    )
                    trace["trajectory_session"] = {
                        **session_reuse_metadata(spec),
                        "session_reuse_fallback": None
                        if mode == "window_reuse_v1"
                        else session_reuse_metadata(spec)["session_reuse_fallback"],
                        "decoder_cache_reuse_effective": False,
                    }
                    knobs = spec["graph_knobs"]
                    save_raw_graph = bool(knobs.get("save_raw_graph", False))
                    raw_graph_path = token_dir / "graph.pt"
                    trace["graph_path"] = str(
                        raw_graph_path if save_raw_graph else graph_path
                    )
                    full_sequence_mode = (
                        knobs.get("input_context_mode") == "full_sequence"
                    )
                    if full_sequence_mode and full_sequence_cache is None:
                        raise ValueError(
                            "full_sequence cache missing for full_sequence trace spec"
                        )
                    if full_sequence_mode:
                        assert full_sequence_cache is not None
                        prompt_token_ids = full_sequence_cache["token_ids"]
                    else:
                        prompt_token_ids = prefix_token_ids
                    request = _trace_request(
                        model=model,
                        prompt_token_ids=prompt_token_ids,
                        spec=spec,
                        prefix_metadata=prefix_metadata,
                        full_sequence_mode=full_sequence_mode,
                        telemetry_jsonl_path=(
                            token_dir / "telemetry_live.jsonl"
                            if (
                                knobs.get("incremental_telemetry_jsonl", False)
                                or knobs.get("correctness_probe_mode", "off") != "off"
                            )
                            else None
                        ),
                    )
                    if mode == "window_reuse_v1":
                        if window_session is None:
                            window_session = open_session(
                                request,
                                window=SessionWindow(
                                    max_prefix_len=int(
                                        window["window_max_target_position"]
                                    )
                                ),
                            )
                        trace_result = window_session.trace_window(
                            int(spec["target_position"]),
                            reuse=bool(knobs.get("reuse_phase0_window_state", False)),
                            request=request,
                        )
                    elif request.execution.session.decoder_cache.enabled:
                        if window_session is None:
                            window_session = open_session(request)
                        trace_result = window_session.trace(request)
                    else:
                        trace_result = trace_one(request)
                    trace.update(
                        _execution_result_payload(
                            trace_result,
                            selected_config=spec["graph_knobs"],
                        )
                    )
                    compact_result = _persist_successful_result_artifacts(
                        token_dir=token_dir,
                        trace_result=trace_result,
                        selected_config=spec["graph_knobs"],
                        trace=trace,
                    )
                    trace["trajectory_session"]["decoder_cache_reuse_effective"] = bool(
                        request.execution.session.decoder_cache.enabled
                        and window_session is not None
                    )
                    trace_status = (
                        getattr(trace_result.status, "value", trace_result.status)
                        if hasattr(trace_result, "status")
                        else "succeeded"
                    )
                    if trace_status == "probe_completed":
                        trace.update(
                            {
                                "status": "probe_completed",
                                "error": None,
                                "graph_path": None,
                                "graph_summary": None,
                                "diagnostic_stop_mode": (
                                    trace_result.telemetry_summary.get(
                                        "diagnostic_stop_mode"
                                    )
                                ),
                                "phase4_batches_completed": (
                                    trace_result.telemetry_summary.get(
                                        "phase4_batches_completed", 0
                                    )
                                ),
                                "semantic_fingerprint": (
                                    trace_result.semantic_fingerprint
                                ),
                                "telemetry_summary": _json_ready(
                                    trace_result.telemetry_summary
                                ),
                                "timings": {
                                    "trace_seconds": (time.perf_counter() - started),
                                },
                            }
                        )
                        raise _DiagnosticProbeCompleted
                    graph_result = trace_result.output
                    if trace_result.telemetry_summary:
                        trace["telemetry_summary"] = _json_ready(
                            trace_result.telemetry_summary
                        )
                    if save_raw_graph:
                        raw_graph_path.parent.mkdir(parents=True, exist_ok=True)
                        graph_result.to_pt(str(raw_graph_path))
                        graph_summary: dict[str, Any] = {
                            "saved": True,
                            "graph_path": str(raw_graph_path),
                            "format": "raw_graph_pt",
                            "file_bytes": raw_graph_path.stat().st_size,
                        }
                        adjacency = getattr(graph_result, "adjacency_matrix", None)
                        if adjacency is not None:
                            graph_summary["adjacency_shape"] = list(adjacency.shape)
                            graph_summary["adjacency_dtype"] = str(adjacency.dtype)
                        active_features = getattr(graph_result, "active_features", None)
                        if active_features is not None:
                            graph_summary["active_feature_count"] = int(
                                active_features.shape[0]
                            )
                        selected_features = getattr(
                            graph_result, "selected_features", None
                        )
                        if selected_features is not None:
                            graph_summary["selected_feature_count"] = int(
                                selected_features.numel()
                            )
                    else:
                        if compact_result is None:
                            raise TypeError(
                                "compact trace output must be a mapping, got "
                                f"{type(graph_result).__name__}"
                            )
                        graph = _build_typed_compact_graph(
                            FullAnswerTypedGraphRequest(
                                compact_result=compact_result,
                                step_idx=spec["generated_index"],
                                token_text=spec["target_token_text"],
                                logprob=_target_token_metadata(trajectory, spec).get(
                                    "target_logprob"
                                ),
                                retention_policy_id=str(
                                    knobs["edge_retention_policy_id"]
                                ),
                                provider_identity={
                                    key: knobs.get(key)
                                    for key in PUBLIC_TRANSCODER_KNOB_KEYS
                                },
                                trace_identity={
                                    "trace_id": spec["trace_id"],
                                    "semantic_fingerprint": getattr(
                                        trace_result, "semantic_fingerprint", None
                                    ),
                                    "execution_fingerprint": getattr(
                                        trace_result, "execution_fingerprint", None
                                    ),
                                },
                                target_identity={
                                    "target_position": spec["target_position"],
                                    "target_token_id": spec["target_token_id"],
                                    "target_token_text": spec["target_token_text"],
                                },
                                step_identity={
                                    "trajectory_id": spec["trajectory_id"],
                                    "generated_index": spec["generated_index"],
                                    "prefix_token_count": spec["prefix_token_count"],
                                },
                            )
                        )
                        _save_typed_compact_graph(graph, graph_path)
                        if knobs.get("correctness_probe_mode", "off") != "off":
                            from ..correctness.contracts import ExpectedGraphIdentity

                            correctness_context = {
                                "graph_path": graph_path,
                                "compact_result": compact_result,
                                "graph_identity": ExpectedGraphIdentity(
                                    step_idx=graph.step_idx,
                                    graph_fingerprint=graph.graph_fingerprint,
                                    target_fingerprint=graph.target_fingerprint,
                                    provider_fingerprint=graph.provider_fingerprint,
                                    trace_fingerprint=graph.trace_fingerprint,
                                    step_fingerprint=graph.step_fingerprint,
                                ),
                                "trace_result": trace_result,
                                "spec": spec,
                            }
                        graph_summary = _graph_summary(graph, graph_path)
                        graph_summary.update(
                            {
                                "format": graph.compact_save_format,
                                "edge_count": graph.edge_count,
                                "feature_node_count": graph.n_features,
                                "edge_retention_policy_id": (graph.retention_policy_id),
                                "edge_retention_policy_fingerprint": (
                                    graph.retention_policy_fingerprint
                                ),
                                "graph_fingerprint": graph.graph_fingerprint,
                            }
                        )
                    trace.update(
                        {
                            "status": "ok",
                            "error": None,
                            "timings": {
                                "trace_seconds": time.perf_counter() - started,
                            },
                            "graph_summary": graph_summary,
                        }
                    )
                    if not save_raw_graph:
                        assert compact_result is not None
                        for key in (
                            "phase0_window_state_reuse_requested",
                            "phase0_window_state_reuse_effective",
                            "target_logit_source",
                        ):
                            if key in compact_result:
                                trace["trajectory_session"][key] = _json_ready(
                                    compact_result.get(key)
                                )
                        trace["trajectory_session"][
                            "reuse_phase0_window_state_effective"
                        ] = bool(
                            compact_result.get(
                                "phase0_window_state_reuse_effective", False
                            )
                        )
                        trace["trajectory_session"]["reuse_target_logits_effective"] = (
                            compact_result.get("target_logit_source")
                            == "full_sequence_window_logits"
                        )
                        trace["trajectory_session"]["target_logit_source"] = (
                            _normalize_target_logit_source(
                                compact_result.get("target_logit_source")
                            )
                        )
                        trace["trajectory_session"]["session_reuse_effective"] = bool(
                            trace["trajectory_session"].get(
                                "decoder_cache_reuse_effective", False
                            )
                            or trace["trajectory_session"].get(
                                "reuse_phase0_window_state_effective", False
                            )
                            or trace["trajectory_session"].get(
                                "reuse_target_logits_effective", False
                            )
                        )
                    if (
                        not save_raw_graph
                    ) and "phase3_frontier_buffer_metadata" in compact_result:
                        trace["phase3_frontier_buffer_metadata"] = _json_ready(
                            compact_result.get("phase3_frontier_buffer_metadata")
                        )
                    if (
                        not save_raw_graph
                    ) and "phase4_frontier_buffer_metadata" in compact_result:
                        trace["phase4_frontier_buffer_metadata"] = _json_ready(
                            compact_result.get("phase4_frontier_buffer_metadata")
                        )
                except _DiagnosticProbeCompleted:
                    pass
                except (
                    Exception
                ) as exc:  # pragma: no cover - exercised only in SLURM real mode
                    exception_payload = _exception_payload(exc)
                    trace.update(exception_payload)
                    exception_telemetry = _exception_telemetry_payload(exc)
                    if exception_telemetry is not None:
                        trace["telemetry_summary"] = exception_telemetry[
                            "telemetry_summary"
                        ]
                        trace["telemetry_exception_type"] = exception_telemetry[
                            "telemetry_exception_type"
                        ]
                        trace.update(
                            _persist_compact_telemetry_events(
                                token_dir=token_dir,
                                compact_result=exception_telemetry,
                                trace=trace,
                            )
                        )
                    trace.update(
                        {
                            "status": "error",
                            "error_traceback": exception_payload["traceback"],
                            "timings": {
                                "trace_seconds": time.perf_counter() - started,
                            },
                        }
                    )
                pending_records.append((token_dir, trace, correctness_context))
        finally:
            if window_session is not None:
                window_session.close()
        envelope_cleanup = None
        if window_session is not None:
            from ..correctness.contracts import EnvelopeCleanupEvidence

            envelope_cleanup = EnvelopeCleanupEvidence(
                complete=True,
                envelope_id=(
                    f"shard-{shard_id}:"
                    f"{window.get('window_start_generated_index')}:"
                    f"{window.get('window_end_generated_index')}"
                ),
                detail="Reusable tracing session closed before correctness probes.",
            )
        for token_dir, trace, correctness_context in pending_records:
            correctness_started = time.perf_counter()
            correctness_mode = str(
                trace.get("graph_knobs", {}).get(
                    "correctness_probe_mode",
                    "off",
                )
            )
            correctness_policy_id = str(
                trace.get("graph_knobs", {}).get(
                    "correctness_policy_id",
                    "behavioral_closure_v1",
                )
            )
            if correctness_mode == "off":
                trace["correctness"] = {
                    "mode": "off",
                    "policy_id": correctness_policy_id,
                    "status": "not_applicable",
                    "required_policy_satisfied": False,
                    "axes": {
                        "structural": "not_applicable",
                        "numerical": "not_applicable",
                        "behavioral": "not_applicable",
                    },
                    "artifacts": {},
                    "errors": [],
                }
            elif correctness_context is None or trace.get("status") != "ok":
                trace["correctness"] = {
                    "mode": correctness_mode,
                    "policy_id": correctness_policy_id,
                    "status": "unavailable",
                    "required_policy_satisfied": False,
                    "axes": {
                        "structural": "invalid",
                        "numerical": ["unknown"],
                        "behavioral": "unknown",
                    },
                    "artifacts": {},
                    "errors": [
                        {
                            "stage": "trace",
                            "error_type": "TraceArtifactUnavailable",
                            "detail": (
                                "Correctness evidence requires a successful typed "
                                "compact graph trace."
                            ),
                        }
                    ],
                }
            else:
                try:
                    from .correctness_gate import run_full_answer_correctness_gate

                    outcome = run_full_answer_correctness_gate(
                        token_dir=token_dir,
                        model=model,
                        transcoder_metadata=transcoder_metadata,
                        envelope_cleanup=envelope_cleanup,
                        **correctness_context,
                    )
                    trace["correctness"] = outcome.trace_payload
                except Exception as error:  # correctness evidence is mode-governed
                    trace["correctness"] = {
                        "mode": correctness_mode,
                        "policy_id": correctness_policy_id,
                        "status": "unavailable",
                        "required_policy_satisfied": False,
                        "axes": {
                            "structural": "invalid",
                            "numerical": ["unknown"],
                            "behavioral": "unknown",
                        },
                        "artifacts": {},
                        "errors": [
                            {
                                "stage": "correctness_gate",
                                "error_type": type(error).__name__,
                                "detail": str(error),
                            }
                        ],
                    }
            _apply_correctness_requirement(trace, mode=correctness_mode)
            correctness_seconds = time.perf_counter() - correctness_started
            trace_timings = trace.setdefault("timings", {})
            trace_timings["correctness_probe_seconds"] = correctness_seconds
            trace_timings["total_token_seconds"] = (
                float(trace_timings.get("trace_seconds", 0.0))
                + correctness_seconds
            )
            if resource_policy is not RuntimeResourcePolicy.OFF:
                token_sample = resource_sample(
                    boundary=(
                        f"token_{int(trace['generated_index']):06d}_terminal"
                    ),
                    started_monotonic=resource_started,
                )
                try:
                    validate_resource_policy(
                        policy=resource_policy,
                        envelope=planning_envelope,
                        sample=token_sample,
                    )
                except RuntimeError as resource_error:
                    trace.update(
                        {
                            "status": "error",
                            "resource_policy_error": str(resource_error),
                            "error": repr(resource_error),
                            "error_type": type(resource_error).__name__,
                        }
                    )
                trace["resource_sample"] = token_sample
                with resource_samples_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(token_sample, sort_keys=True) + "\n")
            write_json(token_dir / "trace.json", trace)
            rows.append(trace)
            with trace_results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, sort_keys=True) + "\n")
    row_statuses = {str(row.get("status")) for row in rows}
    if row_statuses == {"probe_completed"}:
        final_status = "probe_completed"
    elif row_statuses == {"ok"}:
        final_status = "complete"
    elif row_statuses.issubset({"ok", "probe_completed"}):
        final_status = "complete_with_diagnostics"
    else:
        final_status = "error"
    token_seconds = [
        float(r.get("timings", {}).get("total_token_seconds", 0.0)) for r in rows
    ]
    health = {
        "actual_total_seconds": sum(token_seconds),
        "max_token_seconds": max(token_seconds) if token_seconds else 0.0,
        "failed_token_count": sum(
            1 for r in rows if r.get("status") not in {"ok", "probe_completed"}
        ),
        "diagnostic_token_count": sum(
            1 for r in rows if r.get("status") == "probe_completed"
        ),
        "predicted_cost_sum": int(
            shard.get("estimated_cost_sum", sum(s["estimated_cost"] for s in specs))
        ),
        "retry_recommended": any(
            r.get("status") not in {"ok", "probe_completed"} for r in rows
        ),
    }
    terminal_resource = None
    if resource_policy is not RuntimeResourcePolicy.OFF:
        terminal_resource = resource_sample(
            boundary="shard_terminal", started_monotonic=resource_started
        )
        with resource_samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(terminal_resource, sort_keys=True) + "\n")
        write_json(
            root / "resource_summary.json",
            {
                "schema_version": 1,
                "runtime_policy": resource_policy.value,
                "planning_envelope": planning_envelope,
                "scheduler_request": resource_payload["scheduler_request"],
                "scheduler_allocation": resource_payload["scheduler_allocation"],
                "terminal_sample": terminal_resource,
                "trace_telemetry_summaries": [
                    row.get("telemetry_summary") for row in rows
                ],
            },
        )
    write_json(
        root / "shard.json",
        _shard_record(
            status=final_status,
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata={**metadata, "shard_health": health},
            target_positions=[spec["target_position"] for spec in specs],
        ),
    )
    return {"shard_dir": str(root), "token_count": len(rows), "status": final_status}


def _summary_row(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "trace_id": spec["trace_id"],
        "trajectory_id": spec["trajectory_id"],
        "generated_index": spec["generated_index"],
        "target_position": spec["target_position"],
        "prefix_token_count": spec["prefix_token_count"],
        "target_token_id": spec["target_token_id"],
        "target_token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "estimated_cost": spec["estimated_cost"],
        "selection_reasons": spec["selection_reasons"],
    }


def _exception_payload(exc: Exception) -> dict[str, Any]:
    """Return diagnostic-safe exception metadata for per-token trace failures.

    nnsight wraps inner exceptions in dynamically generated ``NNsightException``
    classes whose ``repr`` is often just ``NNsightException()``.  ``str(exc)``
    and ``print_exception`` carry the reconstructed trace, so preserve those in
    the token trace JSON instead of only storing ``repr(exc)``.
    """

    from circuit_tracer.observability.errors import safe_exception_attrs

    primary = safe_exception_attrs(exc)
    try:
        formatted_traceback = traceback.format_exc()
    except BaseException as formatting_error:  # pragma: no cover - adversarial
        formatted_traceback = (
            "<traceback unavailable: "
            f"{type(formatting_error).__module__}."
            f"{type(formatting_error).__qualname__}>"
        )
    payload: dict[str, Any] = {
        "error": primary["error_repr"],
        "error_type": primary["error_type"],
        "error_module": type(exc).__module__,
        "error_message": primary["error_message"],
        "traceback": formatted_traceback,
    }
    if "error_details" in primary:
        payload["error_details"] = primary["error_details"]
    try:
        original = getattr(exc, "original", None)
    except BaseException:  # pragma: no cover - adversarial wrappers
        original = None
    if original is not None:
        original_attrs = safe_exception_attrs(original)
        payload.update(
            {
                "original_error_type": original_attrs["error_type"],
                "original_error_module": type(original).__module__,
                "original_error": original_attrs["error_repr"],
                "original_error_message": original_attrs["error_message"],
            }
        )
    cause = exc.__cause__
    if cause is not None:
        cause_attrs = safe_exception_attrs(cause)
        payload["cause_error"] = cause_attrs["error_repr"]
        payload["cause_error_type"] = cause_attrs["error_type"]
    context = exc.__context__
    if context is not None:
        context_attrs = safe_exception_attrs(context)
        payload["context_error"] = context_attrs["error_repr"]
        payload["context_error_type"] = context_attrs["error_type"]
    if "error_details" not in payload:
        for candidate in _iter_exception_chain(exc)[1:]:
            candidate_attrs = safe_exception_attrs(candidate)
            if "error_details" in candidate_attrs:
                payload["error_details"] = candidate_attrs["error_details"]
                break
    return payload


def _trace_payload(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    forced_target = forced_target_payload(spec)
    payload = _summary_row(spec, shard_id=shard_id)
    payload.update(
        {
            "schema_version": 1,
            "status": "dry_run",
            "graph_path": None,
            "graph_knobs": spec["graph_knobs"],
            "forced_target": forced_target,
            "error": None,
        }
    )
    return payload
