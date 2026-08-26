from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from ..backward_selection import (
    BACKWARD_ENGINE_PRESETS,
    normalize_backward_overrides,
    resolve_backward_execution_selection,
)
from ..config import (
    CORRECTNESS_POLICY_IDS,
    CORRECTNESS_PROBE_MODES,
    DEFAULT_CORRECTNESS_POLICY_ID,
    DEFAULT_CORRECTNESS_PROBE_MODE,
    base_trace_defaults,
)
from ..io_utils import iter_jsonl, read_json, write_json, write_jsonl
from ..transcoder_config import resolve_transcoder_load_config

SCHEMA_VERSION = 1
TARGET_MODE = "frozen_target_only"
INPUT_CONTEXT_MODES = {"independent_prefix", "full_sequence"}
TRAJECTORY_SESSION_MODES = {"per_token", "experimental_reuse", "window_reuse_v1"}
PHASE0_WINDOW_SCOPES = {"shard_window", "trajectory"}
PHASE0_WINDOW_MAX_PREFIX_POLICIES = {"max_target_position"}
PHASE0_WINDOW_REFERENCE_CHECKS = {"off", "sampled", "all"}
ROW_STORE_CACHE_CONTROLS = {
    "off",
    "fadvise_dontneed_after_append_v1",
    "fadvise_dontneed_after_append_and_read_v1",
}
FEATURE_ROW_INFLUENCE_MODES = {
    "cpu_exact",
    "cpu_prepared",
    "cuda_full",
    "cuda_windowed",
    "auto",
}
FEATURE_ROW_INFLUENCE_REQUIREMENTS = {"preferred", "required"}
RUNTIME_RESOURCE_POLICIES = {"off", "measure_only", "enforce"}
BACKWARD_ENGINE_MODES = frozenset(BACKWARD_ENGINE_PRESETS)
EDGE_RETENTION_POLICY_IDS = {"typed_top_p_v1"}


def _is_sha256_fingerprint(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


class GeneratedToken(TypedDict, total=False):
    generated_index: int
    absolute_token_position: int
    token_id: int
    token_text: str
    logprob: float | None
    probability: float | None
    rank: int | None
    is_stop: bool


class Trajectory(TypedDict, total=False):
    schema_version: int
    trajectory_id: str
    prompt_token_count: int
    prompt_token_ids: list[int]
    generated_tokens: list[GeneratedToken]
    prompt_text: str
    fixture_metadata: dict[str, Any]


class TraceSpec(TypedDict):
    schema_version: int
    trace_id: str
    trajectory_id: str
    generated_index: int
    target_position: int
    prefix_token_count: int
    target_token_id: int
    target_token_text: str
    target_mode: str
    selection_reasons: list[str]
    graph_knobs: dict[str, Any]
    estimated_cost: int


def load_trajectory(path: Path) -> Trajectory:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"trajectory must be a JSON object: {path}")
    validate_trajectory(payload)
    return payload


def write_trajectory(path: Path, trajectory: Trajectory) -> None:
    validate_trajectory(trajectory)
    write_json(path, cast(dict[str, Any], trajectory))


def validate_trajectory(trajectory: Mapping[str, Any]) -> None:
    if trajectory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trajectory schema_version must be 1")
    if not trajectory.get("trajectory_id"):
        raise ValueError("trajectory_id is required")
    prompt_token_count = trajectory.get("prompt_token_count")
    if not isinstance(prompt_token_count, int) or prompt_token_count < 0:
        raise ValueError("prompt_token_count must be an int")
    prompt_token_ids = trajectory.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("prompt_token_ids must be a list of ints")
    if len(prompt_token_ids) != prompt_token_count:
        raise ValueError("prompt_token_count must match len(prompt_token_ids)")
    tokens = trajectory.get("generated_tokens")
    if not isinstance(tokens, list):
        raise ValueError("generated_tokens must be a list")
    for expected_index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError("generated token entries must be objects")
        token_payload = cast(Mapping[str, Any], token)
        if token_payload.get("generated_index") != expected_index:
            raise ValueError("generated token indices must be contiguous from 0")
        expected_position = prompt_token_count + expected_index
        if token_payload.get("absolute_token_position") != expected_position:
            raise ValueError(
                "generated token absolute_token_position must equal "
                "prompt_token_count + generated_index"
            )
        if not isinstance(token_payload.get("token_id"), int):
            raise ValueError(
                f"generated token {expected_index} token_id must be an int"
            )
        if not isinstance(token_payload.get("token_text"), str):
            raise ValueError(
                f"generated token {expected_index} token_text must be a string"
            )
        if not isinstance(token_payload.get("is_stop"), bool):
            raise ValueError(f"generated token {expected_index} is_stop must be a bool")


def validate_trace_selection(
    selection: Mapping[str, Any],
    *,
    trajectory: Trajectory | None = None,
) -> None:
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace selection schema_version must be 1")
    trajectory_id = selection.get("trajectory_id")
    if not trajectory_id:
        raise ValueError("trace selection trajectory_id is required")
    if trajectory is not None:
        validate_trajectory(trajectory)
        if trajectory_id != trajectory["trajectory_id"]:
            raise ValueError("trace selection trajectory_id does not match trajectory")
        token_count = len(trajectory["generated_tokens"])
    else:
        token_count = None

    selected_indices = selection.get("selected_indices")
    if not isinstance(selected_indices, list) or not all(
        isinstance(index, int) for index in selected_indices
    ):
        raise ValueError("selected_indices must be a list of ints")
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError("selected_indices must be sorted and unique")
    if token_count is not None:
        for index in selected_indices:
            if index < 0 or index >= token_count:
                raise ValueError(f"selected index out of bounds: {index}")

    reasons = selection.get("selection_reasons")
    if not isinstance(reasons, dict):
        raise ValueError("selection_reasons must be an object")
    for index in selected_indices:
        key = str(index)
        values = reasons.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"selection_reasons missing non-empty reasons for {key}")
        if not all(isinstance(reason, str) and reason for reason in values):
            raise ValueError(f"selection_reasons for {key} must be strings")


def validate_trace_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trace spec schema_version must be 1")
    for key in ("trace_id", "trajectory_id", "target_token_text"):
        if not isinstance(spec.get(key), str):
            raise ValueError(f"trace spec {key} must be a string")
    for key in (
        "generated_index",
        "target_position",
        "prefix_token_count",
        "target_token_id",
        "estimated_cost",
    ):
        value = spec.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"trace spec {key} must be a non-negative int")
    if spec.get("target_mode") != TARGET_MODE:
        raise ValueError(f"trace spec target_mode must be {TARGET_MODE!r}")
    if spec["target_position"] != spec["prefix_token_count"]:
        raise ValueError(
            "trace spec target_position must equal prefix_token_count for "
            "independent-prefix full-answer traces"
        )
    if not isinstance(spec.get("selection_reasons"), list) or not all(
        isinstance(reason, str) for reason in spec.get("selection_reasons", [])
    ):
        raise ValueError("trace spec selection_reasons must be a list of strings")
    if not isinstance(spec.get("graph_knobs"), dict):
        raise ValueError("trace spec graph_knobs must be an object")
    if "max_edges" in spec["graph_knobs"]:
        raise ValueError(
            "trace spec graph_knobs.max_edges is a legacy global projection; "
            "use edge_retention_policy_id"
        )
    retention_policy_id = spec["graph_knobs"].get("edge_retention_policy_id")
    if retention_policy_id not in EDGE_RETENTION_POLICY_IDS:
        raise ValueError(
            "trace spec graph_knobs.edge_retention_policy_id must be one of "
            f"{sorted(EDGE_RETENTION_POLICY_IDS)!r}"
        )
    correctness_probe_mode = spec["graph_knobs"].get("correctness_probe_mode")
    if correctness_probe_mode not in CORRECTNESS_PROBE_MODES:
        raise ValueError(
            "trace spec graph_knobs.correctness_probe_mode must be one of "
            f"{sorted(CORRECTNESS_PROBE_MODES)!r}"
        )
    correctness_policy_id = spec["graph_knobs"].get("correctness_policy_id")
    if correctness_policy_id not in CORRECTNESS_POLICY_IDS:
        raise ValueError(
            "trace spec graph_knobs.correctness_policy_id must be one of "
            f"{sorted(CORRECTNESS_POLICY_IDS)!r}"
        )
    numerical_manifest_path = spec["graph_knobs"].get(
        "correctness_numerical_manifest_path"
    )
    numerical_manifest_sha256 = spec["graph_knobs"].get(
        "correctness_numerical_manifest_sha256"
    )
    if (numerical_manifest_path is None) != (numerical_manifest_sha256 is None):
        raise ValueError(
            "trace spec numerical correctness manifest path and sha256 must be "
            "declared together"
        )
    if numerical_manifest_path is not None:
        if (
            not isinstance(numerical_manifest_path, str)
            or not numerical_manifest_path
            or not Path(numerical_manifest_path).is_absolute()
        ):
            raise ValueError(
                "trace spec graph_knobs.correctness_numerical_manifest_path "
                "must be an absolute path"
            )
        if not _is_sha256_fingerprint(numerical_manifest_sha256):
            raise ValueError(
                "trace spec graph_knobs.correctness_numerical_manifest_sha256 "
                "must be a sha256 fingerprint"
            )
    try:
        backward_selection = resolve_backward_execution_selection(spec["graph_knobs"])
    except ValueError as error:
        raise ValueError(f"trace spec graph_knobs {error}") from error
    if (
        backward_selection.vjp_kernel_mode in {"autograd_batched", "autograd_serial"}
        and spec["graph_knobs"].get("phase3_gradient_replay_mode", "disabled")
        != "disabled"
    ):
        raise ValueError(
            f"{backward_selection.backward_engine_mode} does not support "
            "phase3_gradient_replay_mode"
        )
    input_context_mode = spec["graph_knobs"].get("input_context_mode")
    if input_context_mode is not None and input_context_mode not in INPUT_CONTEXT_MODES:
        raise ValueError(
            "trace spec graph_knobs.input_context_mode must be one of "
            f"{sorted(INPUT_CONTEXT_MODES)!r}"
        )
    session_mode = spec["graph_knobs"].get("trajectory_session_mode")
    if session_mode is not None and session_mode not in TRAJECTORY_SESSION_MODES:
        raise ValueError(
            "trace spec graph_knobs.trajectory_session_mode must be one of "
            f"{sorted(TRAJECTORY_SESSION_MODES)!r}"
        )
    for key in (
        "reuse_phase0_window_state",
        "reuse_target_logits",
        "decoder_active_row_residency",
        "phase0_decoder_row_ranges",
    ):
        value = spec["graph_knobs"].get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"trace spec graph_knobs.{key} must be a bool")
    active_row_max_bytes = spec["graph_knobs"].get("decoder_active_row_max_bytes", 0)
    active_row_safety_margin_bytes = spec["graph_knobs"].get(
        "decoder_active_row_safety_margin_bytes", 0
    )
    for key, value in (
        ("decoder_active_row_max_bytes", active_row_max_bytes),
        ("decoder_active_row_safety_margin_bytes", active_row_safety_margin_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"trace spec graph_knobs.{key} must be a non-negative int")
    active_row_requirement = spec["graph_knobs"].get(
        "decoder_active_row_residency_requirement", "preferred"
    )
    if active_row_requirement not in {"preferred", "required"}:
        raise ValueError(
            "trace spec graph_knobs.decoder_active_row_residency_requirement "
            "must be preferred or required"
        )
    if (
        active_row_requirement == "required"
        and spec["graph_knobs"].get("decoder_active_row_residency") is not True
    ):
        raise ValueError(
            "required decoder active-row residency requires "
            "decoder_active_row_residency=true"
        )
    if (
        spec["graph_knobs"].get("decoder_active_row_residency") is True
        and active_row_max_bytes == 0
        and active_row_safety_margin_bytes == 0
    ):
        raise ValueError(
            "dynamic decoder active-row residency requires a positive "
            "decoder_active_row_safety_margin_bytes"
        )
    if spec["graph_knobs"].get("phase0_decoder_row_ranges"):
        provider = resolve_transcoder_load_config(
            spec["graph_knobs"], preserve_default_values=True
        )
        if provider.transcoder_architecture != "plt":
            raise ValueError(
                "trace spec graph_knobs.phase0_decoder_row_ranges requires "
                "a PLT-compatible provider"
            )
        if spec["graph_knobs"].get("decoder_active_row_residency") is not True:
            raise ValueError(
                "trace spec graph_knobs.phase0_decoder_row_ranges requires "
                "decoder_active_row_residency=true"
            )
        if spec["graph_knobs"].get("reuse_phase0_window_state"):
            raise ValueError(
                "trace spec graph_knobs.phase0_decoder_row_ranges is "
                "incompatible with reuse_phase0_window_state until "
                "forward-session policy is shared"
            )
    phase0_window_scope = spec["graph_knobs"].get("phase0_window_scope")
    if (
        phase0_window_scope is not None
        and phase0_window_scope not in PHASE0_WINDOW_SCOPES
    ):
        raise ValueError(
            "trace spec graph_knobs.phase0_window_scope must be one of "
            f"{sorted(PHASE0_WINDOW_SCOPES)!r}"
        )
    max_prefix_policy = spec["graph_knobs"].get("phase0_window_max_prefix_policy")
    if (
        max_prefix_policy is not None
        and max_prefix_policy not in PHASE0_WINDOW_MAX_PREFIX_POLICIES
    ):
        raise ValueError(
            "trace spec graph_knobs.phase0_window_max_prefix_policy must be one of "
            f"{sorted(PHASE0_WINDOW_MAX_PREFIX_POLICIES)!r}"
        )
    reference_checks = spec["graph_knobs"].get("phase0_window_reference_checks")
    if (
        reference_checks is not None
        and reference_checks not in PHASE0_WINDOW_REFERENCE_CHECKS
    ):
        raise ValueError(
            "trace spec graph_knobs.phase0_window_reference_checks must be one of "
            f"{sorted(PHASE0_WINDOW_REFERENCE_CHECKS)!r}"
        )
    row_store_cache_control = spec["graph_knobs"].get("row_store_cache_control")
    if (
        row_store_cache_control is not None
        and row_store_cache_control not in ROW_STORE_CACHE_CONTROLS
    ):
        raise ValueError(
            "trace spec graph_knobs.row_store_cache_control must be one of "
            f"{sorted(ROW_STORE_CACHE_CONTROLS)!r}"
        )
    influence_mode = spec["graph_knobs"].get("feature_row_influence_mode", "cpu_exact")
    if influence_mode not in FEATURE_ROW_INFLUENCE_MODES:
        raise ValueError(
            "trace spec graph_knobs.feature_row_influence_mode must be one of "
            f"{sorted(FEATURE_ROW_INFLUENCE_MODES)!r}"
        )
    influence_requirement = spec["graph_knobs"].get(
        "feature_row_influence_requirement", "preferred"
    )
    if influence_requirement not in FEATURE_ROW_INFLUENCE_REQUIREMENTS:
        raise ValueError(
            "trace spec graph_knobs.feature_row_influence_requirement must be "
            "preferred or required"
        )
    if influence_mode == "auto" and influence_requirement == "required":
        raise ValueError("auto feature-row influence cannot be required")
    resident_bytes = spec["graph_knobs"].get("feature_row_gpu_resident_max_bytes", 0)
    window_bytes = spec["graph_knobs"].get("feature_row_gpu_window_max_bytes", 0)
    safety_bytes = spec["graph_knobs"].get(
        "feature_row_gpu_resident_safety_margin_bytes", 0
    )
    for key, value in (
        ("feature_row_gpu_resident_max_bytes", resident_bytes),
        ("feature_row_gpu_window_max_bytes", window_bytes),
        ("feature_row_gpu_resident_safety_margin_bytes", safety_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"trace spec graph_knobs.{key} must be a non-negative int")
    if influence_mode == "cuda_full" and resident_bytes == 0:
        raise ValueError(
            "cuda_full feature-row influence requires a resident byte budget"
        )
    if influence_mode == "cuda_windowed" and window_bytes == 0:
        raise ValueError(
            "cuda_windowed feature-row influence requires a window byte budget"
        )
    if influence_mode == "auto" and (resident_bytes == 0 or window_bytes == 0):
        raise ValueError(
            "auto feature-row influence requires resident and window budgets"
        )
    resource_policy = spec["graph_knobs"].get("runtime_resource_policy", "off")
    if resource_policy not in RUNTIME_RESOURCE_POLICIES:
        raise ValueError(
            "trace spec graph_knobs.runtime_resource_policy must be off, "
            "measure_only, or enforce"
        )
    planning_envelope = spec["graph_knobs"].get("resource_planning_envelope", {})
    if not isinstance(planning_envelope, dict):
        raise ValueError(
            "trace spec graph_knobs.resource_planning_envelope must be an object"
        )
    if resource_policy == "enforce" and not any(
        key in planning_envelope for key in ("host_rss_stop_gib", "walltime_seconds")
    ):
        raise ValueError(
            "enforce resource policy requires host_rss_stop_gib or walltime_seconds"
        )
    diagnostic_mode = spec["graph_knobs"].get("diagnostic_stop_mode", "none")
    diagnostic_batches = spec["graph_knobs"].get("diagnostic_stop_phase4_batches")
    if diagnostic_mode not in {
        "none",
        "phase0_probe",
        "phase3_probe",
        "transition_probe",
    }:
        raise ValueError(
            "trace spec graph_knobs.diagnostic_stop_mode must be none, "
            "phase0_probe, phase3_probe, or transition_probe"
        )
    if diagnostic_mode == "transition_probe":
        if (
            isinstance(diagnostic_batches, bool)
            or not isinstance(diagnostic_batches, int)
            or diagnostic_batches <= 0
        ):
            raise ValueError("transition_probe requires positive phase4 batches")
    elif diagnostic_batches is not None:
        raise ValueError(
            "diagnostic_stop_phase4_batches is valid only for transition_probe"
        )
    if (
        spec["graph_knobs"].get("reuse_target_logits")
        and session_mode != "window_reuse_v1"
    ):
        raise ValueError(
            "trace spec graph_knobs.reuse_target_logits requires "
            "trajectory_session_mode='window_reuse_v1'"
        )
    if (
        spec["graph_knobs"].get("reuse_phase0_window_state")
        and session_mode != "window_reuse_v1"
    ):
        raise ValueError(
            "trace spec graph_knobs.reuse_phase0_window_state requires "
            "trajectory_session_mode='window_reuse_v1'"
        )
    if session_mode == "window_reuse_v1":
        if phase0_window_scope not in (None, "shard_window"):
            raise ValueError(
                "window_reuse_v1 currently supports only "
                "phase0_window_scope='shard_window'"
            )
        if reference_checks not in (None, "off"):
            raise ValueError(
                "window_reuse_v1 phase0_window_reference_checks sampled/all are "
                "not implemented; use 'off'"
            )


def normalize_trace_spec(spec: Mapping[str, Any]) -> TraceSpec:
    """Return a current trace spec, accepting older v1 rows.

    Early full-answer v1 specs did not persist ``target_position`` because the
    independent-prefix target is numerically identical to ``prefix_token_count``.
    Specs predating the correctness bridge also imply the compatibility-off
    selection. Keep those provenance artifacts loadable while ensuring every
    in-memory spec now carries both explicit contracts.
    """
    payload = dict(spec)
    if "target_position" not in payload:
        payload["target_position"] = payload.get("prefix_token_count")
    graph_knobs = payload.get("graph_knobs")
    if isinstance(graph_knobs, Mapping):
        normalized_knobs = dict(graph_knobs)
        normalized_knobs.setdefault(
            "correctness_probe_mode", DEFAULT_CORRECTNESS_PROBE_MODE
        )
        normalized_knobs.setdefault(
            "correctness_policy_id", DEFAULT_CORRECTNESS_POLICY_ID
        )
        payload["graph_knobs"] = normalized_knobs
    validate_trace_spec(payload)
    return cast(TraceSpec, payload)


def write_trace_selection(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def load_trace_selection(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"trace selection must be a JSON object: {path}")
    validate_trace_selection(payload)
    return payload


def build_trace_specs(
    trajectory: Trajectory,
    selection: dict[str, Any],
    *,
    graph_knob_overrides: dict[str, Any] | None = None,
) -> list[TraceSpec]:
    validate_trajectory(trajectory)
    validate_trace_selection(selection, trajectory=trajectory)
    trajectory_id = str(trajectory["trajectory_id"])
    prompt_token_count = int(trajectory["prompt_token_count"])
    generated_tokens = trajectory["generated_tokens"]
    reasons_by_index = {
        int(index): list(reasons)
        for index, reasons in selection.get("selection_reasons", {}).items()
    }
    overrides = graph_knob_overrides or {}
    knobs = base_trace_defaults()
    knobs.update(overrides)
    normalize_backward_overrides(knobs, overrides)
    specs: list[TraceSpec] = []
    for generated_index in selection.get("selected_indices", []):
        token = generated_tokens[generated_index]
        prefix_token_count = prompt_token_count + generated_index
        target_position = prompt_token_count + generated_index
        specs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "trace_id": f"{trajectory_id}_tok{generated_index:06d}",
                "trajectory_id": trajectory_id,
                "generated_index": generated_index,
                "target_position": target_position,
                "prefix_token_count": prefix_token_count,
                "target_token_id": int(token["token_id"]),
                "target_token_text": str(token.get("token_text", "")),
                "target_mode": TARGET_MODE,
                "selection_reasons": reasons_by_index.get(generated_index, []),
                "graph_knobs": dict(knobs),
                "estimated_cost": prefix_token_count,
            }
        )
    return specs


def write_trace_specs(path: Path, specs: list[TraceSpec]) -> None:
    for spec in specs:
        validate_trace_spec(spec)
    write_jsonl(path, cast(list[dict[str, Any]], specs))


def load_trace_specs(path: Path) -> list[TraceSpec]:
    specs: list[TraceSpec] = []
    for row in iter_jsonl(path):
        specs.append(normalize_trace_spec(row))
    return specs


def write_shards(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def load_shards(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"shards must be a JSON object: {path}")
    validate_shards(payload)
    return payload


def validate_shards(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("shards schema_version must be 1")
    if not isinstance(payload.get("trace_specs_file"), str):
        raise ValueError("shards trace_specs_file must be a string")
    if not isinstance(payload.get("cost_model"), str):
        raise ValueError("shards cost_model must be a string")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise ValueError("shards must contain a list of shard objects")
    for expected_id, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError("shard entries must be objects")
        shard_payload = cast(Mapping[str, Any], shard)
        if shard_payload.get("shard_id") != expected_id:
            raise ValueError("shard_id values must be contiguous from 0")
        cost = shard_payload.get("estimated_cost_sum")
        if not isinstance(cost, int) or cost < 0:
            raise ValueError("estimated_cost_sum must be a non-negative int")
        spec_indices = shard_payload.get("spec_indices")
        if not isinstance(spec_indices, list) or not all(
            isinstance(index, int) and index >= 0 for index in spec_indices
        ):
            raise ValueError("spec_indices must be a list of non-negative ints")
