"""Canonical full-answer execution selection and resource observation."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..backward_selection import backward_mechanism_record
from .schemas import TraceSpec


class RuntimeResourcePolicy(str, Enum):
    OFF = "off"
    MEASURE_ONLY = "measure_only"
    ENFORCE = "enforce"


_ENFORCEABLE_RESOURCE_LIMITS = frozenset({"host_rss_stop_gib", "walltime_seconds"})


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_config_groups(specs: Sequence[TraceSpec]) -> list[dict[str, Any]]:
    if not specs:
        raise ValueError("selected execution requires at least one trace spec")
    groups: dict[str, dict[str, Any]] = {}
    for spec in specs:
        selected = dict(spec["graph_knobs"])
        config_fingerprint = _fingerprint(selected)
        group = groups.setdefault(
            config_fingerprint,
            {
                "config_fingerprint": config_fingerprint,
                "selected_config": selected,
                "trace_ids": [],
            },
        )
        group["trace_ids"].append(spec["trace_id"])
    return list(groups.values())


def _shared_resource_selection(
    config_groups: Sequence[Mapping[str, Any]],
) -> tuple[RuntimeResourcePolicy, dict[str, Any]]:
    selections: list[tuple[RuntimeResourcePolicy, dict[str, Any]]] = []
    for group in config_groups:
        knobs = group["selected_config"]
        assert isinstance(knobs, Mapping)
        selections.append(
            (
                _resource_policy(knobs),
                dict(knobs.get("resource_planning_envelope", {})),
            )
        )
    first = selections[0]
    if any(candidate != first for candidate in selections[1:]):
        raise ValueError(
            "all specs in one execution must share runtime resource policy and "
            "planning envelope"
        )
    return first


def _resource_policy(knobs: Mapping[str, Any]) -> RuntimeResourcePolicy:
    try:
        policy = RuntimeResourcePolicy(str(knobs.get("runtime_resource_policy", "off")))
    except ValueError as error:
        raise ValueError(
            "runtime_resource_policy must be off, measure_only, or enforce"
        ) from error
    envelope = knobs.get("resource_planning_envelope", {})
    if not isinstance(envelope, Mapping):
        raise ValueError("resource_planning_envelope must be an object")
    if policy is RuntimeResourcePolicy.ENFORCE and not any(
        key in envelope for key in _ENFORCEABLE_RESOURCE_LIMITS
    ):
        raise ValueError(
            "enforce resource policy requires host_rss_stop_gib or walltime_seconds"
        )
    return policy


def slurm_allocation_record(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_END_TIME",
        "CUDA_VISIBLE_DEVICES",
    )
    return {name: values.get(name) for name in names}


def scheduler_request_record(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    names = (
        "EXACT_TRACE_REQUESTED_CLUSTER",
        "EXACT_TRACE_REQUESTED_RESOURCE_PROFILE",
        "EXACT_TRACE_REQUESTED_ACCOUNT",
        "EXACT_TRACE_REQUESTED_PARTITION",
        "EXACT_TRACE_REQUESTED_QOS",
        "EXACT_TRACE_REQUESTED_GRES",
        "EXACT_TRACE_REQUESTED_GPUS_PER_TASK",
        "EXACT_TRACE_REQUESTED_CPUS",
        "EXACT_TRACE_REQUESTED_MEM",
        "EXACT_TRACE_REQUESTED_WALLTIME",
    )
    return {name: values.get(name) for name in names}


def selected_execution_record(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    specs: Sequence[TraceSpec],
    shard: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    config_groups = _selected_config_groups(specs)
    policy, planning_envelope = _shared_resource_selection(config_groups)
    homogeneous = len(config_groups) == 1
    selected = config_groups[0]["selected_config"] if homogeneous else None
    mechanism_selections = [
        {
            "config_fingerprint": group["config_fingerprint"],
            "trace_ids": group["trace_ids"],
            "feature_row_influence_mode": group["selected_config"].get(
                "feature_row_influence_mode", "cpu_exact"
            ),
            "feature_row_influence_requirement": group["selected_config"].get(
                "feature_row_influence_requirement", "preferred"
            ),
            **backward_mechanism_record(group["selected_config"]),
        }
        for group in config_groups
    ]
    scheduler_request = scheduler_request_record()
    selection_contract = {
        "schema_version": 1,
        "workload": {
            "trajectory_path": str(trajectory_path),
            "trace_specs_path": str(trace_specs_path),
            "shards_path": str(shards_path),
            "trajectory_id": specs[0]["trajectory_id"],
            "trace_ids": [spec["trace_id"] for spec in specs],
            "prefix_token_counts": [spec["prefix_token_count"] for spec in specs],
            "target_token_ids": [spec["target_token_id"] for spec in specs],
            "shard": dict(shard),
        },
        "configuration_mode": "homogeneous" if homogeneous else "per_trace",
        "selected_config": selected,
        "selected_configs": config_groups,
        "mechanism_selection": (
            {
                "feature_row_influence_mode": mechanism_selections[0][
                    "feature_row_influence_mode"
                ],
                "feature_row_influence_requirement": mechanism_selections[0][
                    "feature_row_influence_requirement"
                ],
                "backward_engine_mode": mechanism_selections[0]["backward_engine_mode"],
                "forward_graph_mode": mechanism_selections[0]["forward_graph_mode"],
                "vjp_kernel_mode": mechanism_selections[0]["vjp_kernel_mode"],
                "forward_lane_count": mechanism_selections[0]["forward_lane_count"],
                "backward_selection_source": mechanism_selections[0][
                    "backward_selection_source"
                ],
            }
            if homogeneous
            else None
        ),
        "mechanism_selections": mechanism_selections,
        "resource_selection": {
            "runtime_policy": policy.value,
            "planning_envelope": planning_envelope,
            "scheduler_request": scheduler_request,
        },
    }
    selection = {
        **selection_contract,
        "resources": {
            "runtime_policy": policy.value,
            "planning_envelope": planning_envelope,
            "runtime_enforced_fields": sorted(
                key
                for key in planning_envelope
                if policy is RuntimeResourcePolicy.ENFORCE
                and key in _ENFORCEABLE_RESOURCE_LIMITS
            ),
            "scheduler_request": scheduler_request,
            "scheduler_allocation": slurm_allocation_record(),
        },
        "provenance": {
            key: metadata.get(key)
            for key in (
                "run_id",
                "run_name",
                "run_description",
                "run_goal",
                "workspace_root",
                "library_workspace_root",
                "slurm_job_id",
                "slurm_array_task_id",
            )
        },
    }
    return {
        **selection,
        "selection_fingerprint": _fingerprint(selection_contract),
        "record_fingerprint": _fingerprint(selection),
    }


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_v2_root() -> Path | None:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(":", maxsplit=2)
        if len(parts) == 3 and parts[0] == "0":
            path = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
            return path if path.exists() else None
    return None


def resource_sample(*, boundary: str, started_monotonic: float) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cgroup = _cgroup_v2_root()
    sample: dict[str, Any] = {
        "schema_version": 1,
        "boundary": boundary,
        "wall_time_epoch_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "process_max_rss_kib": int(usage.ru_maxrss),
    }
    if cgroup is not None:
        sample.update(
            {
                "cgroup_path": str(cgroup),
                "cgroup_memory_current_bytes": _read_int(cgroup / "memory.current"),
                "cgroup_memory_peak_bytes": _read_int(cgroup / "memory.peak"),
                "cgroup_memory_max_bytes": _read_int(cgroup / "memory.max"),
            }
        )
    return sample


_RESOLUTION_FIELDS: tuple[tuple[str, tuple[str, str], str, str], ...] = (
    (
        "backward_engine_mode",
        ("batches", "backward_engine_mode"),
        "preparation",
        "physical",
    ),
    (
        "forward_graph_mode",
        ("batches", "forward_graph_mode"),
        "preparation",
        "physical",
    ),
    (
        "vjp_kernel_mode",
        ("batches", "vjp_kernel_mode"),
        "preparation",
        "physical",
    ),
    (
        "forward_lane_count",
        ("batches", "forward_lane_count"),
        "preparation",
        "physical",
    ),
    (
        "feature_row_influence_mode",
        ("storage", "feature_row_influence_mode_resolved"),
        "phase2_row_store",
        "physical",
    ),
    (
        "feature_row_influence_requirement",
        ("storage", "feature_row_influence_requirement"),
        "request_validation",
        "physical",
    ),
    (
        "attribution_batch_size",
        ("batches", "source_batch_size"),
        "preparation",
        "semantic",
    ),
    (
        "feature_batch_size",
        ("batches", "feature_batch_size"),
        "preparation",
        "semantic",
    ),
    ("logit_batch_size", ("batches", "logit_batch_size"), "preparation", "semantic"),
    (
        "nnsight_session_capacity",
        ("batches", "session_capacity"),
        "preparation",
        "physical",
    ),
    (
        "phase3_compute_microbatch_max_rows",
        ("batches", "phase3_microbatch_max_rows"),
        "preparation",
        "physical",
    ),
    (
        "phase4_execution_batch_max_rows",
        ("batches", "phase4_execution_batch_max_rows"),
        "preparation",
        "physical",
    ),
    (
        "phase4_scheduler_mode",
        ("frontier", "scheduler_mode"),
        "preparation",
        "physical",
    ),
    (
        "phase4_refresh_optimization",
        ("frontier", "refresh_optimization_mode"),
        "preparation",
        "physical",
    ),
    (
        "phase4_row_executor",
        ("frontier", "row_executor_mode"),
        "preparation",
        "physical",
    ),
    (
        "phase4_row_reduction",
        ("frontier", "row_reduction_mode"),
        "preparation",
        "physical",
    ),
    (
        "exact_encoder_residency",
        ("frontier", "exact_encoder_residency_mode"),
        "preparation",
        "physical",
    ),
)


def execution_resolution(
    *,
    selected_config: Mapping[str, Any],
    effective_execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if effective_execution is None:
        return {"schema_version": 1, "status": "unavailable", "fields": []}
    fields = []
    backward_selection = backward_mechanism_record(selected_config)
    for requested_name, (
        group,
        effective_name,
    ), stage, classification in _RESOLUTION_FIELDS:
        requested = backward_selection.get(
            requested_name, selected_config.get(requested_name)
        )
        effective_group = effective_execution.get(group, {})
        effective = (
            effective_group.get(effective_name)
            if isinstance(effective_group, Mapping)
            else None
        )
        reason = None
        if requested_name == "feature_row_influence_mode" and isinstance(
            effective_group, Mapping
        ):
            reason = effective_group.get("feature_row_influence_resolution_reason")
        fields.append(
            {
                "field": requested_name,
                "requested": requested,
                "effective": effective,
                "changed": requested != effective,
                "resolution_stage": stage,
                "reason": reason,
                "classification": classification,
            }
        )
    return {"schema_version": 1, "status": "resolved", "fields": fields}


def validate_resource_policy(
    *,
    policy: RuntimeResourcePolicy,
    envelope: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> None:
    if policy is not RuntimeResourcePolicy.ENFORCE:
        return
    elapsed_limit = envelope.get("walltime_seconds")
    if (
        isinstance(elapsed_limit, (int, float))
        and sample.get("elapsed_seconds", 0) > elapsed_limit
    ):
        raise RuntimeError("runtime resource policy exceeded walltime_seconds")
    rss_limit_gib = envelope.get("host_rss_stop_gib")
    rss_kib = sample.get("process_max_rss_kib")
    if (
        isinstance(rss_limit_gib, (int, float))
        and isinstance(rss_kib, int)
        and rss_kib > float(rss_limit_gib) * 1024**2
    ):
        raise RuntimeError("runtime resource policy exceeded host_rss_stop_gib")
