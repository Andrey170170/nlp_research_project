"""Normalized project-owned calibration observations.

The tracing library owns runtime telemetry.  This module joins those persisted
records with harness outcome, comparison, campaign, and provenance data without
creating a second canonical telemetry stream.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from circuit_tracer.governor.calibration import DEFAULT_KNOB_SENSITIVITIES

from .baselines import COMPARISON_SUMMARY_KEYS, SCENARIO_KNOB_KEYS
from .io_utils import iter_jsonl, read_json, write_json


CALIBRATION_OBSERVATION_SCHEMA_VERSION = 2
FIDELITY_MODES = frozenset({"exact", "bounded", "best_effort", "research"})
_SCENARIO_AXIS_IDS = {
    "attribution_batch_size": "source_batch_size",
    "cross_batch_decoder_cache_bytes": "decoder_cache_bytes",
    "decoder_chunk_size": "decoder_fetch_chunk_size",
    "feature_batch_size": "feature_batch_size",
    "logit_batch_size": "logit_batch_size",
    "nnsight_session_capacity": "session_capacity",
    "phase1_trace_batch_size_max": "phase1_source_batch_size",
    "phase3_compute_microbatch_max_rows": "logit_microbatch_size",
    "phase4_execution_batch_max_rows": "feature_microbatch_size",
    "phase4_frontier_refresh_interval": "frontier_refresh_stride",
}
_AXIS_CLASSES = {
    item.knob: item.classification.value for item in DEFAULT_KNOB_SENSITIVITIES
}


@dataclass(frozen=True)
class FidelityMetricBudget:
    minimum: float
    confidence: float


@dataclass(frozen=True)
class FidelityBudget:
    metrics: dict[str, FidelityMetricBudget]
    allowed_sensitive_axes: tuple[str, ...] = ()
    penalty: float | None = None


@dataclass(frozen=True)
class NormalizedFidelityPolicy:
    mode: str
    budget: FidelityBudget
    override_fields: tuple[str, ...]
    evidence_name: str | None
    evidence_version: str | None


def parse_fidelity_policy(scenario: Mapping[str, Any]) -> NormalizedFidelityPolicy:
    """Parse the project-facing four-mode fidelity contract and its budget."""

    raw_mode = str(scenario.get("governor_fidelity_mode", "exact"))
    mode = raw_mode
    if mode not in FIDELITY_MODES:
        known = ", ".join(sorted(FIDELITY_MODES))
        raise ValueError(f"unsupported governor_fidelity_mode {raw_mode!r}; expected {known}")

    raw_budget = scenario.get("governor_fidelity_budget") or {}
    if not isinstance(raw_budget, Mapping):
        raise ValueError("governor_fidelity_budget must be an object")
    unexpected = set(raw_budget) - {"metrics", "allowed_sensitive_axes", "penalty"}
    if unexpected:
        raise ValueError(
            "unsupported governor_fidelity_budget fields: "
            + ", ".join(sorted(str(key) for key in unexpected))
        )
    raw_metrics = raw_budget.get("metrics") or {}
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("governor_fidelity_budget.metrics must be an object")
    metrics: dict[str, FidelityMetricBudget] = {}
    for name, raw_metric in sorted(raw_metrics.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("fidelity metric names must be non-empty strings")
        if not isinstance(raw_metric, Mapping):
            raise ValueError(f"fidelity metric budget {name!r} must be an object")
        if set(raw_metric) != {"minimum", "confidence"}:
            raise ValueError(
                f"fidelity metric budget {name!r} requires minimum and confidence"
            )
        minimum = _finite_float(raw_metric["minimum"], f"{name}.minimum")
        confidence = _finite_float(raw_metric["confidence"], f"{name}.confidence")
        if not 0.0 <= minimum <= 1.0 or not 0.0 < confidence <= 1.0:
            raise ValueError(
                f"fidelity metric budget {name!r} must use minimum in [0, 1] "
                "and confidence in (0, 1]"
            )
        metrics[name] = FidelityMetricBudget(minimum, confidence)

    raw_axes = raw_budget.get("allowed_sensitive_axes") or ()
    if isinstance(raw_axes, (str, bytes)) or not isinstance(raw_axes, (list, tuple)):
        raise ValueError("governor_fidelity_budget.allowed_sensitive_axes must be a list")
    if any(not isinstance(axis, str) or not axis for axis in raw_axes):
        raise ValueError("allowed sensitive axes must be non-empty strings")
    allowed_sensitive_axes = tuple(sorted(set(raw_axes)))

    penalty = raw_budget.get("penalty")
    if penalty is not None:
        penalty = _finite_float(penalty, "governor_fidelity_budget.penalty")
        if penalty < 0:
            raise ValueError("governor_fidelity_budget.penalty must be nonnegative")
    if mode == "exact" and (metrics or allowed_sensitive_axes or penalty is not None):
        raise ValueError("exact fidelity accepts no metric budget or penalty")
    if mode == "bounded" and (not metrics or not allowed_sensitive_axes):
        raise ValueError(
            "bounded fidelity requires metric budgets and allowed sensitive axes"
        )
    if mode == "best_effort" and (penalty is None or not allowed_sensitive_axes):
        raise ValueError(
            "best_effort fidelity requires a penalty and allowed sensitive axes"
        )

    raw_fields = scenario.get("governor_fidelity_override_fields", ())
    if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, (list, tuple)):
        raise ValueError("governor_fidelity_override_fields must be a list of strings")
    if any(not isinstance(field, str) or not field for field in raw_fields):
        raise ValueError("governor_fidelity_override_fields must contain non-empty strings")
    override_fields = tuple(sorted(set(raw_fields)))
    if mode == "exact" and override_fields:
        raise ValueError("exact fidelity accepts no override fields")
    if mode == "bounded" and override_fields:
        raise ValueError("bounded fidelity does not accept semantic override fields")
    if mode == "research" and not override_fields:
        raise ValueError("research fidelity requires explicit override fields")

    evidence_name = scenario.get("governor_fidelity_evidence_name")
    evidence_version = scenario.get("governor_fidelity_evidence_version")
    if (evidence_name is None) != (evidence_version is None):
        raise ValueError("fidelity evidence name and version must be supplied together")
    return NormalizedFidelityPolicy(
        mode=mode,
        budget=FidelityBudget(
            metrics=metrics,
            allowed_sensitive_axes=allowed_sensitive_axes,
            penalty=penalty,
        ),
        override_fields=override_fields,
        evidence_name=evidence_name,
        evidence_version=evidence_version,
    )


def normalize_calibration_campaign(scenario: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = scenario.get("calibration_campaign")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("calibration_campaign must be an object")
    campaign_id = raw.get("campaign_id")
    reference = raw.get("reference")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("calibration_campaign.campaign_id must be a non-empty string")
    if not isinstance(reference, Mapping):
        raise ValueError("calibration_campaign.reference must be an object")
    kind = str(reference.get("kind") or "baseline_registry")
    if kind != "baseline_registry":
        raise ValueError("only baseline_registry campaign references are currently supported")
    registry_key = reference.get("registry_key")
    if not isinstance(registry_key, str) or not registry_key:
        raise ValueError("calibration campaign reference requires registry_key")
    split = str(raw.get("split") or "fit")
    role = str(
        raw.get("role")
        or (
            "reference"
            if scenario.get("calibration_case") in {"reference", "reference_repeat"}
            else "fit"
        )
    )
    if split not in {"fit", "heldout", "reference"}:
        raise ValueError("calibration_campaign.split must be fit, heldout, or reference")
    if role not in {"fit", "heldout", "reference"}:
        raise ValueError("calibration_campaign.role must be fit, heldout, or reference")
    fixed_controls = raw.get("fixed_controls") or {}
    if not isinstance(fixed_controls, Mapping):
        raise ValueError("calibration_campaign.fixed_controls must be an object")
    return {
        "campaign_id": campaign_id,
        "case": scenario.get("calibration_case"),
        "factor": scenario.get("calibration_factor"),
        "split": split,
        "role": role,
        "fit_group": raw.get("fit_group"),
        "holdout_group": raw.get("holdout_group"),
        "fixed_controls": dict(sorted(fixed_controls.items())),
        "reference": {"kind": kind, "registry_key": registry_key},
    }


def apply_campaign_reference_to_baseline_check(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the existing static-baseline comparator for campaign references."""

    normalized = dict(scenario)
    campaign = normalize_calibration_campaign(scenario)
    if campaign is None:
        return normalized
    reference_key = campaign["reference"]["registry_key"]
    raw_check = scenario.get("baseline_check")
    if raw_check is None:
        normalized["baseline_check"] = {
            "enabled": True,
            "mode": "metrics",
            "registry_key": reference_key,
            "baseline_required": True,
        }
    elif not isinstance(raw_check, Mapping):
        raise ValueError("scenario baseline_check must be an object when present")
    elif raw_check.get("registry_key") != reference_key:
        raise ValueError(
            "calibration campaign reference and baseline_check registry keys differ"
        )
    return normalized


def build_calibration_observation(
    *,
    scenario_root: Path,
    scenario: Mapping[str, Any],
    result: Mapping[str, Any],
    baseline_entry: Mapping[str, Any] | None,
    resource_sidecar_paths: Iterable[Path] = (),
    environ: Mapping[str, str] | None = None,
    scheduler_accounting: Mapping[str, Any] | None = None,
    finalization: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Join one persisted scenario into a stable calibration observation."""

    campaign = normalize_calibration_campaign(scenario)
    if campaign is None:
        return None
    fidelity = parse_fidelity_policy(scenario)
    artifacts_dir = Path(str(result.get("output_dir") or scenario_root / "artifacts"))
    completions = _completion_manifests(artifacts_dir)
    telemetry, planning = _telemetry_summary(artifacts_dir)
    sidecars = _resource_sidecar_summary(resource_sidecar_paths)
    comparison = _comparison_summary(scenario_root, result)
    outcome = _outcome(result)
    missing = _missing_measurements(completions, telemetry, sidecars, comparison)
    environment = {} if environ is None else environ

    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_OBSERVATION_SCHEMA_VERSION,
        "observation_id": "pending",
        "campaign": campaign,
        "scope": _scope(scenario),
        "configuration": {
            key: scenario.get(key) for key in SCENARIO_KNOB_KEYS if key in scenario
        },
        "decision_axes": [
            {
                "scenario_field": field,
                "axis_id": axis_id,
                "sensitivity": _AXIS_CLASSES[axis_id],
                "requested_value": scenario[field],
            }
            for field, axis_id in sorted(_SCENARIO_AXIS_IDS.items())
            if field in scenario and axis_id in _AXIS_CLASSES
        ],
        "fidelity": {
            "mode": fidelity.mode,
            "budget": {
                "metrics": {
                    name: asdict(metric)
                    for name, metric in fidelity.budget.metrics.items()
                },
                "allowed_sensitive_axes": list(
                    fidelity.budget.allowed_sensitive_axes
                ),
                "penalty": fidelity.budget.penalty,
            },
            "override_fields": list(fidelity.override_fields),
            "evidence": {
                "name": fidelity.evidence_name,
                "version": fidelity.evidence_version,
            },
            "comparison": comparison,
        },
        "outcome": outcome,
        "runtime": {
            "walltime_seconds": _number(result.get("duration_seconds")),
            "profiling_summary": result.get("profiling_summary") or {},
            "completion_timing": _completion_timing(completions),
            "phase_telemetry": telemetry,
            "planning": planning,
        },
        "workload": _workload(result),
        "resources": {
            "completion_snapshots": [
                row.get("resource_snapshot")
                for row in completions
                if isinstance(row.get("resource_snapshot"), Mapping)
            ],
            "step_peaks": _step_resource_peaks(completions),
            "gpu_sidecar": sidecars,
        },
        "uncertainty": {
            "support": "observed" if outcome["scientific_value"] else "none",
            "censoring": outcome["censoring"],
            "missing_measurements": missing,
            "comparison_complete": comparison.get("comparison_complete"),
        },
        "provenance": {
            "scenario_json": str(scenario_root / "scenario.json"),
            "result_json": str(scenario_root / "result.json"),
            "artifacts_dir": str(artifacts_dir),
            "baseline_compare_json": comparison.get("comparison_json"),
            "reference": _reference_provenance(baseline_entry),
            "run_metadata": result.get("run_metadata") or scenario.get("run_metadata") or {},
            "semantic_fingerprints": _fingerprints(completions, "semantic_fingerprint"),
            "execution_fingerprints": _fingerprints(completions, "execution_fingerprint"),
            "slurm": {
                key: environment.get(key)
                for key in (
                    "SLURM_JOB_ID",
                    "SLURM_ARRAY_JOB_ID",
                    "SLURM_ARRAY_TASK_ID",
                    "SLURM_JOB_PARTITION",
                    "SLURM_JOB_QOS",
                )
                if environment.get(key) is not None
            },
            "scheduler_accounting": dict(scheduler_accounting or {}),
            "finalization": dict(finalization or {}),
        },
    }
    payload["observation_fingerprint"] = _observation_fingerprint(payload)
    payload["observation_id"] = f"calobs-{payload['observation_fingerprint'][:24]}"
    return payload


def write_calibration_observation(scenario_root: Path, payload: dict[str, Any]) -> Path:
    path = scenario_root / "calibration_observation.json"
    write_json(path, payload)
    return path


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return value


def _completion_manifests(artifacts_dir: Path) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted(artifacts_dir.glob("prompt_*/completion_*/completion.json"))
    ]


def _telemetry_summary(artifacts_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = sorted(artifacts_dir.glob("prompt_*/completion_*/telemetry.live.jsonl"))
    stream = "telemetry.live.jsonl"
    if not paths:
        paths = sorted(artifacts_dir.glob("prompt_*/completion_*/telemetry.jsonl"))
        stream = "telemetry.jsonl"
    events: dict[str, dict[str, Any]] = {}
    total = 0
    invalid = 0
    planning: list[dict[str, Any]] = []
    for path in paths:
        try:
            records = iter_jsonl(path)
            for record in records:
                total += 1
                if not isinstance(record, Mapping):
                    invalid += 1
                    continue
                name = str(record.get("name") or record.get("event") or record.get("kind") or "unknown")
                raw_attrs = record.get("attrs")
                attrs: Mapping[str, Any] = (
                    raw_attrs if isinstance(raw_attrs, Mapping) else record
                )
                if name.startswith("planning.") and name not in {
                    "planning.observation",
                    "planning.resource_actual",
                    "planning.terminal_cleanup",
                }:
                    vector_keys = (
                        "row_store_policy", "row_store_bytes", "spill_target",
                        "session_capacity", "phase1_source_batch_size",
                        "source_microbatch_size", "feature_microbatch_size",
                        "logit_microbatch_size", "decoder_cache_bytes",
                        "decoder_fetch_chunk_size", "replay_window", "prefetch_depth",
                        "replay_tile_cache_bytes",
                    )
                    planning.append(
                        {
                            "epoch": name.removeprefix("planning."),
                            "selected_vector": {
                                key: attrs[key] for key in vector_keys if key in attrs
                            },
                            "candidate_count": attrs.get("candidate_count"),
                            "admissible_candidate_count": attrs.get(
                                "admissible_candidate_count"
                            ),
                            "selected_objective": attrs.get("selected_objective"),
                            "execution_fingerprint": attrs.get("execution_fingerprint"),
                            "semantic_fingerprint": attrs.get("semantic_fingerprint"),
                        }
                    )
                summary = events.setdefault(name, {"count": 0, "numeric": {}})
                summary["count"] += 1
                for key, value in attrs.items():
                    numeric = _number(value)
                    if numeric is None:
                        continue
                    metric = summary["numeric"].setdefault(
                        str(key), {"count": 0, "min": numeric, "max": numeric, "latest": numeric}
                    )
                    metric["count"] += 1
                    metric["min"] = min(metric["min"], numeric)
                    metric["max"] = max(metric["max"], numeric)
                    metric["latest"] = numeric
        except (OSError, ValueError, json.JSONDecodeError):
            invalid += 1
    return {
        "source_stream": stream if paths else None,
        "file_count": len(paths),
        "event_count": total,
        "invalid_file_count": invalid,
        "events": events,
    }, planning


def _resource_sidecar_summary(paths: Iterable[Path]) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    source_paths: list[str] = []
    invalid_rows = 0
    for path in sorted(set(paths)):
        if not path.is_file() or path.suffix != ".gpulog":
            continue
        source_paths.append(str(path))
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
            for row in rows:
                try:
                    samples.append(
                        {
                            "timestamp": row["Timestamp"],
                            "gpu_utilization_percent": float(row["GPU_utilization(%)"]),
                            "gpu_vram_fraction": float(row["GPU_VRAM(%)"]),
                            "gpu_vram_mib": float(row["GPU_VRAM"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    invalid_rows += 1
    return {
        "format": "chpc_gpu_monitor_v1" if source_paths else None,
        "paths": source_paths,
        "sample_count": len(samples),
        "invalid_row_count": invalid_rows,
        "gpu_utilization_percent": _series_summary(samples, "gpu_utilization_percent"),
        "gpu_vram_fraction": _series_summary(samples, "gpu_vram_fraction"),
        "gpu_vram_mib": _series_summary(samples, "gpu_vram_mib"),
        "samples": samples,
    }


def _series_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows]
    if not values:
        return None
    return {"minimum": min(values), "maximum": max(values), "mean": sum(values) / len(values)}


def _comparison_summary(scenario_root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    status = result.get("baseline_check") or {}
    path_value = status.get("comparison_json")
    path = Path(str(path_value)) if path_value else scenario_root / "baseline_compare.json"
    summary = read_json(path) if path.is_file() else {}
    return {
        "status": status.get("status"),
        "passed": status.get("passed"),
        "reference_registry_key": status.get("registry_key"),
        "comparison_json": str(path) if path.is_file() else None,
        **{key: summary.get(key, status.get(key)) for key in COMPARISON_SUMMARY_KEYS},
    }


def _outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "unknown")
    if status == "success":
        censoring = "none"
        scientific_value = True
    elif status in {"oom", "refused"}:
        censoring = "feasibility"
        scientific_value = True
    elif status == "timeout":
        censoring = "runtime_lower_bound"
        scientific_value = True
    else:
        censoring = "infrastructure_or_unknown"
        scientific_value = False
    return {
        "status": status,
        "returncode": result.get("returncode"),
        "censoring": censoring,
        "scientific_value": scientific_value,
        "failure_reasons": (result.get("baseline_check") or {}).get("failure_reasons", []),
    }


def _scope(scenario: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "name", "stage", "cluster", "resource_profile", "method", "model_name",
        "model_id", "transcoder_architecture", "transcoder_provider_family",
        "transcoder_set", "fixture_name",
        "fixture_kind", "exact_trace_internal_dtype", "governor_profile_name",
    )
    return {key: scenario.get(key) for key in keys if key in scenario}


def _workload(result: Mapping[str, Any]) -> dict[str, float | int]:
    summary = result.get("artifact_summary")
    if not isinstance(summary, Mapping):
        return {}
    keys = (
        "prompt_token_count",
        "initial_input_token_count",
        "generated_token_count",
        "n_steps_traced",
        "max_active_features",
    )
    return {
        key: value
        for key in keys
        if (value := _number(summary.get(key))) is not None
    }


def _completion_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": row.get("prompt_id"),
            "completion_id": row.get("completion_id"),
            "duration_seconds": row.get("duration_seconds"),
            "timing_summary": row.get("timing_summary") or {},
        }
        for row in rows
    ]


def _step_resource_peaks(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    fields = (
        "rss_gib", "cuda_allocated_gib", "cuda_reserved_gib",
        "cuda_peak_allocated_gib", "cuda_peak_reserved_gib",
    )
    values: dict[str, list[float]] = {field: [] for field in fields}
    for completion in rows:
        for step in completion.get("steps", []):
            snapshot = step.get("resource_snapshot") or {}
            for field in fields:
                numeric = _number(snapshot.get(field))
                if numeric is not None:
                    values[field].append(float(numeric))
    return {field: max(observed) if observed else None for field, observed in values.items()}


def _fingerprints(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(row[key]) for row in rows if row.get(key) is not None})


def _reference_provenance(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    keys = (
        "registry_key", "scenario_name", "scenario_root", "artifacts_dir",
        "result_json", "run_id", "run_name", "comparison_contract", "prompt_identity",
    )
    return {key: entry.get(key) for key in keys if key in entry}


def _missing_measurements(
    completions: list[dict[str, Any]],
    telemetry: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    missing = []
    if not completions:
        missing.append("completion_manifests")
    if not telemetry.get("event_count"):
        missing.append("phase_telemetry")
    if not sidecars.get("sample_count"):
        missing.append("gpu_sidecar_samples")
    if comparison.get("comparison_complete") is not True:
        missing.append("complete_reference_comparison")
    return missing


def _observation_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical["observation_id"] = None
    canonical["observation_fingerprint"] = None
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return digest
