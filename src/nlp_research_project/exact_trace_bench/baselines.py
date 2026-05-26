from __future__ import annotations

import json
import math
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT
from .graph_compare import compare_artifact_dirs
from .io_utils import read_json, write_csv, write_json


BASELINE_DISABLED = {
    "enabled": False,
    "mode": "off",
    "status": "disabled",
    "passed": None,
    "failure_reasons": [],
}

COMPARISON_SUMMARY_KEYS = (
    "shared_completion_count",
    "left_only_completion_count",
    "right_only_completion_count",
    "overall_mean_feature_jaccard",
    "overall_mean_edge_jaccard",
    "overall_mean_weighted_edge_jaccard",
)

SCENARIO_IDENTITY_KEYS = (
    "name",
    "stage",
    "cluster",
    "resource_profile",
    "fixture_name",
    "fixture_kind",
    "gsm8k_indices",
    "prepared_prompt_file",
    "prepared_prompt_meta_file",
    "method",
)

SCENARIO_KNOB_KEYS = (
    "exact_trace_internal_dtype",
    "attribution_batch_size",
    "feature_batch_size",
    "logit_batch_size",
    "decoder_chunk_size",
    "cross_batch_decoder_cache_bytes",
    "phase1_trace_batch_policy",
    "phase1_trace_batch_size_max",
    "chunked_feature_replay_window",
    "error_vector_prefetch_lookahead",
    "stage_encoder_vecs_on_cpu",
    "stage_error_vectors_on_cpu",
    "row_subchunk_size",
    "plan_feature_batch_size",
    "feature_batch_size_max",
    "feature_batch_target_reserved_fraction",
    "feature_batch_min_free_fraction",
    "feature_batch_probe_batches",
    "phase4_refresh_policy",
    "phase4_refresh_interval_multiplier",
    "phase4_ranker",
    "row_store_cache_control",
    "exact_encoder_residency",
    "phase4_scheduler_mode",
    "phase4_scheduler_telemetry_detail",
    "phase4_refresh_optimization",
    "phase4_row_executor",
)

METRICS_PREFERRED_HEADERS = (
    "name",
    "stage",
    "cluster",
    "resource_profile",
    "fixture_name",
    "fixture_kind",
    "gsm8k_indices",
    "method",
    "status",
    "returncode",
    "duration_seconds",
    "run_id",
    "run_name",
    "output_dir",
    "baseline_enabled",
    "baseline_mode",
    "baseline_registry_key",
    "baseline_status",
    "baseline_passed",
    "baseline_failure_reasons",
    *COMPARISON_SUMMARY_KEYS,
)


def load_baseline_registry(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Baseline registry must contain an object 'entries': {path}")
    return {str(key): value for key, value in entries.items()}


def normalize_baseline_check(scenario: dict[str, Any]) -> dict[str, Any]:
    raw = scenario.get("baseline_check")
    if not raw:
        return dict(BASELINE_DISABLED)
    if not isinstance(raw, dict):
        raise ValueError("scenario baseline_check must be an object when present")
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return dict(BASELINE_DISABLED)
    mode = str(raw.get("mode") or "metrics")
    if mode not in {"metrics", "gate"}:
        raise ValueError(f"Unsupported baseline_check mode: {mode!r}")
    return {
        "enabled": True,
        "mode": mode,
        "registry_key": raw.get("registry_key"),
        "baseline_required": bool(raw.get("baseline_required", True)),
        "thresholds": raw.get("thresholds") or {},
        "status": "pending",
        "passed": None,
        "failure_reasons": [],
    }


def _append_reason(status: dict[str, Any], reason: str) -> None:
    reasons = status.setdefault("failure_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def resolve_baseline_entry(
    baseline_check: dict[str, Any],
    *,
    registry: dict[str, dict[str, Any]] | None,
    registry_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not baseline_check.get("enabled"):
        return dict(baseline_check), None

    status = dict(baseline_check)
    if registry_path is not None:
        status["registry_path"] = str(registry_path)
    key = status.get("registry_key")
    if not key:
        status["status"] = "baseline_missing"
        _append_reason(status, "baseline_check.registry_key is required")
        return status, None
    if registry is None:
        status["status"] = "baseline_missing"
        _append_reason(status, "baseline registry was not provided")
        return status, None
    entry = registry.get(str(key))
    if not isinstance(entry, dict):
        status["status"] = "baseline_missing"
        _append_reason(status, f"baseline registry key not found: {key}")
        return status, None
    return status, entry


def validate_baseline_entry(
    entry: dict[str, Any],
    *,
    status: dict[str, Any],
) -> dict[str, Any]:
    artifacts_dir = entry.get("artifacts_dir")
    result_json = entry.get("result_json")
    if artifacts_dir is None:
        _append_reason(status, "baseline entry missing artifacts_dir")
    else:
        artifacts_path = Path(str(artifacts_dir))
        status["reference_artifacts"] = str(artifacts_path)
        if not artifacts_path.is_dir():
            _append_reason(
                status, f"baseline artifacts_dir does not exist: {artifacts_path}"
            )

    if result_json is None:
        _append_reason(status, "baseline entry missing result_json")
    else:
        result_path = Path(str(result_json))
        status["reference_result_json"] = str(result_path)
        if not result_path.is_file():
            _append_reason(
                status, f"baseline result_json does not exist: {result_path}"
            )
        else:
            try:
                result_payload = read_json(result_path)
            except Exception as exc:  # noqa: BLE001 - provenance diagnostics
                _append_reason(status, f"baseline result_json unreadable: {exc}")
            else:
                expected = str(entry.get("expected_status") or "success")
                actual = str(result_payload.get("status"))
                status["reference_result_status"] = actual
                if actual != expected:
                    _append_reason(
                        status,
                        f"baseline result status {actual!r} != expected {expected!r}",
                    )

    if status.get("failure_reasons"):
        status["status"] = "baseline_invalid"
        status["passed"] = False
    else:
        status["status"] = "baseline_valid"
    return status


def flatten_comparison_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in COMPARISON_SUMMARY_KEYS}


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def evaluate_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any] | None,
) -> tuple[bool | None, list[str]]:
    if not thresholds:
        return None, []
    reasons: list[str] = []
    for threshold_key, threshold_value in thresholds.items():
        if threshold_key.endswith("_min"):
            metric_key = threshold_key[: -len("_min")]
            comparator = "min"
        elif threshold_key.endswith("_max"):
            metric_key = threshold_key[: -len("_max")]
            comparator = "max"
        else:
            reasons.append(f"unsupported threshold key: {threshold_key}")
            continue
        metric_value = _as_finite_float(metrics.get(metric_key))
        threshold_float = _as_finite_float(threshold_value)
        if metric_value is None or threshold_float is None:
            reasons.append(f"non-numeric threshold comparison for {threshold_key}")
            continue
        if comparator == "min" and metric_value < threshold_float:
            reasons.append(f"{metric_key}={metric_value} < min {threshold_float}")
        if comparator == "max" and metric_value > threshold_float:
            reasons.append(f"{metric_key}={metric_value} > max {threshold_float}")
    return not reasons, reasons


def run_baseline_comparison(
    *,
    scenario_root: Path,
    current_artifacts: Path,
    baseline_check: dict[str, Any],
    baseline_entry: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = dict(baseline_check)
    if not status.get("enabled"):
        return status, {}
    if baseline_entry is None:
        status.setdefault("status", "baseline_missing")
        status["passed"] = False
        return status, {}

    status = validate_baseline_entry(baseline_entry, status=status)
    if status.get("status") == "baseline_invalid":
        return status, {}

    reference_artifacts = Path(str(baseline_entry["artifacts_dir"]))
    try:
        summary = compare_artifact_dirs(reference_artifacts, current_artifacts)
    except Exception as exc:  # noqa: BLE001 - preserve trace success, report validation
        status["status"] = "compare_error"
        status["passed"] = False
        _append_reason(status, f"baseline comparison failed: {exc}")
        return status, {}

    comparison_path = scenario_root / "baseline_compare.json"
    write_json(comparison_path, summary)
    metrics = flatten_comparison_summary(summary)
    status["comparison_json"] = str(comparison_path)
    status.update(metrics)

    if status.get("mode") == "gate":
        passed, reasons = evaluate_thresholds(metrics, status.get("thresholds"))
        status["passed"] = bool(passed)
        status["status"] = "gate_pass" if passed else "gate_fail"
        for reason in reasons:
            _append_reason(status, reason)
    else:
        status["passed"] = None
        status["status"] = "compared"
    return status, metrics


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return (
            str(value) if isinstance(value, set) else json.dumps(value, sort_keys=True)
        )
    return value


def build_scenario_metrics_row(
    *,
    scenario: dict[str, Any],
    result: dict[str, Any],
    baseline_status: dict[str, Any] | None = None,
    comparison_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_status = baseline_status or dict(BASELINE_DISABLED)
    comparison_metrics = comparison_metrics or {}
    row: dict[str, Any] = {}
    for key in SCENARIO_IDENTITY_KEYS:
        if key in scenario:
            row[key] = _csv_value(scenario.get(key))
    for key in SCENARIO_KNOB_KEYS:
        if key in scenario:
            row[key] = _csv_value(scenario.get(key))

    for key in (
        "status",
        "returncode",
        "duration_seconds",
        "run_id",
        "launch_id",
        "run_name",
        "run_description",
        "run_goal",
        "output_dir",
        "log_path",
    ):
        if key in result:
            row[key] = _csv_value(result.get(key))

    for key, value in (result.get("profiling_summary") or {}).items():
        row[f"profiling_{key}"] = _csv_value(value)
    for key, value in (result.get("artifact_summary") or {}).items():
        row[f"artifact_{key}"] = _csv_value(value)

    row.update(
        {
            "baseline_enabled": bool(baseline_status.get("enabled", False)),
            "baseline_mode": baseline_status.get("mode"),
            "baseline_registry_key": baseline_status.get("registry_key"),
            "baseline_registry_path": baseline_status.get("registry_path"),
            "baseline_status": baseline_status.get("status"),
            "baseline_passed": baseline_status.get("passed"),
            "baseline_reference_artifacts": baseline_status.get("reference_artifacts"),
            "baseline_comparison_json": baseline_status.get("comparison_json"),
            "baseline_failure_reasons": _csv_value(
                baseline_status.get("failure_reasons") or []
            ),
        }
    )
    for key in COMPARISON_SUMMARY_KEYS:
        row[key] = comparison_metrics.get(key, baseline_status.get(key))
    return row


def write_scenario_metrics(
    scenario_root: Path,
    row: dict[str, Any],
    *,
    baseline_status: dict[str, Any] | None = None,
) -> None:
    payload = {"metrics": row}
    if baseline_status is not None:
        payload["baseline_check"] = baseline_status
    write_json(scenario_root / "scenario_metrics.json", payload)
    write_csv(
        scenario_root / "scenario_metrics.csv",
        [row],
        preferred_headers=METRICS_PREFERRED_HEADERS,
    )


def _safe_git_output(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _workspace_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "branch": _safe_git_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path
        ),
        "commit": _safe_git_output(["git", "rev-parse", "--short", "HEAD"], cwd=path),
        "dirty_files": (
            _safe_git_output(["git", "status", "--short"], cwd=path) or ""
        ).splitlines(),
    }


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_tier_from_root_or_stage(
    scenario_root: Path,
    scenario: dict[str, Any],
) -> str | None:
    for part in scenario_root.parts:
        if part in {"fast", "anomaly", "long_eval"}:
            return part
    stage = str(scenario.get("stage") or "")
    for tier in ("fast", "anomaly", "long_eval"):
        if tier in stage:
            return tier
    return None


def _profile_for_contract(contract: dict[str, Any]) -> str:
    dtype = str(contract.get("exact_trace_internal_dtype") or "unknown")
    return f"{dtype}_default"


def _comparison_contract(scenario: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "exact_trace_internal_dtype",
        "temperature",
        "completions",
        "max_steps",
        "max_feature_nodes",
        "max_edges",
        "max_n_logits",
        "desired_logit_prob",
    )
    return {key: scenario.get(key) for key in keys if key in scenario}


def _entry_from_scenario_root(scenario_root: Path) -> dict[str, Any]:
    scenario = read_json(scenario_root / "scenario.json")
    result = read_json(scenario_root / "result.json")
    prompt_file_text = scenario.get("prepared_prompt_file")
    prompt_file = Path(str(prompt_file_text)) if prompt_file_text else None
    return {
        "cluster": scenario.get("cluster"),
        "tier": _infer_tier_from_root_or_stage(scenario_root, scenario),
        "fixture_name": scenario.get("fixture_name"),
        "fixture_kind": scenario.get("fixture_kind"),
        "gsm8k_indices": scenario.get("gsm8k_indices"),
        "run_id": result.get("run_id")
        or scenario.get("run_metadata", {}).get("run_id"),
        "run_name": result.get("run_name")
        or scenario.get("run_metadata", {}).get("run_name"),
        "scenario_name": scenario.get("name") or scenario_root.name,
        "scenario_root": str(scenario_root),
        "artifacts_dir": str(scenario_root / "artifacts"),
        "result_json": str(scenario_root / "result.json"),
        "expected_status": "success",
        "result_status": result.get("status"),
        "resource_profile": scenario.get("resource_profile"),
        "wave": scenario.get("wave"),
        "wave0_role": scenario.get("wave0_role"),
        "wave0_repeat_index": scenario.get("wave0_repeat_index"),
        "decoder_chunk_size": scenario.get("decoder_chunk_size"),
        "cross_batch_decoder_cache_bytes": scenario.get(
            "cross_batch_decoder_cache_bytes"
        ),
        "attribution_batch_size": scenario.get("attribution_batch_size"),
        "feature_batch_size": scenario.get("feature_batch_size"),
        "logit_batch_size": scenario.get("logit_batch_size"),
        "comparison_contract": _comparison_contract(scenario),
        "prompt_identity": {
            "prepared_prompt_file": str(prompt_file) if prompt_file else None,
            "prepared_prompt_sha256": _sha256_file(prompt_file)
            if prompt_file is not None
            else None,
            "prepared_prompt_meta_file": scenario.get("prepared_prompt_meta_file"),
        },
    }


def _keys_for_entry(entry: dict[str, Any]) -> list[str]:
    fixture_name = entry.get("fixture_name")
    cluster = entry.get("cluster")
    tier = entry.get("tier")
    if not fixture_name or not cluster or not tier:
        return []
    profile = _profile_for_contract(entry.get("comparison_contract", {}))
    base = f"wave0/{fixture_name}/{cluster}/{tier}/{profile}"
    repeat_index = entry.get("wave0_repeat_index")
    if repeat_index is None:
        return [base]
    keys = [f"{base}_r{repeat_index}"]
    if int(repeat_index) == 1:
        keys.insert(0, base)
    return keys


def build_baseline_registry_from_run_roots(
    run_roots: list[Path],
    *,
    registry_id: str,
    project_root: Path = REPO_ROOT,
    library_root: Path | None = None,
) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    invalid_roots: list[str] = []
    duplicate_keys: list[str] = []
    for run_root in run_roots:
        if not run_root.is_dir():
            invalid_roots.append(str(run_root))
            continue
        for scenario_root in sorted(
            path for path in run_root.iterdir() if path.is_dir()
        ):
            if (
                not (scenario_root / "scenario.json").is_file()
                or not (scenario_root / "result.json").is_file()
            ):
                continue
            entry = _entry_from_scenario_root(scenario_root)
            if entry.get("result_status") != "success":
                continue
            for key in _keys_for_entry(entry):
                if key in entries:
                    duplicate_keys.append(key)
                    continue
                entries[key] = dict(entry, registry_key=key)
    library_root = library_root or (project_root.parent / "circuit-tracer_chunked")
    return {
        "schema_version": 1,
        "registry_id": registry_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_from": "nlp_research_project.exact_trace_bench build-baseline-registry",
        "source_roots": [str(root) for root in run_roots],
        "invalid_roots": invalid_roots,
        "duplicate_keys": duplicate_keys,
        "project_repo": _workspace_state(project_root),
        "sibling_library_repo": _workspace_state(library_root)
        if library_root.exists()
        else {"path": str(library_root), "missing": True},
        "entries": dict(sorted(entries.items())),
    }


def wave0_run_roots(
    *,
    run_id: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    clusters: tuple[str, ...] = ("ascend", "cardinal"),
    tiers: tuple[str, ...] = ("fast", "anomaly", "long_eval"),
) -> list[Path]:
    return [
        scratch_root / cluster / tier / run_id for cluster in clusters for tier in tiers
    ]
