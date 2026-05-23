from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nlp_research_project.exact_trace_bench.baselines import (  # noqa: E402
    BASELINE_DISABLED,
    build_scenario_metrics_row,
    load_baseline_registry,
    normalize_baseline_check,
    resolve_baseline_entry,
    run_baseline_comparison,
    validate_baseline_entry,
    write_scenario_metrics,
)
from nlp_research_project.exact_trace_bench.io_utils import write_csv  # noqa: E402


DEFAULT_SCENARIOS = (
    Path(__file__).with_name("generated") / "sparsification_calibration_scenarios.json"
)
DEFAULT_OUTPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/sparsification_experiment")

PHASE_DURATION_RE = re.compile(r"completed in (?P<seconds>\d+(?:\.\d+)?)s")
PHASE4_BATCH_RE = re.compile(
    r"Phase 4 batch (?P<batch_idx>\d+)/(?P<total_batches>\d+) in (?P<seconds>\d+(?:\.\d+)?)s"
)
MEMORY_RE = re.compile(
    r"rss=(?P<rss>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"(?:rss_current=(?P<rss_current>n/a|\d+(?:\.\d+)?(?: GiB)?), )?"
    r"cuda_alloc=(?P<cuda_alloc>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_reserved=(?P<cuda_reserved>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_peak_alloc=(?P<cuda_peak_alloc>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_peak_reserved=(?P<cuda_peak_reserved>n/a|\d+(?:\.\d+)?(?: GiB)?)"
)


def apply_runtime_overrides(
    scenario: dict[str, Any],
    *,
    cross_batch_decoder_cache_bytes_override: int | None = None,
) -> dict[str, Any]:
    effective_scenario = dict(scenario)
    if (
        effective_scenario["method"] != "old_patch"
        and cross_batch_decoder_cache_bytes_override is not None
    ):
        effective_scenario["cross_batch_decoder_cache_bytes"] = (
            cross_batch_decoder_cache_bytes_override
        )
    return effective_scenario


def _normalize_run_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _has_run_metadata(run_metadata: dict[str, str | None]) -> bool:
    return any(value is not None for value in run_metadata.values())


def _assert_fresh_scenario_root(scenario_root: Path) -> None:
    collision_paths = [
        scenario_root / "scenario.json",
        scenario_root / "result.json",
        scenario_root / "run.log",
        scenario_root / "artifacts",
    ]
    existing = [path for path in collision_paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Scenario output directory already contains artifacts; refusing to reuse it: "
            f"{scenario_root} ({joined})"
        )


def _build_prompt_source_args(scenario: dict[str, Any]) -> list[str]:
    prepared_prompt_file = scenario.get("prepared_prompt_file")
    if prepared_prompt_file is not None:
        args = ["--prepared-prompt-file", str(prepared_prompt_file)]
        prepared_prompt_meta_file = scenario.get("prepared_prompt_meta_file")
        if prepared_prompt_meta_file is not None:
            args.extend(["--prepared-prompt-meta-file", str(prepared_prompt_meta_file)])
        return args

    gsm8k_indices = scenario.get("gsm8k_indices")
    if not gsm8k_indices:
        raise ValueError(
            "Scenario must define either gsm8k_indices or prepared_prompt_file"
        )

    return [
        "--prompts",
        str(len(gsm8k_indices)),
        "--gsm8k-indices",
        ",".join(str(i) for i in gsm8k_indices),
    ]


def _parse_optional_gib(value: str) -> float | None:
    if value == "n/a":
        return None
    normalized = value.removesuffix(" GiB")
    return float(normalized)


def _format_optional_bool_arg(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return "true"
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return "false"
        raise ValueError(f"Unsupported boolean string override: {value!r}")
    if isinstance(value, int):
        if value in {0, 1}:
            return str(bool(value)).lower()
    raise ValueError(f"Unsupported boolean override type/value: {value!r}")


def _extract_benchmark_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}

    phase4_batch_durations: list[float] = []
    phase4_total_batches = None
    peak_rss_gib = None
    peak_cuda_allocated_gib = None
    peak_cuda_reserved_gib = None
    peak_cuda_peak_allocated_gib = None
    peak_cuda_peak_reserved_gib = None
    phase3_duration_seconds = None
    phase4_duration_seconds = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "logit attribution(s) completed in" in line:
            match = PHASE_DURATION_RE.search(line)
            if match:
                phase3_duration_seconds = float(match.group("seconds"))
        elif "Feature attributions completed in" in line:
            match = PHASE_DURATION_RE.search(line)
            if match:
                phase4_duration_seconds = float(match.group("seconds"))

        batch_match = PHASE4_BATCH_RE.search(line)
        if batch_match:
            phase4_batch_durations.append(float(batch_match.group("seconds")))
            phase4_total_batches = int(batch_match.group("total_batches"))

        memory_match = MEMORY_RE.search(line)
        if memory_match:
            rss_gib = _parse_optional_gib(memory_match.group("rss"))
            cuda_alloc_gib = _parse_optional_gib(memory_match.group("cuda_alloc"))
            cuda_reserved_gib = _parse_optional_gib(memory_match.group("cuda_reserved"))
            cuda_peak_alloc_gib = _parse_optional_gib(
                memory_match.group("cuda_peak_alloc")
            )
            cuda_peak_reserved_gib = _parse_optional_gib(
                memory_match.group("cuda_peak_reserved")
            )

            if rss_gib is not None:
                peak_rss_gib = max(peak_rss_gib or rss_gib, rss_gib)
            if cuda_alloc_gib is not None:
                peak_cuda_allocated_gib = max(
                    peak_cuda_allocated_gib or cuda_alloc_gib,
                    cuda_alloc_gib,
                )
            if cuda_reserved_gib is not None:
                peak_cuda_reserved_gib = max(
                    peak_cuda_reserved_gib or cuda_reserved_gib,
                    cuda_reserved_gib,
                )
            if cuda_peak_alloc_gib is not None:
                peak_cuda_peak_allocated_gib = max(
                    peak_cuda_peak_allocated_gib or cuda_peak_alloc_gib,
                    cuda_peak_alloc_gib,
                )
            if cuda_peak_reserved_gib is not None:
                peak_cuda_peak_reserved_gib = max(
                    peak_cuda_peak_reserved_gib or cuda_peak_reserved_gib,
                    cuda_peak_reserved_gib,
                )

    phase4_avg_batch_seconds = None
    phase4_projected_total_seconds = None
    if phase4_batch_durations:
        phase4_avg_batch_seconds = sum(phase4_batch_durations) / len(
            phase4_batch_durations
        )
        if phase4_total_batches is not None:
            phase4_projected_total_seconds = (
                phase4_avg_batch_seconds * phase4_total_batches
            )

    return {
        "phase3_duration_seconds": phase3_duration_seconds,
        "phase4_duration_seconds": phase4_duration_seconds,
        "phase4_avg_batch_seconds": phase4_avg_batch_seconds,
        "phase4_projected_total_seconds": phase4_projected_total_seconds,
        "phase4_batches_observed": len(phase4_batch_durations),
        "phase4_total_batches": phase4_total_batches,
        "peak_rss_gib": peak_rss_gib,
        "peak_cuda_allocated_gib": peak_cuda_allocated_gib,
        "peak_cuda_reserved_gib": peak_cuda_reserved_gib,
        "peak_cuda_peak_allocated_gib": peak_cuda_peak_allocated_gib,
        "peak_cuda_peak_reserved_gib": peak_cuda_peak_reserved_gib,
    }


def _summarize_artifacts(run_output_dir: Path) -> dict[str, Any]:
    prompt_meta_files = sorted(run_output_dir.glob("prompt_*/prompt_meta.json"))
    completion_files = sorted(
        run_output_dir.glob("prompt_*/completion_*/completion.json")
    )
    if not prompt_meta_files and not completion_files:
        return {}

    prompt_metas = [json.loads(path.read_text()) for path in prompt_meta_files]
    completion_manifests = [json.loads(path.read_text()) for path in completion_files]

    active_feature_counts = [
        step.get("n_active_features")
        for manifest in completion_manifests
        for step in manifest.get("steps", [])
        if step.get("n_active_features") is not None
    ]
    cache_hits = [
        step.get("transcoder_diagnostics", {}).get("decoder_cache_hit_count")
        for manifest in completion_manifests
        for step in manifest.get("steps", [])
        if step.get("transcoder_diagnostics")
    ]
    cache_misses = [
        step.get("transcoder_diagnostics", {}).get("decoder_cache_miss_count")
        for manifest in completion_manifests
        for step in manifest.get("steps", [])
        if step.get("transcoder_diagnostics")
    ]
    cache_evictions = [
        step.get("transcoder_diagnostics", {}).get("decoder_cache_eviction_count")
        for manifest in completion_manifests
        for step in manifest.get("steps", [])
        if step.get("transcoder_diagnostics")
    ]
    feature_semantic_descriptor_statuses = [
        step.get("feature_semantic_descriptor_status")
        for manifest in completion_manifests
        for step in manifest.get("steps", [])
        if isinstance(step.get("feature_semantic_descriptor_status"), str)
    ]
    feature_semantic_descriptor_paths = [
        Path(path).parent / step.get("feature_semantic_descriptor_path")
        for path, manifest in zip(completion_files, completion_manifests, strict=False)
        for step in manifest.get("steps", [])
        if isinstance(step.get("feature_semantic_descriptor_path"), str)
    ]

    first_prompt_meta = prompt_metas[0] if prompt_metas else {}
    first_completion = completion_manifests[0] if completion_manifests else {}
    return {
        "prompt_count": len(prompt_metas),
        "completion_count": len(completion_manifests),
        "prompt_source": first_prompt_meta.get("prompt_source"),
        "fixture_name": first_prompt_meta.get("fixture_name"),
        "fixture_kind": first_prompt_meta.get("fixture_kind"),
        "prompt_token_count": first_completion.get(
            "prompt_token_count",
            first_prompt_meta.get("prompt_token_count"),
        ),
        "initial_input_token_count": first_completion.get(
            "initial_input_token_count",
            first_prompt_meta.get("initial_input_token_count"),
        ),
        "generated_token_count": first_completion.get("generated_token_count"),
        "n_steps_traced": first_completion.get("n_steps_traced"),
        "max_active_features": max(active_feature_counts)
        if active_feature_counts
        else None,
        "decoder_cache_hit_count": max(cache_hits) if cache_hits else None,
        "decoder_cache_miss_count": max(cache_misses) if cache_misses else None,
        "decoder_cache_eviction_count": max(cache_evictions)
        if cache_evictions
        else None,
        "feature_semantic_descriptor_status": (
            feature_semantic_descriptor_statuses[-1]
            if feature_semantic_descriptor_statuses
            else None
        ),
        "feature_semantic_descriptor_file_count": sum(
            1 for path in feature_semantic_descriptor_paths if path.exists()
        ),
        "resource_snapshot": first_completion.get("resource_snapshot"),
    }


def _classify_status(log_path: Path, *, returncode: int | None) -> str:
    if returncode == 0:
        return "success"
    if returncode is None:
        return "timeout"
    if not log_path.exists():
        return "failed"

    log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if "out of memory" in log_text or "cuda oom" in log_text:
        return "oom"
    return "failed"


def build_command(
    output_dir: Path,
    scenario: dict[str, Any],
    *,
    cross_batch_decoder_cache_bytes_override: int | None = None,
) -> list[str]:
    scenario = apply_runtime_overrides(
        scenario,
        cross_batch_decoder_cache_bytes_override=cross_batch_decoder_cache_bytes_override,
    )
    method = scenario["method"]
    script_name = (
        "trace_pipeline.py" if method == "old_patch" else "trace_pipeline_chunked.py"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / script_name),
        "--completions",
        str(scenario["completions"]),
        "--temperature",
        str(scenario["temperature"]),
        "--output-dir",
        str(output_dir),
        "--max-feature-nodes",
        str(scenario["max_feature_nodes"]),
        "--max-edges",
        str(scenario["max_edges"]),
        "--max-steps",
        str(scenario["max_steps"]),
        "--attribution-batch-size",
        str(scenario["attribution_batch_size"]),
        "--max-n-logits",
        str(scenario["max_n_logits"]),
        "--desired-logit-prob",
        str(scenario["desired_logit_prob"]),
        "--attribution-update-interval",
        str(scenario["attribution_update_interval"]),
    ]
    cmd[2:2] = _build_prompt_source_args(scenario)
    if scenario.get("feature_batch_size") is not None:
        cmd.extend(["--feature-batch-size", str(scenario["feature_batch_size"])])
    if scenario.get("logit_batch_size") is not None:
        cmd.extend(["--logit-batch-size", str(scenario["logit_batch_size"])])
    if scenario.get("exact_trace_internal_dtype") is not None:
        cmd.extend(
            [
                "--exact-trace-internal-dtype",
                str(scenario["exact_trace_internal_dtype"]),
            ]
        )
    if (
        method != "old_patch"
        and scenario.get("phase0_activation_threshold_compare_mode") is not None
    ):
        cmd.extend(
            [
                "--phase0-activation-threshold-compare-mode",
                str(scenario["phase0_activation_threshold_compare_mode"]),
            ]
        )

    if method != "old_patch":
        if scenario.get("phase1_trace_batch_policy") is not None:
            cmd.extend(
                [
                    "--phase1-trace-batch-policy",
                    str(scenario["phase1_trace_batch_policy"]),
                ]
            )
        if scenario.get("phase1_trace_batch_size_max") is not None:
            cmd.extend(
                [
                    "--phase1-trace-batch-size-max",
                    str(scenario["phase1_trace_batch_size_max"]),
                ]
            )
        cmd.extend(["--decoder-chunk-size", str(scenario["decoder_chunk_size"])])
        cross_batch_decoder_cache_bytes = (
            cross_batch_decoder_cache_bytes_override
            if cross_batch_decoder_cache_bytes_override is not None
            else scenario.get("cross_batch_decoder_cache_bytes")
        )
        if cross_batch_decoder_cache_bytes is not None:
            cmd.extend(
                [
                    "--cross-batch-decoder-cache-bytes",
                    str(cross_batch_decoder_cache_bytes),
                ]
            )
        if scenario.get("sparsify_per_layer_position_topk") is not None:
            cmd.extend(
                [
                    "--sparsify-per-layer-position-topk",
                    str(scenario["sparsify_per_layer_position_topk"]),
                ]
            )
        if scenario.get("sparsify_global_cap") is not None:
            cmd.extend(["--sparsify-global-cap", str(scenario["sparsify_global_cap"])])
        if scenario.get("chunked_feature_replay_window") is not None:
            cmd.extend(
                [
                    "--chunked-feature-replay-window",
                    str(scenario["chunked_feature_replay_window"]),
                ]
            )
        if scenario.get("error_vector_prefetch_lookahead") is not None:
            cmd.extend(
                [
                    "--error-vector-prefetch-lookahead",
                    str(scenario["error_vector_prefetch_lookahead"]),
                ]
            )
        if scenario.get("stage_encoder_vecs_on_cpu") is not None:
            cmd.extend(
                [
                    "--stage-encoder-vecs-on-cpu",
                    _format_optional_bool_arg(scenario["stage_encoder_vecs_on_cpu"]),
                ]
            )
        if scenario.get("stage_error_vectors_on_cpu") is not None:
            cmd.extend(
                [
                    "--stage-error-vectors-on-cpu",
                    _format_optional_bool_arg(scenario["stage_error_vectors_on_cpu"]),
                ]
            )
        if scenario.get("row_subchunk_size") is not None:
            cmd.extend(["--row-subchunk-size", str(scenario["row_subchunk_size"])])
        if scenario.get("plan_feature_batch_size", False):
            cmd.append("--plan-feature-batch-size")
        if scenario.get("auto_scale_feature_batch_size", False):
            cmd.append("--auto-scale-feature-batch-size")
        if scenario.get("feature_batch_size_max") is not None:
            cmd.extend(
                ["--feature-batch-size-max", str(scenario["feature_batch_size_max"])]
            )
        if scenario.get("feature_batch_target_reserved_fraction") is not None:
            cmd.extend(
                [
                    "--feature-batch-target-reserved-fraction",
                    str(scenario["feature_batch_target_reserved_fraction"]),
                ]
            )
        if scenario.get("feature_batch_min_free_fraction") is not None:
            cmd.extend(
                [
                    "--feature-batch-min-free-fraction",
                    str(scenario["feature_batch_min_free_fraction"]),
                ]
            )
        if scenario.get("feature_batch_probe_batches") is not None:
            cmd.extend(
                [
                    "--feature-batch-probe-batches",
                    str(scenario["feature_batch_probe_batches"]),
                ]
            )
        if scenario.get("phase4_anomaly_debug", False):
            cmd.append("--phase4-anomaly-debug")
        if scenario.get("phase4_refresh_policy") is not None:
            cmd.extend(
                [
                    "--phase4-refresh-policy",
                    str(scenario["phase4_refresh_policy"]),
                ]
            )
        if scenario.get("phase4_refresh_interval_multiplier") is not None:
            cmd.extend(
                [
                    "--phase4-refresh-interval-multiplier",
                    str(scenario["phase4_refresh_interval_multiplier"]),
                ]
            )
        if scenario.get("phase4_ranker") is not None:
            cmd.extend(["--phase4-ranker", str(scenario["phase4_ranker"])])
        if scenario.get("row_store_cache_control") is not None:
            cmd.extend(
                [
                    "--row-store-cache-control",
                    str(scenario["row_store_cache_control"]),
                ]
            )
        if scenario.get("exact_encoder_residency") is not None:
            cmd.extend(
                [
                    "--exact-encoder-residency",
                    str(scenario["exact_encoder_residency"]),
                ]
            )
        if scenario.get("phase4_scheduler_mode") is not None:
            cmd.extend(
                [
                    "--phase4-scheduler-mode",
                    str(scenario["phase4_scheduler_mode"]),
                ]
            )
        if scenario.get("phase4_scheduler_debug", False):
            cmd.append("--phase4-scheduler-debug")
        if scenario.get("phase4_scheduler_telemetry_detail") is not None:
            cmd.extend(
                [
                    "--phase4-scheduler-telemetry-detail",
                    str(scenario["phase4_scheduler_telemetry_detail"]),
                ]
            )
        if scenario.get("phase4_refresh_optimization") is not None:
            cmd.extend(
                [
                    "--phase4-refresh-optimization",
                    str(scenario["phase4_refresh_optimization"]),
                ]
            )
        if scenario.get("phase4_row_executor") is not None:
            cmd.extend(
                [
                    "--phase4-row-executor",
                    str(scenario["phase4_row_executor"]),
                ]
            )
        if scenario.get("cross_cluster_debug", False):
            cmd.append("--cross-cluster-debug")
        if scenario.get("capture_phase0_donor_bundle", False):
            cmd.append("--capture-phase0-donor-bundle")
        if scenario.get("phase0_donor_bundle") is not None:
            cmd.extend(["--phase0-donor-bundle", str(scenario["phase0_donor_bundle"])])
        if scenario.get("phase0_replay_mode") is not None:
            cmd.extend(["--phase0-replay-mode", str(scenario["phase0_replay_mode"])])
        if scenario.get("phase0_donor_context_policy") is not None:
            cmd.extend(
                [
                    "--phase0-donor-context-policy",
                    str(scenario["phase0_donor_context_policy"]),
                ]
            )
        if scenario.get("phase3_gradient_donor_bundle") is not None:
            cmd.extend(
                [
                    "--phase3-gradient-donor-bundle",
                    str(scenario["phase3_gradient_donor_bundle"]),
                ]
            )
        if scenario.get("phase3_gradient_replay_mode") is not None:
            cmd.extend(
                [
                    "--phase3-gradient-replay-mode",
                    str(scenario["phase3_gradient_replay_mode"]),
                ]
            )
        if scenario.get("phase3_row_donor_bundle") is not None:
            cmd.extend(
                ["--phase3-row-donor-bundle", str(scenario["phase3_row_donor_bundle"])]
            )
        if scenario.get("phase3_row_replay_mode") is not None:
            cmd.extend(
                ["--phase3-row-replay-mode", str(scenario["phase3_row_replay_mode"])]
            )
        if scenario.get("phase3_replay_validation_policy") is not None:
            cmd.extend(
                [
                    "--phase3-replay-validation-policy",
                    str(scenario["phase3_replay_validation_policy"]),
                ]
            )
        if scenario.get("capture_phase3_seed_bundle", False):
            cmd.append("--capture-phase3-seed-bundle")
        if scenario.get("capture_phase3_gradient_bundle", False):
            cmd.append("--capture-phase3-gradient-bundle")
        if scenario.get("capture_phase3_row_bundle", False):
            cmd.append("--capture-phase3-row-bundle")
        if scenario.get("capture_feature_semantic_descriptors", False):
            cmd.append("--capture-feature-semantic-descriptors")
        if scenario.get("semantic_descriptor_top_k") is not None:
            cmd.extend(
                [
                    "--semantic-descriptor-top-k",
                    str(scenario["semantic_descriptor_top_k"]),
                ]
            )
        if scenario.get("semantic_descriptor_dim") is not None:
            cmd.extend(
                [
                    "--semantic-descriptor-dim",
                    str(scenario["semantic_descriptor_dim"]),
                ]
            )
        if scenario.get("telemetry_max_events") is not None:
            cmd.extend(
                ["--telemetry-max-events", str(scenario["telemetry_max_events"])]
            )

    if scenario.get("verbose_attribution", False):
        cmd.append("--verbose-attribution")
    if scenario.get("profile_attribution", False):
        cmd.append("--profile-attribution")
    if "profile_log_interval" in scenario:
        cmd.extend(["--profile-log-interval", str(scenario["profile_log_interval"])])
    if scenario.get("diagnostic_feature_cap") is not None:
        cmd.extend(
            ["--diagnostic-feature-cap", str(scenario["diagnostic_feature_cap"])]
        )
    if scenario.get("save_raw", False):
        cmd.append("--save-raw")
    if scenario.get("no_offload", False):
        cmd.append("--no-offload")
    if scenario.get("no_lazy_encoder", False) and method != "old_patch":
        cmd.append("--no-lazy-encoder")
    if scenario.get("no_lazy_decoder", False) and method != "old_patch":
        cmd.append("--no-lazy-decoder")

    return cmd


def run_scenario(
    output_root: Path,
    scenario: dict[str, Any],
    *,
    env: dict[str, str],
    run_metadata: dict[str, str | None],
    baseline_registry: dict[str, dict[str, Any]] | None = None,
    baseline_registry_path: Path | None = None,
    fail_on_baseline_missing: bool = False,
    fail_on_validation_fail: bool = False,
    cross_batch_decoder_cache_bytes_override: int | None = None,
) -> dict[str, Any]:
    scenario_name = scenario["name"]
    scenario_root = (
        output_root
        if output_root.name == scenario_name
        else output_root / scenario_name
    )
    run_output_dir = scenario_root / "artifacts"
    _assert_fresh_scenario_root(scenario_root)
    scenario_root.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    effective_scenario = apply_runtime_overrides(
        scenario,
        cross_batch_decoder_cache_bytes_override=cross_batch_decoder_cache_bytes_override,
    )
    scenario_payload = dict(effective_scenario)
    if _has_run_metadata(run_metadata):
        scenario_payload["run_metadata"] = run_metadata
    (scenario_root / "scenario.json").write_text(json.dumps(scenario_payload, indent=2))

    log_path = scenario_root / "run.log"
    cmd = build_command(
        run_output_dir,
        scenario,
        cross_batch_decoder_cache_bytes_override=cross_batch_decoder_cache_bytes_override,
    )
    timeout_minutes = scenario.get("timeout_minutes")
    timeout_seconds = None if timeout_minutes is None else int(timeout_minutes * 60)

    result: dict[str, Any] = {
        "name": scenario_name,
        "stage": scenario.get("stage"),
        "method": scenario["method"],
        "run_id": run_metadata.get("run_id"),
        "launch_id": run_metadata.get("run_id"),
        "run_name": run_metadata.get("run_name"),
        "run_description": run_metadata.get("run_description"),
        "run_goal": run_metadata.get("run_goal"),
        "run_metadata": run_metadata,
        "gsm8k_indices": scenario.get("gsm8k_indices"),
        "prepared_prompt_file": scenario.get("prepared_prompt_file"),
        "prepared_prompt_meta_file": scenario.get("prepared_prompt_meta_file"),
        "command": cmd,
        "output_dir": str(run_output_dir),
        "status": "unknown",
    }
    if timeout_minutes is not None:
        result["timeout_minutes"] = timeout_minutes

    baseline_check = {}
    baseline_entry = None
    try:
        baseline_check = normalize_baseline_check(effective_scenario)
        baseline_check, baseline_entry = resolve_baseline_entry(
            baseline_check,
            registry=baseline_registry,
            registry_path=baseline_registry_path,
        )
        if baseline_entry is not None:
            baseline_check = validate_baseline_entry(
                baseline_entry,
                status=baseline_check,
            )
    except Exception as exc:  # noqa: BLE001 - keep failure in scenario artifacts
        baseline_check = {
            **BASELINE_DISABLED,
            "enabled": True,
            "status": "baseline_invalid",
            "passed": False,
            "failure_reasons": [str(exc)],
        }

    if (
        baseline_check.get("enabled")
        and baseline_check.get("baseline_required", True)
        and baseline_check.get("status") in {"baseline_missing", "baseline_invalid"}
    ):
        result["status"] = "baseline_invalid"
        result["returncode"] = None
        result["duration_seconds"] = 0.0
        result["log_path"] = str(log_path)
        result["baseline_check"] = baseline_check
        row = build_scenario_metrics_row(
            scenario=effective_scenario,
            result=result,
            baseline_status=baseline_check,
        )
        write_scenario_metrics(
            scenario_root,
            row,
            baseline_status=baseline_check,
        )
        (scenario_root / "result.json").write_text(json.dumps(result, indent=2))
        if fail_on_baseline_missing:
            raise RuntimeError(
                f"Required baseline invalid for scenario {scenario_name}: "
                f"{baseline_check.get('failure_reasons')}"
            )
        return result

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Scenario: {scenario_name}\n")
        log_file.write(f"Stage: {scenario.get('stage')}\n")
        log_file.write(f"Method: {scenario['method']}\n")
        if _has_run_metadata(run_metadata):
            log_file.write(f"Run ID: {run_metadata.get('run_id')}\n")
            log_file.write(f"Run name: {run_metadata.get('run_name')}\n")
            if run_metadata.get("run_description"):
                log_file.write(f"Run description: {run_metadata['run_description']}\n")
            if run_metadata.get("run_goal"):
                log_file.write(f"Run goal: {run_metadata['run_goal']}\n")
        log_file.write(f"Start: {time.ctime(start)}\n")
        log_file.write(f"Working directory: {REPO_ROOT}\n")
        log_file.write(f"Command: {shlex.join(cmd)}\n\n")
        log_file.flush()

        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["returncode"] = None
            log_file.write(f"\nTimed out after {timeout_minutes} minutes.\n")
        else:
            result["returncode"] = completed.returncode

    result["duration_seconds"] = round(time.time() - start, 2)
    result["log_path"] = str(log_path)
    result["status"] = _classify_status(
        log_path,
        returncode=result.get("returncode"),
    )
    result["profiling_summary"] = _extract_benchmark_metrics(log_path)
    result["artifact_summary"] = _summarize_artifacts(run_output_dir)

    comparison_metrics: dict[str, Any] = {}
    if baseline_check.get("enabled"):
        if result["status"] != "success":
            baseline_check["status"] = "skipped_trace_failed"
            baseline_check["passed"] = False
            failure_reasons = baseline_check.setdefault("failure_reasons", [])
            if isinstance(failure_reasons, list):
                failure_reasons.append(f"trace status was {result['status']}")
        elif baseline_check.get("status") in {"baseline_missing", "baseline_invalid"}:
            baseline_check["passed"] = False
        else:
            baseline_check, comparison_metrics = run_baseline_comparison(
                scenario_root=scenario_root,
                current_artifacts=run_output_dir,
                baseline_check=baseline_check,
                baseline_entry=baseline_entry,
            )
    else:
        baseline_check = dict(BASELINE_DISABLED)

    result["baseline_check"] = baseline_check
    row = build_scenario_metrics_row(
        scenario=effective_scenario,
        result=result,
        baseline_status=baseline_check,
        comparison_metrics=comparison_metrics,
    )
    write_scenario_metrics(
        scenario_root,
        row,
        baseline_status=baseline_check,
    )
    (scenario_root / "result.json").write_text(json.dumps(result, indent=2))
    if (
        fail_on_validation_fail
        and baseline_check.get("enabled")
        and baseline_check.get("passed") is False
        and baseline_check.get("status") in {"gate_fail", "compare_error"}
    ):
        raise RuntimeError(
            f"Baseline validation failed for scenario {scenario_name}: "
            f"{baseline_check.get('failure_reasons')}"
        )
    return result


def load_scenarios(scenarios_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads(scenarios_file.read_text())
    defaults = config.get("defaults", {})
    scenarios = [defaults | scenario for scenario in config["scenarios"]]
    return scenarios, config.get("metadata", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibration/main scenarios for the sparsification experiment"
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="JSON file describing experiment scenarios",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for scenario logs, artifacts, and summaries",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help="Run only the scenario at this 0-based index",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Run only the scenario with this name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print scenario indices and names without running anything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them",
    )
    parser.add_argument(
        "--cross-batch-decoder-cache-bytes",
        type=int,
        default=None,
        help="Optional override for exact chunked Phase-4 cross-batch decoder cache budget",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier for grouping scenario outputs and metadata",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable run name",
    )
    parser.add_argument(
        "--run-description",
        default=None,
        help="Optional short free-text run description",
    )
    parser.add_argument(
        "--run-goal",
        default=None,
        help="Optional free-text run goal",
    )
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=None,
        help="Optional pinned baseline registry used by scenario baseline_check blocks",
    )
    parser.add_argument(
        "--fail-on-baseline-missing",
        action="store_true",
        help="Exit nonzero after writing artifacts if a required baseline is missing/invalid",
    )
    parser.add_argument(
        "--fail-on-validation-fail",
        action="store_true",
        help="Exit nonzero after writing artifacts if a gate comparison fails",
    )
    args = parser.parse_args()

    scenarios, metadata = load_scenarios(args.scenarios_file)
    baseline_registry_path = args.baseline_registry
    if baseline_registry_path is None and metadata.get("baseline_registry"):
        baseline_registry_path = Path(str(metadata["baseline_registry"]))
    baseline_registry = None
    if baseline_registry_path is not None:
        baseline_registry = load_baseline_registry(baseline_registry_path)
    run_metadata = {
        "run_id": _normalize_run_metadata_value(args.run_id)
        or _normalize_run_metadata_value(metadata.get("run_id")),
        "run_name": _normalize_run_metadata_value(args.run_name)
        or _normalize_run_metadata_value(metadata.get("run_name")),
        "run_description": _normalize_run_metadata_value(args.run_description)
        or _normalize_run_metadata_value(metadata.get("run_description")),
        "run_goal": _normalize_run_metadata_value(args.run_goal)
        or _normalize_run_metadata_value(metadata.get("run_goal")),
    }

    if args.list:
        for idx, scenario in enumerate(scenarios):
            print(
                f"[{idx:03d}] {scenario['name']} | stage={scenario.get('stage')} | method={scenario['method']}"
            )
        return

    if args.scenario_index is not None and args.scenario_name is not None:
        raise ValueError("Use only one of --scenario-index or --scenario-name")

    if args.scenario_index is not None:
        scenarios = [scenarios[args.scenario_index]]
    elif args.scenario_name is not None:
        scenarios = [
            scenario for scenario in scenarios if scenario["name"] == args.scenario_name
        ]
        if not scenarios:
            raise ValueError(f"Scenario not found: {args.scenario_name}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root / timestamp
        if len(scenarios) > 1
        else args.output_root / scenarios[0]["name"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"Writing experiment results to {output_root}")
    print(f"Loaded {len(scenarios)} scenario(s) from {args.scenarios_file}")
    if metadata:
        print(f"Scenario metadata: {json.dumps(metadata, indent=2)}")
    if _has_run_metadata(run_metadata):
        print(f"Run metadata: {json.dumps(run_metadata, indent=2)}")

    results = []
    scenario_metric_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = scenario["name"]
        scenario_root = output_root if len(scenarios) == 1 else output_root / name
        if args.dry_run:
            cmd = build_command(
                scenario_root / "artifacts",
                scenario,
                cross_batch_decoder_cache_bytes_override=args.cross_batch_decoder_cache_bytes,
            )
            print(f"DRY RUN {name}: {shlex.join(cmd)}")
            continue
        print(f"\n{'=' * 80}\nRunning scenario: {name}\n{'=' * 80}")
        result = run_scenario(
            output_root,
            scenario,
            env=env,
            run_metadata=run_metadata,
            baseline_registry=baseline_registry,
            baseline_registry_path=baseline_registry_path,
            fail_on_baseline_missing=args.fail_on_baseline_missing,
            fail_on_validation_fail=args.fail_on_validation_fail,
            cross_batch_decoder_cache_bytes_override=args.cross_batch_decoder_cache_bytes,
        )
        results.append(result)
        scenario_metrics_path = (
            Path(result["output_dir"]).parent / "scenario_metrics.json"
        )
        if scenario_metrics_path.exists():
            payload = json.loads(scenario_metrics_path.read_text())
            if isinstance(payload.get("metrics"), dict):
                scenario_metric_rows.append(payload["metrics"])
        print(
            f"Completed {name}: status={results[-1]['status']} duration={results[-1]['duration_seconds']:.2f}s"
        )

    if args.dry_run:
        return

    summary = {
        "timestamp": timestamp,
        "scenarios_file": str(args.scenarios_file),
        "metadata": metadata,
        "run_id": run_metadata.get("run_id"),
        "launch_id": run_metadata.get("run_id"),
        "run_name": run_metadata.get("run_name"),
        "run_description": run_metadata.get("run_description"),
        "run_goal": run_metadata.get("run_goal"),
        "run_metadata": run_metadata,
        "output_root": str(output_root),
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if (
        args.scenario_index is None
        and args.scenario_name is None
        and scenario_metric_rows
    ):
        write_csv(output_root / "summary.csv", scenario_metric_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
