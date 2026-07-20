from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_GENERATED_DIR, DEFAULT_SCRATCH_ROOT, base_trace_defaults
from ..fixtures import resolve_fixture
from ..io_utils import ensure_dir, write_json
from ..transcoder_config import (
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)
from .governor_calibration import GIB, GOVERNOR_CALIBRATION_BASELINE_REGISTRY


PHASE4_FEEDBACK_DIAGNOSTIC_FIXTURE = "361_base"
PHASE4_FEEDBACK_DIAGNOSTIC_VARIANT = "gemma3_1b_plt"
PHASE4_FEEDBACK_DIAGNOSTIC_PROVIDER_FAMILY = "gemmascope2-plt-1b-small"
PHASE4_FEEDBACK_DIAGNOSTIC_GOVERNOR_PROFILE = (
    "granite_h200_1b_plt_b128_c4096_cache0"
)
PHASE4_FEEDBACK_DIAGNOSTIC_RESOURCE_PROFILE = "governor_calibration_h200"
PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_REGISTRY = GOVERNOR_CALIBRATION_BASELINE_REGISTRY
PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_KEY = (
    "governor_calibration/gemma3_1b_plt/361_base"
)
PHASE4_FEEDBACK_DIAGNOSTIC_TIMEOUT_MINUTES = 120


def _diagnostic_rows() -> tuple[dict[str, Any], ...]:
    return (
        {
            "diagnostic_case": "canonical_reference",
            "feature_batch_size": 128,
            "attribution_update_interval": 4,
            "nnsight_session_capacity": 128,
            "phase4_execution_batch_max_rows": 128,
            "governor_fidelity_mode": "strict",
        },
        {
            "diagnostic_case": "constant_window_logical_b256_session128_physical_b128",
            "feature_batch_size": 256,
            "attribution_update_interval": 2,
            "nnsight_session_capacity": 128,
            "phase4_execution_batch_max_rows": 128,
            "governor_fidelity_mode": "research",
            "governor_fidelity_override_fields": [
                "feature_batch_size",
                "frontier_refresh_stride",
            ],
        },
        {
            "diagnostic_case": "constant_window_logical_b256_physical_b128",
            "feature_batch_size": 256,
            "attribution_update_interval": 2,
            "nnsight_session_capacity": 256,
            "phase4_execution_batch_max_rows": 128,
            "governor_fidelity_mode": "research",
            "governor_fidelity_override_fields": [
                "feature_batch_size",
                "frontier_refresh_stride",
            ],
        },
        {
            "diagnostic_case": "constant_window_logical_b256_physical_b256",
            "feature_batch_size": 256,
            "attribution_update_interval": 2,
            "nnsight_session_capacity": 256,
            "phase4_execution_batch_max_rows": 256,
            "governor_fidelity_mode": "research",
            "governor_fidelity_override_fields": [
                "feature_batch_size",
                "frontier_refresh_stride",
            ],
        },
        {
            "diagnostic_case": "stale_window_logical_b256_physical_b128",
            "feature_batch_size": 256,
            "attribution_update_interval": 4,
            "nnsight_session_capacity": 256,
            "phase4_execution_batch_max_rows": 128,
            "governor_fidelity_mode": "research",
            "governor_fidelity_override_fields": ["feature_batch_size"],
        },
    )


def build_phase4_feedback_diagnostic_config(
    *,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
    baseline_registry: Path = PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_REGISTRY,
) -> dict[str, Any]:
    fixture = resolve_fixture(
        PHASE4_FEEDBACK_DIAGNOSTIC_FIXTURE,
        catalog_by_name=catalog_by_name,
        allow_fallback=True,
    )
    cache_root = model_cache_root or scratch_root.parent / "huggingface"
    provider_config = transcoder_config_to_json(
        resolve_transcoder_load_config(
            transcoder_provider_family=PHASE4_FEEDBACK_DIAGNOSTIC_PROVIDER_FAMILY,
            transcoder_cache_dir=str(cache_root),
        )
    )
    stage = "phase4_feedback_diagnostic_gemma3_1b_plt"
    output_root = (
        scratch_root
        / "granite"
        / "sweep"
        / "governor_calibration"
        / "phase4_feedback_diagnostic"
        / PHASE4_FEEDBACK_DIAGNOSTIC_VARIANT
    )
    defaults = {
        **base_trace_defaults(),
        **provider_config,
        "stage": stage,
        "cluster": "granite",
        "tier": "sweep",
        "resource_profile": PHASE4_FEEDBACK_DIAGNOSTIC_RESOURCE_PROFILE,
        "governor_profile_name": PHASE4_FEEDBACK_DIAGNOSTIC_GOVERNOR_PROFILE,
        "governor_admission_mode": "advisory",
        "attribution_batch_size": 128,
        "feature_batch_size": 128,
        "logit_batch_size": 128,
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "exact_trace_internal_dtype": "fp32",
        "completions": 1,
        "max_steps": 1,
        "save_raw": False,
        "incremental_telemetry_jsonl": True,
        "verbose_attribution": True,
        "profile_attribution": True,
        "profile_log_interval": 1,
    }
    envelope = {
        "total_vram_bytes": 151_397_597_184,
        "vram_fraction": 0.9,
        "host_budget_bytes": 200 * GIB,
        "file_cache_allowance_bytes": 64 * GIB,
        "local_disk_bytes": 64 * GIB,
        "scratch_disk_bytes": 64 * GIB,
        "walltime_seconds": PHASE4_FEEDBACK_DIAGNOSTIC_TIMEOUT_MINUTES * 60,
    }
    scenarios = []
    for row in _diagnostic_rows():
        case = str(row["diagnostic_case"])
        scenario = {
            "name": f"granite_phase4_feedback_diagnostic_gemma3_1b_plt_{case}",
            "run_name": f"phase4-feedback-diagnostic-gemma3-1b-plt-{case}",
            "run_goal": (
                "Isolate Phase-4 frontier feedback effects while holding source, "
                "logit, decoder, cache, and non-Phase-4 physical axes fixed."
            ),
            **fixture.to_source_payload(),
            "attribution_batch_size": 128,
            "logit_batch_size": 128,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": 128,
            "phase3_compute_microbatch_max_rows": 128,
            "decoder_chunk_size": 4096,
            "cross_batch_decoder_cache_bytes": 0,
            **row,
            "governor_resource_envelope": dict(envelope),
            "timeout_minutes": PHASE4_FEEDBACK_DIAGNOSTIC_TIMEOUT_MINUTES,
            "baseline_check": {
                "enabled": True,
                "mode": "metrics",
                "registry_key": PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_KEY,
                "baseline_required": True,
            },
        }
        scenarios.append(scenario)

    return {
        "schema_version": 1,
        "defaults": defaults,
        "metadata": {
            "cluster": "granite",
            "stage": stage,
            "operational_class": "sweep",
            "resource_profile": PHASE4_FEEDBACK_DIAGNOSTIC_RESOURCE_PROFILE,
            "array_concurrency": 1,
            "baseline_registry": str(baseline_registry),
            "baseline_registry_pinned": True,
            "fail_on_baseline_missing": True,
            "fail_on_validation_fail": True,
            "immutable_validation_config": True,
            "recommended_output_root": str(output_root),
            "matrix_variant": PHASE4_FEEDBACK_DIAGNOSTIC_VARIANT,
            "matrix_case_count": len(scenarios),
            "governor_admission_policy": {
                "mode": "advisory",
                "scope": "phase4_feedback_diagnostic_only",
                "reason": (
                    "Forced diagnostic values must execute to produce evidence even "
                    "when the governor would refuse them under enforce admission."
                ),
            },
            "fixed_protocol": {
                "hardware": "single_h200",
                "gpu_count": 1,
                "dtype": "fp32",
                "fixture": PHASE4_FEEDBACK_DIAGNOSTIC_FIXTURE,
                "decoder_chunk_size": 4096,
                "decoder_cache_bytes": 0,
                "source_batch_size": 128,
                "logit_batch_size": 128,
                "phase1_physical_cap": 128,
                "phase3_physical_cap": 128,
            },
            "contrasts": {
                "logical_grouping_constant_window": {
                    "reference": "canonical_reference",
                    "comparison": "constant_window_logical_b256_session128_physical_b128",
                    "change": "logical grouping 128x4 to 256x2 at constant 512-feature frontier window with session and physical Phase-4 batch fixed at 128",
                },
                "session_capacity": {
                    "reference": "constant_window_logical_b256_session128_physical_b128",
                    "comparison": "constant_window_logical_b256_physical_b128",
                    "change": "session capacity 128 to 256 with logical 256x2 frontier window and physical Phase-4 batch fixed at 128",
                },
                "physical_microbatch_constant_window": {
                    "reference": "constant_window_logical_b256_physical_b128",
                    "comparison": "constant_window_logical_b256_physical_b256",
                    "change": "Phase-4 physical microbatch 128 to 256 at constant 256x2 logical frontier window",
                },
                "stale_window": {
                    "reference": "constant_window_logical_b256_physical_b128",
                    "comparison": "stale_window_logical_b256_physical_b128",
                    "change": "stale 1024-feature window 256x4 versus constant-window 512-feature window 256x2",
                },
            },
            "slurm": {
                "account": "rai",
                "partition": "rai-gpu-grn",
                "qos": "rai-gpu-grn-short",
                "gres": "gpu:h200:1",
                "cpus_per_task": 12,
                "mem": "200G",
                "time": "02:00:00",
            },
        },
        "scenarios": scenarios,
    }


def phase4_feedback_diagnostic_scenario_file_name() -> str:
    return "exact_trace_phase4_feedback_diagnostic_gemma3_1b_plt_granite_scenarios.json"


def write_phase4_feedback_diagnostic_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
) -> Path:
    ensure_dir(output_dir)
    path = output_dir / phase4_feedback_diagnostic_scenario_file_name()
    write_json(path, payload)
    return path
