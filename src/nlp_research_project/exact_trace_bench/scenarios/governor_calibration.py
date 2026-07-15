from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DEFAULT_GENERATED_DIR, DEFAULT_SCRATCH_ROOT, base_trace_defaults
from ..fixtures import resolve_fixture
from ..io_utils import ensure_dir, write_json
from ..transcoder_config import (
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)


GIB = 1024**3
GOVERNOR_CALIBRATION_FIXTURE = "361_base"


@dataclass(frozen=True)
class GovernorCalibrationProvider:
    name: str
    provider_family: str
    governor_profile: str
    logical_batch: int
    reference_decoder_cache_bytes: int
    lower_session_capacity: int
    host_memory_bytes: int
    timeout_minutes: int
    slurm_memory: str
    slurm_time: str


GOVERNOR_CALIBRATION_PROVIDERS = {
    provider.name: provider
    for provider in (
        GovernorCalibrationProvider(
            name="gemma3_1b_clt",
            provider_family="gemmascope2-clt-1b-medium-affine",
            governor_profile="granite_h200_1b_clt_b1000_c4096_cache8",
            logical_batch=1000,
            reference_decoder_cache_bytes=8 * GIB,
            lower_session_capacity=500,
            host_memory_bytes=200 * GIB,
            timeout_minutes=180,
            slurm_memory="200G",
            slurm_time="03:00:00",
        ),
        GovernorCalibrationProvider(
            name="gemma3_1b_plt",
            provider_family="gemmascope2-plt-1b-small",
            governor_profile="granite_h200_1b_plt_b128_c4096_cache0",
            logical_batch=128,
            reference_decoder_cache_bytes=0,
            lower_session_capacity=64,
            host_memory_bytes=200 * GIB,
            timeout_minutes=480,
            slurm_memory="200G",
            slurm_time="08:00:00",
        ),
        GovernorCalibrationProvider(
            name="gemma3_4b_plt",
            provider_family="gemmascope2-plt-4b-small",
            governor_profile="granite_h200_4b_plt_b128_c4096_cache0",
            logical_batch=128,
            reference_decoder_cache_bytes=0,
            lower_session_capacity=64,
            host_memory_bytes=400 * GIB,
            timeout_minutes=480,
            slurm_memory="400G",
            slurm_time="08:00:00",
        ),
        GovernorCalibrationProvider(
            name="gemma3_12b_plt",
            provider_family="gemmascope2-plt-12b-small",
            governor_profile="granite_h200_12b_plt_b64_c4096_cache0",
            logical_batch=64,
            reference_decoder_cache_bytes=0,
            lower_session_capacity=32,
            host_memory_bytes=600 * GIB,
            timeout_minutes=960,
            slurm_memory="600G",
            slurm_time="16:00:00",
        ),
    )
}


def _reference_controls(provider: GovernorCalibrationProvider) -> dict[str, Any]:
    batch = provider.logical_batch
    return {
        "nnsight_session_capacity": batch,
        "phase1_trace_batch_policy": "legacy",
        "phase3_compute_microbatch_max_rows": batch,
        "phase4_compute_microbatch_max_rows": batch,
        "cross_batch_decoder_cache_bytes": provider.reference_decoder_cache_bytes,
        "chunked_feature_replay_window": 4,
        "error_vector_prefetch_lookahead": 2,
        "governor_required_encoder_residency": "lazy",
        "governor_required_row_store_policy": "file_backed_full",
        "governor_required_replay_tile_cache_bytes": 0,
        "governor_required_spill_target": "local",
    }


def _contrast_rows(provider: GovernorCalibrationProvider) -> list[dict[str, Any]]:
    reference = _reference_controls(provider)
    lower = provider.lower_session_capacity
    rows = [
        {
            "calibration_case": "reference",
            "calibration_factor": "reference",
            "calibration_pair": "reference",
            **reference,
        },
        {
            "calibration_case": "session_reference",
            "calibration_factor": "session_capacity",
            "calibration_pair": "session",
            **reference,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": lower,
            "phase3_compute_microbatch_max_rows": lower,
            "phase4_compute_microbatch_max_rows": lower,
        },
        {
            "calibration_case": "lower_session",
            "calibration_factor": "session_capacity",
            "calibration_pair": "session",
            **reference,
            "nnsight_session_capacity": lower,
            "phase1_trace_batch_policy": "cap_effective_batches",
            "phase1_trace_batch_size_max": lower,
            "phase3_compute_microbatch_max_rows": lower,
            "phase4_compute_microbatch_max_rows": lower,
        },
    ]
    if provider.name.startswith("gemma3_1b_"):
        decoder_contrast = 0 if provider.reference_decoder_cache_bytes else 4 * GIB
        replay_reference = 1 * GIB if provider.name.endswith("clt") else 4 * GIB
        rows.extend(
            (
                {
                    "calibration_case": "lower_phase1_batch",
                    "calibration_factor": "phase1_source_batch",
                    "calibration_pair": "phase1",
                    **reference,
                    "phase1_trace_batch_policy": "cap_effective_batches",
                    "phase1_trace_batch_size_max": lower,
                },
                {
                    "calibration_case": "lower_phase3_microbatch",
                    "calibration_factor": "phase3_microbatch",
                    "calibration_pair": "phase3",
                    **reference,
                    "phase3_compute_microbatch_max_rows": lower,
                },
                {
                    "calibration_case": "lower_phase4_microbatch",
                    "calibration_factor": "phase4_microbatch",
                    "calibration_pair": "phase4",
                    **reference,
                    "phase4_compute_microbatch_max_rows": lower,
                },
                {
                    "calibration_case": "decoder_cache_contrast",
                    "calibration_factor": "decoder_cache",
                    "calibration_pair": "decoder_cache",
                    **reference,
                    "cross_batch_decoder_cache_bytes": decoder_contrast,
                },
                {
                    "calibration_case": "replay_window_contrast",
                    "calibration_factor": "replay_window",
                    "calibration_pair": "replay_window",
                    **reference,
                    "chunked_feature_replay_window": 2,
                },
                {
                    "calibration_case": "prefetch_contrast",
                    "calibration_factor": "prefetch_depth",
                    "calibration_pair": "prefetch_depth",
                    **reference,
                    "error_vector_prefetch_lookahead": 0,
                },
                {
                    "calibration_case": "recompute_cache_zero",
                    "calibration_factor": "replay_tile_cache",
                    "calibration_pair": "recompute_cache",
                    **reference,
                    "governor_required_row_store_policy": "recompute",
                    "governor_required_replay_tile_cache_bytes": 0,
                    "governor_required_spill_target": None,
                },
                {
                    "calibration_case": "recompute_cache_reference",
                    "calibration_factor": "replay_tile_cache",
                    "calibration_pair": "recompute_cache",
                    **reference,
                    "governor_required_row_store_policy": "recompute",
                    "governor_required_replay_tile_cache_bytes": replay_reference,
                    "governor_required_spill_target": None,
                },
                {
                    "calibration_case": "tiled_policy",
                    "calibration_factor": "row_store_policy",
                    "calibration_pair": "row_store_policy",
                    **reference,
                    "governor_required_row_store_policy": "tiled",
                },
                {
                    "calibration_case": "eager_encoder_residency",
                    "calibration_factor": "encoder_residency",
                    "calibration_pair": "encoder_residency",
                    **reference,
                    "governor_required_encoder_residency": "eager",
                },
                {
                    "calibration_case": "scratch_spill",
                    "calibration_factor": "spill_target",
                    "calibration_pair": "spill_target",
                    **reference,
                    "governor_required_spill_target": "scratch",
                },
                {
                    "calibration_case": "reference_repeat",
                    "calibration_factor": "repeatability",
                    "calibration_pair": "reference",
                    "calibration_replicate": 2,
                    **reference,
                },
            )
        )
    else:
        rows.append(
            {
                "calibration_case": "tiled_policy",
                "calibration_factor": "row_store_policy",
                "calibration_pair": "row_store_policy",
                **reference,
                "governor_required_row_store_policy": "tiled",
            }
        )
    return rows


def build_governor_calibration_config(
    *,
    variant: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
) -> dict[str, Any]:
    try:
        provider = GOVERNOR_CALIBRATION_PROVIDERS[variant]
    except KeyError as error:
        known = ", ".join(sorted(GOVERNOR_CALIBRATION_PROVIDERS))
        raise ValueError(f"unknown governor calibration variant {variant!r}: {known}") from error

    fixture = resolve_fixture(
        GOVERNOR_CALIBRATION_FIXTURE,
        catalog_by_name=catalog_by_name,
        allow_fallback=True,
    )
    cache_root = model_cache_root or scratch_root.parent / "huggingface"
    provider_config = transcoder_config_to_json(
        resolve_transcoder_load_config(
            transcoder_provider_family=provider.provider_family,
            transcoder_cache_dir=str(cache_root),
        )
    )
    stage = f"governor_calibration_{variant}"
    output_root = scratch_root / "granite" / "sweep" / "governor_calibration" / variant
    defaults = {
        **base_trace_defaults(),
        **provider_config,
        "stage": stage,
        "cluster": "granite",
        "tier": "sweep",
        "resource_profile": "governor_calibration_h200",
        "governor_profile_name": provider.governor_profile,
        "attribution_batch_size": provider.logical_batch,
        "feature_batch_size": provider.logical_batch,
        "logit_batch_size": provider.logical_batch,
        "decoder_chunk_size": 4096,
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
        "host_budget_bytes": provider.host_memory_bytes,
        "file_cache_allowance_bytes": 64 * GIB,
        "local_disk_bytes": 64 * GIB,
        "scratch_disk_bytes": 64 * GIB,
        "walltime_seconds": provider.timeout_minutes * 60,
    }
    scenarios = []
    for row in _contrast_rows(provider):
        case = str(row["calibration_case"])
        scenarios.append(
            {
                "name": f"granite_governor_calibration_{variant}_{case}",
                "run_name": f"governor-calibration-{variant}-{case}",
                "run_goal": (
                    f"Measure {row['calibration_factor']} while holding the remaining "
                    "governor controls at the declared reference configuration."
                ),
                "calibration_stage": (
                    "causal_1b" if variant.startswith("gemma3_1b_") else "model_scaling"
                ),
                **fixture.to_source_payload(),
                **row,
                "governor_resource_envelope": envelope,
                "timeout_minutes": provider.timeout_minutes,
            }
        )
    return {
        "schema_version": 1,
        "defaults": defaults,
        "metadata": {
            "cluster": "granite",
            "stage": stage,
            "operational_class": "sweep",
            "resource_profile": "governor_calibration_h200",
            "immutable_validation_config": True,
            "recommended_output_root": str(output_root),
            "matrix_variant": variant,
            "matrix_case_count": len(scenarios),
            "fixed_protocol": {
                "hardware": "single_h200",
                "dtype": "fp32",
                "fixture": GOVERNOR_CALIBRATION_FIXTURE,
                "cache_state": "cold_campaign_start_then_recorded_order",
                "compact_parity_required": True,
            },
            "required_measurements": [
                "phase_elapsed_seconds",
                "phase_cuda_peak_allocated_bytes",
                "phase_cuda_peak_reserved_bytes",
                "phase_end_cuda_allocated_bytes",
                "host_rss_bytes",
                "decoder_cache_hits_misses",
                "replay_tile_cache_hits_misses_evictions",
                "row_store_bytes_read_written",
                "planning_predictions_and_support",
            ],
            "slurm": {
                "account": "rai",
                "partition": "rai-gpu-grn",
                "qos": "rai-gpu-grn",
                "gres": "gpu:h200:1",
                "mem": provider.slurm_memory,
                "time": provider.slurm_time,
            },
        },
        "scenarios": scenarios,
    }


def governor_calibration_scenario_file_name(*, variant: str) -> str:
    return f"exact_trace_governor_calibration_{variant}_granite_scenarios.json"


def write_governor_calibration_config(
    payload: dict[str, Any],
    *,
    variant: str,
    output_dir: Path = DEFAULT_GENERATED_DIR,
) -> Path:
    ensure_dir(output_dir)
    path = output_dir / governor_calibration_scenario_file_name(variant=variant)
    write_json(path, payload)
    return path


def write_all_governor_calibration_configs(
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
) -> tuple[Path, ...]:
    return tuple(
        write_governor_calibration_config(
            build_governor_calibration_config(
                variant=variant,
                scratch_root=scratch_root,
                model_cache_root=model_cache_root,
            ),
            variant=variant,
            output_dir=output_dir,
        )
        for variant in GOVERNOR_CALIBRATION_PROVIDERS
    )
