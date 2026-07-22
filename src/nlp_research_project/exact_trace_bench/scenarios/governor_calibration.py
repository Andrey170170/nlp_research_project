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
GOVERNOR_CALIBRATION_RESOURCE_PROFILE = "governor_calibration_h200"
GOVERNOR_CALIBRATION_BASELINE_REGISTRY = Path(
    "experiments/baselines/governor_calibration_granite_20260719.json"
)


@dataclass(frozen=True)
class GovernorCalibrationProvider:
    name: str
    provider_family: str
    governor_profile: str
    baseline_registry_key: str
    reference_logical_batch: int
    reference_decoder_cache_bytes: int
    logical_batch_ladder: tuple[int, ...]
    decoder_fetch_chunk_ladder: tuple[int, ...]
    decoder_cache_ladder_bytes: tuple[int, ...]
    coupled_corners: tuple[tuple[int, int, int], ...]
    semantic_axis_batch: int
    refresh_cadences: tuple[int, ...]
    host_memory_bytes: int
    timeout_minutes: int
    slurm_memory: str
    slurm_time: str
    calibration_wave: str = "A"
    execution_batch_ladder: tuple[int, ...] = ()


GOVERNOR_CALIBRATION_PROVIDERS = {
    provider.name: provider
    for provider in (
        GovernorCalibrationProvider(
            name="gemma3_1b_clt",
            provider_family="gemmascope2-clt-1b-medium-affine",
            governor_profile="granite_h200_1b_clt_b1000_c4096_cache8",
            baseline_registry_key="governor_calibration/gemma3_1b_clt/361_base",
            reference_logical_batch=1000,
            reference_decoder_cache_bytes=8 * GIB,
            logical_batch_ladder=(1000, 1500, 2000, 3000, 4096),
            decoder_fetch_chunk_ladder=(4096, 8192, 10080),
            decoder_cache_ladder_bytes=(0, 8 * GIB, 16 * GIB, 32 * GIB),
            coupled_corners=(
                (2000, 8192, 16 * GIB),
                (3000, 8192, 32 * GIB),
                (4096, 10080, 32 * GIB),
            ),
            semantic_axis_batch=1500,
            refresh_cadences=(),
            host_memory_bytes=200 * GIB,
            timeout_minutes=60,
            slurm_memory="200G",
            slurm_time="01:00:00",
        ),
        GovernorCalibrationProvider(
            name="gemma3_1b_plt",
            provider_family="gemmascope2-plt-1b-small",
            governor_profile="granite_h200_1b_plt_b128_c4096_cache0",
            baseline_registry_key="governor_calibration/gemma3_1b_plt/361_base",
            reference_logical_batch=128,
            reference_decoder_cache_bytes=0,
            logical_batch_ladder=(128, 256, 512, 768, 1024, 1536),
            decoder_fetch_chunk_ladder=(4096, 8192, 16384, 32768),
            decoder_cache_ladder_bytes=(0, 4 * GIB),
            coupled_corners=(
                (512, 8192, 0),
                (1024, 16384, 0),
                (1536, 32768, 0),
            ),
            semantic_axis_batch=256,
            refresh_cadences=(2, 8),
            host_memory_bytes=200 * GIB,
            timeout_minutes=120,
            slurm_memory="200G",
            slurm_time="02:00:00",
        ),
        GovernorCalibrationProvider(
            name="gemma3_4b_plt",
            provider_family="gemmascope2-plt-4b-small",
            governor_profile="granite_h200_4b_plt_b128_c4096_cache0",
            baseline_registry_key="governor_calibration/gemma3_4b_plt/361_base",
            reference_logical_batch=128,
            reference_decoder_cache_bytes=0,
            logical_batch_ladder=(),
            decoder_fetch_chunk_ladder=(),
            decoder_cache_ladder_bytes=(),
            coupled_corners=(),
            semantic_axis_batch=256,
            refresh_cadences=(),
            host_memory_bytes=400 * GIB,
            timeout_minutes=120,
            slurm_memory="400G",
            slurm_time="02:00:00",
            calibration_wave="B",
            execution_batch_ladder=(128, 256, 512),
        ),
        GovernorCalibrationProvider(
            name="gemma3_12b_plt",
            provider_family="gemmascope2-plt-12b-small",
            governor_profile="granite_h200_12b_plt_b64_c4096_cache0",
            baseline_registry_key="governor_calibration/gemma3_12b_plt/361_base",
            reference_logical_batch=64,
            reference_decoder_cache_bytes=0,
            logical_batch_ladder=(),
            decoder_fetch_chunk_ladder=(),
            decoder_cache_ladder_bytes=(),
            coupled_corners=(),
            semantic_axis_batch=128,
            refresh_cadences=(),
            host_memory_bytes=600 * GIB,
            timeout_minutes=480,
            slurm_memory="600G",
            slurm_time="08:00:00",
            calibration_wave="B",
            execution_batch_ladder=(64, 128, 256),
        ),
    )
}


def _physical_controls(
    provider: GovernorCalibrationProvider,
    *,
    batch: int | None = None,
    decoder_cache_bytes: int | None = None,
) -> dict[str, Any]:
    capacity = provider.reference_logical_batch if batch is None else batch
    cache_bytes = (
        provider.reference_decoder_cache_bytes
        if decoder_cache_bytes is None
        else decoder_cache_bytes
    )
    return {
        "nnsight_session_capacity": capacity,
        "phase1_trace_batch_policy": (
            "legacy"
            if capacity == provider.reference_logical_batch
            else "cap_effective_batches"
        ),
        "phase1_trace_batch_size_max": (
            None if capacity == provider.reference_logical_batch else capacity
        ),
        "phase3_compute_microbatch_max_rows": capacity,
        "phase4_execution_batch_max_rows": capacity,
        "cross_batch_decoder_cache_bytes": cache_bytes,
        "chunked_feature_replay_window": 4,
        "error_vector_prefetch_lookahead": 2,
        "governor_required_encoder_residency": "lazy",
        "governor_required_row_store_policy": "file_backed_full",
        "governor_required_replay_tile_cache_bytes": 0,
        "governor_required_spill_target": "local",
    }


def _row(
    provider: GovernorCalibrationProvider,
    *,
    case: str,
    factor: str,
    research_fields: tuple[str, ...] = (),
    **overrides: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "calibration_case": case,
        "calibration_factor": factor,
        "calibration_pair": factor,
        "governor_fidelity_mode": "research" if research_fields else "exact",
        **_physical_controls(provider),
        **overrides,
    }
    if research_fields:
        row["governor_fidelity_override_fields"] = sorted(research_fields)
    return row


def _coupled_batch_row(
    provider: GovernorCalibrationProvider,
    *,
    batch: int,
    case: str | None = None,
    decoder_chunk_size: int = 4096,
    decoder_cache_bytes: int | None = None,
) -> dict[str, Any]:
    is_reference = (
        batch == provider.reference_logical_batch
        and decoder_chunk_size == 4096
        and (
            decoder_cache_bytes is None
            or decoder_cache_bytes == provider.reference_decoder_cache_bytes
        )
    )
    return _row(
        provider,
        case=case or ("reference" if is_reference else f"logical_batch_b{batch}"),
        factor="reference" if is_reference else "logical_batch",
        research_fields=(
            ()
            if is_reference
            else ("source_batch_size", "feature_batch_size", "logit_batch_size")
        ),
        **_physical_controls(
            provider,
            batch=batch,
            decoder_cache_bytes=decoder_cache_bytes,
        ),
        attribution_batch_size=batch,
        feature_batch_size=batch,
        logit_batch_size=batch,
        decoder_chunk_size=decoder_chunk_size,
    )


def _wave_a_rows(provider: GovernorCalibrationProvider) -> list[dict[str, Any]]:
    rows = [
        _coupled_batch_row(provider, batch=batch)
        for batch in provider.logical_batch_ladder
    ]

    rows.extend(
        _row(
            provider,
            case=f"decoder_chunk_c{chunk}",
            factor="decoder_fetch_chunk",
            decoder_chunk_size=chunk,
        )
        for chunk in provider.decoder_fetch_chunk_ladder
        if chunk != 4096
    )
    rows.extend(
        _row(
            provider,
            case=f"decoder_cache_g{cache_bytes // GIB}",
            factor="decoder_cache",
            cross_batch_decoder_cache_bytes=cache_bytes,
        )
        for cache_bytes in provider.decoder_cache_ladder_bytes
        if cache_bytes != provider.reference_decoder_cache_bytes
    )
    rows.extend(
        _coupled_batch_row(
            provider,
            batch=batch,
            case=(
                f"coupled_b{batch}_c{chunk}_cacheg{cache_bytes // GIB}"
            ),
            decoder_chunk_size=chunk,
            decoder_cache_bytes=cache_bytes,
        )
        for batch, chunk, cache_bytes in provider.coupled_corners
    )

    axis_batch = provider.semantic_axis_batch
    reference_batch = provider.reference_logical_batch
    rows.extend(
        (
            _row(
                provider,
                case=f"semantic_source_b{axis_batch}",
                factor="semantic_axis",
                research_fields=("source_batch_size",),
                attribution_batch_size=axis_batch,
                nnsight_session_capacity=axis_batch,
                phase1_trace_batch_policy="cap_effective_batches",
                phase1_trace_batch_size_max=axis_batch,
            ),
            _row(
                provider,
                case=f"semantic_feature_b{axis_batch}",
                factor="semantic_axis",
                research_fields=("feature_batch_size",),
                feature_batch_size=axis_batch,
                nnsight_session_capacity=axis_batch,
                phase4_execution_batch_max_rows=axis_batch,
            ),
            _row(
                provider,
                case=f"semantic_logit_b{axis_batch}",
                factor="semantic_axis",
                research_fields=("logit_batch_size",),
                logit_batch_size=axis_batch,
                nnsight_session_capacity=axis_batch,
                phase3_compute_microbatch_max_rows=axis_batch,
            ),
        )
    )
    rows.extend(
        _row(
            provider,
            case=f"semantic_refresh_g{cadence}",
            factor="refresh_cadence",
            research_fields=("frontier_refresh_stride",),
            attribution_update_interval=cadence,
        )
        for cadence in provider.refresh_cadences
    )
    rows.append(
        _row(
            provider,
            case="reference_repeat",
            factor="repeatability",
            calibration_pair="reference",
            calibration_replicate=2,
        )
    )

    # The defaults carry the reference semantics. Keep the explicit reference
    # batch local only where it is needed to make an axis-isolation row obvious.
    for row in rows:
        row.setdefault("attribution_batch_size", reference_batch)
        row.setdefault("feature_batch_size", reference_batch)
        row.setdefault("logit_batch_size", reference_batch)
        row.setdefault("decoder_chunk_size", 4096)
    return rows


def _wave_b_rows(provider: GovernorCalibrationProvider) -> list[dict[str, Any]]:
    reference_batch, midpoint_batch, high_batch = provider.execution_batch_ladder

    def row(
        *,
        case: str,
        factor: str,
        execution_batch: int = reference_batch,
        decoder_chunk_size: int = 4096,
        feature_batch_size: int = reference_batch,
        update_interval: int = 4,
        research_fields: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if decoder_chunk_size != 4096:
            # Fetch chunking changes the decoder reduction grouping as well as
            # the physical fetch size. Declare the exact semantic field rather
            # than allowing a non-reference chunk under strict fidelity.
            overrides["row_subchunk_size"] = decoder_chunk_size
        return _row(
            provider,
            case=case,
            factor=factor,
            research_fields=research_fields,
            attribution_batch_size=reference_batch,
            feature_batch_size=feature_batch_size,
            logit_batch_size=reference_batch,
            attribution_update_interval=update_interval,
            decoder_chunk_size=decoder_chunk_size,
            cross_batch_decoder_cache_bytes=0,
            nnsight_session_capacity=execution_batch,
            phase1_trace_batch_policy="cap_effective_batches",
            phase1_trace_batch_size_max=reference_batch,
            phase3_compute_microbatch_max_rows=reference_batch,
            phase4_execution_batch_max_rows=execution_batch,
            **overrides,
        )

    return [
        row(case="reference", factor="reference"),
        row(
            case=f"physical_envelope_b{midpoint_batch}",
            factor="physical_envelope",
            execution_batch=midpoint_batch,
        ),
        row(
            case=f"physical_envelope_b{high_batch}",
            factor="physical_envelope",
            execution_batch=high_batch,
        ),
        row(
            case="decoder_chunk_c16384",
            factor="decoder_fetch_chunk",
            decoder_chunk_size=16384,
            research_fields=("decoder_reduction_tile",),
        ),
        row(
            case="decoder_chunk_c32768",
            factor="decoder_fetch_chunk",
            decoder_chunk_size=32768,
            research_fields=("decoder_reduction_tile",),
        ),
        row(
            case=f"coupled_execution_b{midpoint_batch}_c32768",
            factor="physical_execution_decoder_chunk",
            execution_batch=midpoint_batch,
            decoder_chunk_size=32768,
            research_fields=("decoder_reduction_tile",),
        ),
        row(
            case=(
                f"semantic_feature_b{provider.semantic_axis_batch}"
                f"_execution_b{midpoint_batch}"
            ),
            factor="semantic_grouping_transfer",
            execution_batch=midpoint_batch,
            feature_batch_size=provider.semantic_axis_batch,
            update_interval=2,
            research_fields=("feature_batch_size", "frontier_refresh_stride"),
        ),
    ]


def build_governor_calibration_config(
    *,
    variant: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
    baseline_registry: Path = GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
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
    wave = provider.calibration_wave.lower()
    stage = f"governor_calibration_wave_{wave}_{variant}"
    output_root = (
        scratch_root
        / "granite"
        / "sweep"
        / "governor_calibration"
        / f"wave_{wave}"
        / variant
    )
    defaults = {
        **base_trace_defaults(),
        **provider_config,
        "stage": stage,
        "cluster": "granite",
        "tier": "sweep",
        "resource_profile": GOVERNOR_CALIBRATION_RESOURCE_PROFILE,
        "governor_profile_name": provider.governor_profile,
        "governor_admission_mode": "advisory",
        "attribution_batch_size": provider.reference_logical_batch,
        "feature_batch_size": provider.reference_logical_batch,
        "logit_batch_size": provider.reference_logical_batch,
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
    rows = (
        _wave_a_rows(provider)
        if provider.calibration_wave == "A"
        else _wave_b_rows(provider)
    )
    for row in rows:
        case = str(row["calibration_case"])
        scenario = {
            "name": f"granite_governor_calibration_wave_{wave}_{variant}_{case}",
            "run_name": f"governor-calibration-wave-{wave}-{variant}-{case}",
            "run_goal": (
                (
                    f"Measure {row['calibration_factor']} in the Wave A upward search "
                    "while recording admission, utilization, timing, and graph parity."
                )
                if provider.calibration_wave == "A"
                else (
                    f"Measure {row['calibration_factor']} in the Wave B calibration "
                    "while recording admission, utilization, timing, and graph parity."
                )
            ),
            "calibration_stage": (
                "wave_a_upward_1b"
                if provider.calibration_wave == "A"
                else "wave_b_independent_semantic_physical"
            ),
            **fixture.to_source_payload(),
            **row,
            "governor_resource_envelope": envelope,
            "timeout_minutes": provider.timeout_minutes,
        }
        scenario["baseline_check"] = {
            "enabled": True,
            "mode": "metrics",
            "registry_key": provider.baseline_registry_key,
            "baseline_required": True,
        }
        scenario["calibration_campaign"] = {
            "campaign_id": stage,
            "reference": {
                "kind": "baseline_registry",
                "registry_key": provider.baseline_registry_key,
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
            "calibration_wave": provider.calibration_wave,
            "governor_admission_policy": {
                "mode": "advisory",
                "scope": f"wave_{wave}_calibration_only",
                "reason": (
                    "Deliberate force override so refused configurations still execute "
                    "and produce calibration evidence; ordinary launches remain enforce."
                ),
            },
            "resource_profile": GOVERNOR_CALIBRATION_RESOURCE_PROFILE,
            "array_concurrency": 1,
            "array_concurrency_reason": (
                "Serial execution preserves inspectable cache order and isolates "
                "per-row resource measurements; baseline scoring uses a pinned "
                "registry and does not depend on array task order."
            ),
            "baseline_registry": str(baseline_registry),
            **(
                {"baseline_registry_pinned": True}
                if provider.calibration_wave == "B"
                else {}
            ),
            "fail_on_baseline_missing": True,
            "fail_on_validation_fail": True,
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
                **(
                    {"baseline_reference": "pinned_corrected_hook_registry"}
                    if provider.calibration_wave == "B"
                    else {}
                ),
            },
            "stop_rules": {
                "hard": [
                    "cuda_oom",
                    "peak_vram_fraction_at_least_0.90",
                ],
                "soft": [
                    "peak_vram_fraction_at_least_0.85",
                    "less_than_3_percent_runtime_improvement_for_two_steps",
                    "gpu_utilization_plateau",
                    "semantic_drift_exceeds_declared_threshold",
                ],
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
                "compact_graph_parity",
                *(
                    ["gpu_utilization_time_series"]
                    if provider.calibration_wave == "B"
                    else []
                ),
            ],
            "slurm": {
                "account": "rai",
                "partition": "rai-gpu-grn",
                "qos": (
                    "rai-gpu-grn-short"
                    if provider.calibration_wave == "A"
                    else "rai-gpu-grn"
                ),
                "gres": "gpu:h200:1",
                "cpus_per_task": 12,
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
    baseline_registry: Path = GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
) -> tuple[Path, ...]:
    return tuple(
        write_governor_calibration_config(
            build_governor_calibration_config(
                variant=variant,
                scratch_root=scratch_root,
                model_cache_root=model_cache_root,
                baseline_registry=baseline_registry,
            ),
            variant=variant,
            output_dir=output_dir,
        )
        for variant in GOVERNOR_CALIBRATION_PROVIDERS
    )
