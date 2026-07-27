from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .transcoder_config import transcoder_config_to_json, TranscoderLoadConfig


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHPC_SCRATCH_ROOT = (
    Path("/scratch/general/vast")
    / os.environ.get("USER", "u1653998")
    / "nlp_research_project"
    / "exact_trace_bench"
)
DEFAULT_SCRATCH_ROOT = Path(
    os.environ.get("EXACT_TRACE_BENCH_SCRATCH_ROOT", DEFAULT_CHPC_SCRATCH_ROOT)
)
DEFAULT_GENERATED_DIR = REPO_ROOT / "experiments" / "generated" / "exact_trace_bench"
DEFAULT_EXTRACTED_DIR = REPO_ROOT / "experiments" / "extracted" / "exact_trace_bench"
DEFAULT_FIXTURE_CATALOG = (
    REPO_ROOT
    / "experiments"
    / "generated"
    / "weekend_exact_chunked_fixtures"
    / "fixture_catalog.json"
)
DEFAULT_WAVE0_FIXTURE_CATALOG = (
    REPO_ROOT
    / "experiments"
    / "generated"
    / "exact_trace_wave0_fixtures"
    / "fixture_catalog.json"
)
DEFAULT_WAVE0_FIXTURE_TARGET_SPEC = (
    REPO_ROOT / "experiments" / "exact_trace_wave0_fixture_targets.json"
)
DEFAULT_WAVE0_FIXTURE_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "generated" / "exact_trace_wave0_fixtures"
)
DEFAULT_WAVE0_BASELINE_REGISTRY = (
    DEFAULT_SCRATCH_ROOT / "baselines" / "wave0-baseline-20260520-01.json"
)
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"


def base_trace_defaults() -> dict[str, Any]:
    return {
        **transcoder_config_to_json(TranscoderLoadConfig()),
        "completions": 1,
        "temperature": 0.0,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "verbose_attribution": False,
        "profile_attribution": False,
        "profile_log_interval": 1,
        "attribution_update_interval": 4,
        "save_raw": False,
        "no_offload": False,
        "no_lazy_encoder": False,
        "no_lazy_decoder": False,
        "chunked_feature_replay_window": None,
        "error_vector_prefetch_lookahead": None,
        "stage_encoder_vecs_on_cpu": None,
        "stage_error_vectors_on_cpu": None,
        "row_subchunk_size": None,
        "nnsight_session_capacity": None,
        "phase3_compute_microbatch_max_rows": None,
        "phase4_execution_batch_max_rows": None,
        "feature_vjp_tape_batch_window": 1,
        "feature_vjp_tape_max_bytes": 0,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": False,
        "decoder_active_row_max_bytes": 0,
        "phase0_decoder_row_ranges": False,
        "diagnostic_stop_mode": "none",
        "diagnostic_stop_phase4_batches": None,
        "full_retention_backend": "full_file",
        "feature_row_column_tile_size": 2048,
        "influence_row_tile_size": 4096,
        "influence_column_tile_size": 2048,
        "feature_row_retention": "full_file",
        "replay_tile_cache_bytes": None,
        "exact_encoder_residency": "lazy",
        "exact_trace_internal_dtype": "fp32",
        "governor_admission_mode": "enforce",
        "phase0_activation_threshold_compare_mode": "baseline",
        "plan_feature_batch_size": False,
        "auto_scale_feature_batch_size": False,
        "feature_batch_size_max": None,
        "feature_batch_target_reserved_fraction": 0.9,
        "feature_batch_min_free_fraction": 0.05,
        "feature_batch_probe_batches": 1,
        "phase4_anomaly_debug": False,
        "phase4_scheduler_mode": "locality",
        "phase4_scheduler_debug": False,
        "phase4_scheduler_telemetry_detail": "normal",
        "phase4_refresh_optimization": "v1",
        "phase4_refresh_prepared_chunk_cache_bytes": 0,
        "phase4_refresh_active_row_accumulation": "direct_v1",
        "phase4_row_executor": "batched",
        "phase4_row_reduction": "gpu_v1",
        "row_store_cache_control": "fadvise_dontneed_after_append_and_read_v1",
        "row_store_preallocate": True,
        "trajectory_session_mode": "per_token",
        "reuse_phase0_window_state": False,
        "reuse_target_logits": False,
        "phase0_window_scope": "shard_window",
        "phase0_window_max_prefix_policy": "max_target_position",
        "phase0_window_reference_checks": "off",
        "cross_cluster_debug": False,
        "telemetry_max_events": None,
        "incremental_telemetry_jsonl": False,
        "max_steps": 1,
        "method": "exact",
    }


def gib_to_bytes(value_gib: int) -> int:
    return int(value_gib) * (1024**3)


def recommended_output_root(
    *,
    cluster: str,
    tier: str,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    return scratch_root / cluster / tier
