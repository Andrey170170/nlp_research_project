from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCRATCH_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/exact_trace_bench")
DEFAULT_GENERATED_DIR = REPO_ROOT / "experiments" / "generated" / "exact_trace_bench"
DEFAULT_EXTRACTED_DIR = REPO_ROOT / "experiments" / "extracted" / "exact_trace_bench"
DEFAULT_FIXTURE_CATALOG = (
    REPO_ROOT
    / "experiments"
    / "generated"
    / "weekend_exact_chunked_fixtures"
    / "fixture_catalog.json"
)
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"


def base_trace_defaults() -> dict[str, Any]:
    return {
        "completions": 1,
        "temperature": 0.0,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "verbose_attribution": True,
        "profile_attribution": True,
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
        "exact_trace_internal_dtype": "fp32",
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
        "phase4_refresh_optimization": "off",
        "phase4_row_executor": "batched",
        "cross_cluster_debug": False,
        "telemetry_max_events": None,
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
