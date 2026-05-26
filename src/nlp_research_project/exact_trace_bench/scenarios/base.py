from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import base_trace_defaults, gib_to_bytes, recommended_output_root
from ..fixtures import FixtureRef, resolve_fixture


SCENARIO_TIERS = ("fast", "anomaly", "long_eval")
WAVE2A_PHASE1_TIERS = ("fast", "anomaly")
WAVE2B_PHASE4_TIERS = ("fast", "anomaly")
WAVE2C_ROW_ENCODER_TIERS = ("fast", "anomaly")
WAVE3_INTERACTION_CONFIRMATION_TIERS = ("fast", "anomaly")
WAVE4_GENERALIZATION_TIERS = ("fast", "anomaly", "long_eval")

WAVE0_NEW_GSM8K_INDICES: tuple[int, ...] = (
    18,
    57,
    103,
    214,
    318,
    462,
    579,
    702,
    915,
    1046,
    1201,
    1289,
)
WAVE0_NEW_LATE_GSM8K_INDICES: tuple[int, ...] = (
    214,
    462,
    702,
    1046,
    1201,
    1289,
)
WAVE0_CANONICAL_FAST_FIXTURES: tuple[str, ...] = ("828_base", "361_base")
WAVE0_CANONICAL_ANOMALY_FIXTURES: tuple[str, ...] = ("94_base",)
WAVE0_CANONICAL_LATE_FIXTURES: tuple[str, ...] = (
    "828_late",
    "361_late",
    "94_late",
)
WAVE0_NEW_BASE_FIXTURES: tuple[str, ...] = tuple(
    f"{index}_base" for index in WAVE0_NEW_GSM8K_INDICES
)
WAVE0_NEW_LATE_FIXTURES: tuple[str, ...] = tuple(
    f"{index}_late" for index in WAVE0_NEW_LATE_GSM8K_INDICES
)
WAVE0_REPEAT_COUNT = 2
WAVE2A_PHASE1_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "legacy_default",
        "phase1_trace_batch_policy": "legacy",
    },
    {
        "label": "cap64",
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 64,
    },
    {
        "label": "cap16",
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 16,
    },
    {
        "label": "cap1",
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 1,
    },
)
WAVE2B_PHASE4_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "legacy_default",
        "phase4_scheduler_mode": "locality",
        "phase4_refresh_policy": "standard",
        "phase4_ranker": "argsort",
        "phase4_refresh_optimization": "off",
        "phase4_row_executor": "batched",
    },
    {
        "label": "planner_v1",
        "phase4_scheduler_mode": "planner_v1",
    },
    {
        "label": "planner_v2",
        "phase4_scheduler_mode": "planner_v2",
    },
    {
        "label": "deferred_v1",
        "phase4_refresh_policy": "deferred_v1",
    },
    {
        "label": "topk_v1",
        "phase4_ranker": "topk_v1",
    },
    {
        "label": "refresh_opt_v1",
        "phase4_refresh_optimization": "v1",
    },
    {
        "label": "streaming_v1",
        "phase4_row_executor": "streaming_v1",
    },
)
WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS: dict[str, Any] = {
    "row_store_cache_control": "off",
    "exact_encoder_residency": "lazy",
    "phase4_scheduler_mode": "locality",
    "phase4_refresh_policy": "standard",
    "phase4_ranker": "argsort",
    "phase4_refresh_optimization": "off",
    "phase4_row_executor": "batched",
    "phase1_trace_batch_policy": "legacy",
}
WAVE2C_ROW_ENCODER_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "legacy_default",
        **WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS,
    },
    {
        "label": "row_fadvise",
        "row_store_cache_control": "fadvise_dontneed_after_append_v1",
    },
    {
        "label": "active_cpu_encoder",
        "exact_encoder_residency": "active_cpu",
    },
    {
        "label": "active_pinned_cpu_encoder",
        "exact_encoder_residency": "active_pinned_cpu",
    },
    {
        "label": "no_cpu_staging",
        "stage_encoder_vecs_on_cpu": False,
        "stage_error_vectors_on_cpu": False,
    },
    {
        "label": "row_subchunk_512",
        "row_subchunk_size": 512,
    },
    {
        "label": "feature_batch_planner",
        "plan_feature_batch_size": True,
    },
)
WAVE3_INTERACTION_CONFIRMATION_LEGACY_DEFAULTS: dict[str, Any] = {
    **WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS,
    "row_subchunk_size": None,
    "plan_feature_batch_size": False,
}
WAVE3_INTERACTION_CONFIRMATION_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "baseline",
    },
    {
        "label": "deferred_v1",
        "phase4_refresh_policy": "deferred_v1",
    },
    {
        "label": "row_subchunk_512",
        "row_subchunk_size": 512,
    },
    {
        "label": "plan_feature_batch_size",
        "plan_feature_batch_size": True,
    },
    {
        "label": "deferred_v1_row_subchunk_512",
        "phase4_refresh_policy": "deferred_v1",
        "row_subchunk_size": 512,
    },
    {
        "label": "deferred_v1_plan_feature_batch_size",
        "phase4_refresh_policy": "deferred_v1",
        "plan_feature_batch_size": True,
    },
)
WAVE3_OPTIONAL_SPEED_INTERACTION_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "deferred_v1_streaming_v1_row_subchunk_512",
        "phase4_refresh_policy": "deferred_v1",
        "phase4_row_executor": "streaming_v1",
        "row_subchunk_size": 512,
    },
)
WAVE4_GENERALIZATION_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "baseline",
    },
    {
        "label": "row_subchunk_512",
        "row_subchunk_size": 512,
    },
    {
        "label": "plan_feature_batch_size",
        "plan_feature_batch_size": True,
    },
)

RESOURCE_PROFILE_STANDARD = "standard"
RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM = "long_eval_high_mem"

STABLE_PUBLIC_SCENARIO_KEYS = ("exact_trace_internal_dtype",)

ADVANCED_PUBLIC_TUNING_KEYS = (
    "chunked_feature_replay_window",
    "error_vector_prefetch_lookahead",
    "stage_encoder_vecs_on_cpu",
    "stage_error_vectors_on_cpu",
    "row_subchunk_size",
    "phase1_trace_batch_policy",
    "phase1_trace_batch_size_max",
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
    "phase4_scheduler_debug",
    "phase4_scheduler_telemetry_detail",
    "phase4_refresh_optimization",
    "phase4_row_executor",
)

DEBUG_REPLAY_PUBLIC_KEYS = (
    "phase0_activation_threshold_compare_mode",
    "phase4_anomaly_debug",
    "cross_cluster_debug",
    "capture_phase0_donor_bundle",
    "phase0_donor_bundle",
    "phase0_replay_mode",
    "phase0_donor_context_policy",
    "phase3_gradient_donor_bundle",
    "phase3_gradient_replay_mode",
    "phase3_row_donor_bundle",
    "phase3_row_replay_mode",
    "phase3_replay_validation_policy",
    "capture_phase3_seed_bundle",
    "capture_phase3_gradient_bundle",
    "capture_phase3_row_bundle",
    "capture_feature_semantic_descriptors",
    "semantic_descriptor_top_k",
    "semantic_descriptor_dim",
)

TELEMETRY_KEYS = ("telemetry_max_events",)

DEPRECATED_COMPAT_KEYS = ("auto_scale_feature_batch_size",)

PRIVATE_INTERNAL_KEYS: tuple[str, ...] = ()

EXACT_MODE_KNOB_KEYS = (
    STABLE_PUBLIC_SCENARIO_KEYS
    + ADVANCED_PUBLIC_TUNING_KEYS
    + DEBUG_REPLAY_PUBLIC_KEYS
    + TELEMETRY_KEYS
    + DEPRECATED_COMPAT_KEYS
    + PRIVATE_INTERNAL_KEYS
)

if len(EXACT_MODE_KNOB_KEYS) != len(set(EXACT_MODE_KNOB_KEYS)):
    raise RuntimeError("Duplicate exact-mode scenario knob classification")

CLUSTER_SETTINGS: dict[str, dict[str, Any]] = {
    "ascend": {
        "fast": {
            "batch": 128,
            "chunk": 2048,
            "cache_gib": 0,
        },
        "anomaly": {
            "batch": 256,
            "chunk": 4096,
            "cache_gib": 0,
        },
        "long_eval": {
            "runs": [
                {
                    "label": "no_cache",
                    "batch": 128,
                    "chunk": 2048,
                    "cache_gib": 0,
                    "fixtures": ("361_late", "828_late", "94_late"),
                },
                {
                    "label": "cache_probe",
                    "batch": 128,
                    "chunk": 2048,
                    "cache_gib": 8,
                    "fixtures": ("361_late",),
                },
            ],
        },
    },
    "cardinal": {
        "fast": {
            "batch": 128,
            "chunk": 4096,
            "cache_gib": 0,
        },
        "anomaly": {
            "batch": 256,
            "chunk": 4096,
            "cache_gib": 0,
        },
        "long_eval": {
            "runs": [
                {
                    "label": "no_cache",
                    "batch": 256,
                    "chunk": 4096,
                    "cache_gib": 0,
                    "fixtures": ("361_late", "828_late", "94_late"),
                },
                {
                    "label": "cache_probe",
                    "batch": 256,
                    "chunk": 4096,
                    "cache_gib": 8,
                    "fixtures": ("361_late",),
                },
            ],
        },
    },
}


def _require_cluster(cluster: str) -> None:
    if cluster not in CLUSTER_SETTINGS:
        raise ValueError(f"Unsupported cluster '{cluster}'")


def _scenario_name(
    *,
    cluster: str,
    tier: str,
    fixture_name: str,
    batch: int,
    chunk: int,
    cache_gib: int,
    label: str | None = None,
) -> str:
    label_slug = "" if not label else f"_{label}"
    return (
        f"{cluster}_{tier}{label_slug}_{fixture_name}"
        f"_b{batch:03d}_c{chunk}_cache{cache_gib}g"
    )


def _select_exact_mode_knobs(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: source[key]
        for key in EXACT_MODE_KNOB_KEYS
        if key in source and source[key] is not None
    }


def _resolve_required_fixtures(
    fixture_names: tuple[str, ...],
    *,
    catalog_by_name: dict[str, dict[str, Any]],
) -> list[FixtureRef]:
    missing = [name for name in fixture_names if name not in catalog_by_name]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(
            "Wave 0 scenario generation requires a fixture catalog containing "
            f"all requested fixtures. Missing: {joined}"
        )
    return [
        resolve_fixture(
            fixture_name,
            catalog_by_name=catalog_by_name,
            allow_fallback=False,
        )
        for fixture_name in fixture_names
    ]


def _append_exact_scenario(
    payload: dict[str, Any],
    *,
    cluster: str,
    tier: str,
    stage: str,
    fixture: FixtureRef,
    batch: int,
    chunk: int,
    cache_gib: int,
    resource_profile: str,
    scratch_root: Path,
    label: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    row = {
        "name": _scenario_name(
            cluster=cluster,
            tier=tier,
            fixture_name=fixture.fixture_name,
            batch=batch,
            chunk=chunk,
            cache_gib=cache_gib,
            label=label,
        ),
        "stage": stage,
        "cluster": cluster,
        "resource_profile": resource_profile,
        "recommended_output_root": str(scratch_root / cluster / tier),
        **fixture.to_source_payload(),
        "attribution_batch_size": batch,
        "feature_batch_size": batch,
        "logit_batch_size": batch,
        "decoder_chunk_size": chunk,
        "cross_batch_decoder_cache_bytes": gib_to_bytes(cache_gib),
    }
    if extra:
        row.update(extra)
    payload["scenarios"].append(row)


def _base_payload(
    *,
    cluster: str,
    tier: str,
    notes: list[str],
    scratch_root: Path,
    resource_profile: str,
) -> dict[str, Any]:
    return {
        "defaults": base_trace_defaults(),
        "metadata": {
            "cluster": cluster,
            "stage": f"exact_trace_bench_{tier}",
            "tier": tier,
            "resource_profile": resource_profile,
            "recommended_output_root": str(
                recommended_output_root(
                    cluster=cluster,
                    tier=tier,
                    scratch_root=scratch_root,
                )
            ),
            "notes": notes,
        },
        "scenarios": [],
    }
