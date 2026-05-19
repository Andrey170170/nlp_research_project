from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    DEFAULT_SCRATCH_ROOT,
    base_trace_defaults,
    gib_to_bytes,
    recommended_output_root,
)
from .fixtures import load_fixture_catalog, resolve_tier_fixtures
from .io_utils import ensure_dir, write_json


SCENARIO_TIERS = ("fast", "anomaly", "long_eval")

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


def build_fast_tier(
    *,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    _require_cluster(cluster)
    cfg = CLUSTER_SETTINGS[cluster]["fast"]
    fixtures = resolve_tier_fixtures("base", catalog_by_name=catalog_by_name)
    payload = _base_payload(
        cluster=cluster,
        tier="fast",
        notes=[
            "Quick sanity tier across canonical base fixtures.",
            "Single no-cache exact run per fixture for quick smoke checks.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )

    for fixture in fixtures:
        payload["scenarios"].append(
            {
                "name": _scenario_name(
                    cluster=cluster,
                    tier="fast",
                    fixture_name=fixture.fixture_name,
                    batch=cfg["batch"],
                    chunk=cfg["chunk"],
                    cache_gib=cfg["cache_gib"],
                ),
                "stage": "exact_trace_bench_fast",
                "cluster": cluster,
                "resource_profile": RESOURCE_PROFILE_STANDARD,
                "recommended_output_root": str(scratch_root / cluster / "fast"),
                **fixture.to_source_payload(),
                "attribution_batch_size": cfg["batch"],
                "feature_batch_size": cfg["batch"],
                "logit_batch_size": cfg["batch"],
                "decoder_chunk_size": cfg["chunk"],
                "cross_batch_decoder_cache_bytes": gib_to_bytes(cfg["cache_gib"]),
                **_select_exact_mode_knobs(cfg),
            }
        )

    return payload


def build_anomaly_tier(
    *,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    _require_cluster(cluster)
    cfg = CLUSTER_SETTINGS[cluster]["anomaly"]
    fixtures = resolve_tier_fixtures("anomaly", catalog_by_name=catalog_by_name)
    payload = _base_payload(
        cluster=cluster,
        tier="anomaly",
        notes=[
            "Prompt-94-focused tier for anomaly reproduction and diagnosis.",
            "This tier is anomaly-watch only and should not be used as a general tuning benchmark.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )

    for fixture in fixtures:
        payload["scenarios"].append(
            {
                "name": _scenario_name(
                    cluster=cluster,
                    tier="anomaly",
                    fixture_name=fixture.fixture_name,
                    batch=cfg["batch"],
                    chunk=cfg["chunk"],
                    cache_gib=cfg["cache_gib"],
                ),
                "stage": "exact_trace_bench_anomaly",
                "cluster": cluster,
                "resource_profile": RESOURCE_PROFILE_STANDARD,
                "recommended_output_root": str(scratch_root / cluster / "anomaly"),
                **fixture.to_source_payload(),
                "attribution_batch_size": cfg["batch"],
                "feature_batch_size": cfg["batch"],
                "logit_batch_size": cfg["batch"],
                "decoder_chunk_size": cfg["chunk"],
                "cross_batch_decoder_cache_bytes": gib_to_bytes(cfg["cache_gib"]),
                **_select_exact_mode_knobs(cfg),
            }
        )

    return payload


def build_long_eval_tier(
    *,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    _require_cluster(cluster)
    run_specs = CLUSTER_SETTINGS[cluster]["long_eval"]["runs"]
    fixture_by_name = {
        fixture.fixture_name: fixture
        for fixture in resolve_tier_fixtures(
            "long_eval",
            catalog_by_name=catalog_by_name,
        )
    }
    payload = _base_payload(
        cluster=cluster,
        tier="long_eval",
        notes=[
            "Late-prefix long-eval tier only.",
            "These runs are too expensive for the fast inner loop and are intended for periodic stress validation.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
    )

    for spec in run_specs:
        for fixture_name in spec["fixtures"]:
            selected = fixture_by_name[fixture_name]
            payload["scenarios"].append(
                {
                    "name": _scenario_name(
                        cluster=cluster,
                        tier="long_eval",
                        fixture_name=selected.fixture_name,
                        batch=spec["batch"],
                        chunk=spec["chunk"],
                        cache_gib=spec["cache_gib"],
                        label=spec["label"],
                    ),
                    "stage": "exact_trace_bench_long_eval",
                    "cluster": cluster,
                    "long_eval_group": spec["label"],
                    "resource_profile": RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
                    "recommended_output_root": str(
                        scratch_root / cluster / "long_eval"
                    ),
                    **selected.to_source_payload(),
                    "attribution_batch_size": spec["batch"],
                    "feature_batch_size": spec["batch"],
                    "logit_batch_size": spec["batch"],
                    "decoder_chunk_size": spec["chunk"],
                    "cross_batch_decoder_cache_bytes": gib_to_bytes(spec["cache_gib"]),
                    **_select_exact_mode_knobs(spec),
                }
            )

    return payload


def build_tier_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    if tier == "fast":
        return build_fast_tier(
            cluster=cluster,
            catalog_by_name=catalog_by_name,
            scratch_root=scratch_root,
        )
    if tier == "anomaly":
        return build_anomaly_tier(
            cluster=cluster,
            catalog_by_name=catalog_by_name,
            scratch_root=scratch_root,
        )
    if tier == "long_eval":
        return build_long_eval_tier(
            cluster=cluster,
            catalog_by_name=catalog_by_name,
            scratch_root=scratch_root,
        )

    raise ValueError(f"Unknown tier '{tier}'")


def scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_bench_{tier}_{cluster}_scenarios.json"


def write_tier_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / scenario_file_name(tier=tier, cluster=cluster)
    write_json(output_path, payload)
    return output_path


def write_all_tiers(
    *,
    cluster: str,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    fixture_catalog: Path = DEFAULT_FIXTURE_CATALOG,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> list[Path]:
    catalog_by_name = (
        load_fixture_catalog(fixture_catalog) if fixture_catalog.exists() else {}
    )
    paths: list[Path] = []
    for tier in SCENARIO_TIERS:
        payload = build_tier_config(
            tier=tier,
            cluster=cluster,
            catalog_by_name=catalog_by_name,
            scratch_root=scratch_root,
        )
        paths.append(
            write_tier_config(
                payload,
                output_dir=output_dir,
                tier=tier,
                cluster=cluster,
            )
        )
    return paths
