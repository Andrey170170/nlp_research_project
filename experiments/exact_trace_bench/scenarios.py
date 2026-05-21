from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    DEFAULT_SCRATCH_ROOT,
    DEFAULT_WAVE0_BASELINE_REGISTRY,
    base_trace_defaults,
    gib_to_bytes,
    recommended_output_root,
)
from .fixtures import (
    FixtureRef,
    load_fixture_catalog,
    resolve_fixture,
    resolve_tier_fixtures,
)
from .io_utils import ensure_dir, write_json


SCENARIO_TIERS = ("fast", "anomaly", "long_eval")
WAVE2A_PHASE1_TIERS = ("fast", "anomaly")
WAVE2B_PHASE4_TIERS = ("fast", "anomaly")
WAVE2C_ROW_ENCODER_TIERS = ("fast", "anomaly")
WAVE3_INTERACTION_CONFIRMATION_TIERS = ("fast", "anomaly")

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
        _append_exact_scenario(
            payload,
            cluster=cluster,
            tier="fast",
            stage="exact_trace_bench_fast",
            fixture=fixture,
            batch=cfg["batch"],
            chunk=cfg["chunk"],
            cache_gib=cfg["cache_gib"],
            resource_profile=RESOURCE_PROFILE_STANDARD,
            scratch_root=scratch_root,
            extra=_select_exact_mode_knobs(cfg),
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
        _append_exact_scenario(
            payload,
            cluster=cluster,
            tier="anomaly",
            stage="exact_trace_bench_anomaly",
            fixture=fixture,
            batch=cfg["batch"],
            chunk=cfg["chunk"],
            cache_gib=cfg["cache_gib"],
            resource_profile=RESOURCE_PROFILE_STANDARD,
            scratch_root=scratch_root,
            extra=_select_exact_mode_knobs(cfg),
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
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier="long_eval",
                stage="exact_trace_bench_long_eval",
                fixture=selected,
                batch=spec["batch"],
                chunk=spec["chunk"],
                cache_gib=spec["cache_gib"],
                resource_profile=RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
                scratch_root=scratch_root,
                label=spec["label"],
                extra={
                    "long_eval_group": spec["label"],
                    **_select_exact_mode_knobs(spec),
                },
            )

    return payload


def build_wave0_baseline_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Build expanded Wave 0 baseline scenarios for one scratch tier."""
    _require_cluster(cluster)
    if tier == "fast":
        cfg = CLUSTER_SETTINGS[cluster]["fast"]
        payload = _base_payload(
            cluster=cluster,
            tier="fast",
            notes=[
                "Wave 0 expanded prompt baseline for base GSM prompts.",
                "Canonical fast fixtures are repeated twice to estimate baseline noise.",
            ],
            scratch_root=scratch_root,
            resource_profile=RESOURCE_PROFILE_STANDARD,
        )
        payload["metadata"].update(
            {
                "stage": "exact_trace_wave0_baseline_fast",
                "wave": "wave0",
                "run_goal": "Create expanded Wave 0 base-prompt baselines and timing references.",
            }
        )
        repeated = _resolve_required_fixtures(
            WAVE0_CANONICAL_FAST_FIXTURES,
            catalog_by_name=catalog_by_name,
        )
        for repeat_index in range(1, WAVE0_REPEAT_COUNT + 1):
            for fixture in repeated:
                _append_exact_scenario(
                    payload,
                    cluster=cluster,
                    tier="fast",
                    stage="exact_trace_wave0_baseline_fast",
                    fixture=fixture,
                    batch=cfg["batch"],
                    chunk=cfg["chunk"],
                    cache_gib=cfg["cache_gib"],
                    resource_profile=RESOURCE_PROFILE_STANDARD,
                    scratch_root=scratch_root,
                    label=f"wave0_r{repeat_index}",
                    extra={
                        "wave": "wave0",
                        "wave0_role": "canonical_repeat",
                        "wave0_repeat_index": repeat_index,
                        **_select_exact_mode_knobs(cfg),
                    },
                )
        for fixture in _resolve_required_fixtures(
            WAVE0_NEW_BASE_FIXTURES,
            catalog_by_name=catalog_by_name,
        ):
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier="fast",
                stage="exact_trace_wave0_baseline_fast",
                fixture=fixture,
                batch=cfg["batch"],
                chunk=cfg["chunk"],
                cache_gib=cfg["cache_gib"],
                resource_profile=RESOURCE_PROFILE_STANDARD,
                scratch_root=scratch_root,
                label="wave0_base",
                extra={
                    "wave": "wave0",
                    "wave0_role": "expanded_base",
                    **_select_exact_mode_knobs(cfg),
                },
            )
        return payload

    if tier == "anomaly":
        cfg = CLUSTER_SETTINGS[cluster]["anomaly"]
        payload = _base_payload(
            cluster=cluster,
            tier="anomaly",
            notes=[
                "Wave 0 repeated Prompt-94 anomaly baseline.",
                "Prompt 94 stays in the anomaly scratch tier for continuity.",
            ],
            scratch_root=scratch_root,
            resource_profile=RESOURCE_PROFILE_STANDARD,
        )
        payload["metadata"].update(
            {
                "stage": "exact_trace_wave0_baseline_anomaly",
                "wave": "wave0",
                "run_goal": "Create repeated Wave 0 Prompt-94 anomaly baselines.",
            }
        )
        fixtures = _resolve_required_fixtures(
            WAVE0_CANONICAL_ANOMALY_FIXTURES,
            catalog_by_name=catalog_by_name,
        )
        for repeat_index in range(1, WAVE0_REPEAT_COUNT + 1):
            for fixture in fixtures:
                _append_exact_scenario(
                    payload,
                    cluster=cluster,
                    tier="anomaly",
                    stage="exact_trace_wave0_baseline_anomaly",
                    fixture=fixture,
                    batch=cfg["batch"],
                    chunk=cfg["chunk"],
                    cache_gib=cfg["cache_gib"],
                    resource_profile=RESOURCE_PROFILE_STANDARD,
                    scratch_root=scratch_root,
                    label=f"wave0_r{repeat_index}",
                    extra={
                        "wave": "wave0",
                        "wave0_role": "canonical_repeat",
                        "wave0_repeat_index": repeat_index,
                        **_select_exact_mode_knobs(cfg),
                    },
                )
        return payload

    if tier == "long_eval":
        no_cache_spec = CLUSTER_SETTINGS[cluster]["long_eval"]["runs"][0]
        payload = _base_payload(
            cluster=cluster,
            tier="long_eval",
            notes=[
                "Wave 0 expanded late-prefix baseline.",
                "Runs current late fixtures plus a selected late-prefix subset of new GSM prompts.",
            ],
            scratch_root=scratch_root,
            resource_profile=RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
        )
        payload["metadata"].update(
            {
                "stage": "exact_trace_wave0_baseline_long_eval",
                "wave": "wave0",
                "run_goal": "Create expanded Wave 0 late-prefix baselines and timing references.",
            }
        )
        fixtures = _resolve_required_fixtures(
            WAVE0_CANONICAL_LATE_FIXTURES + WAVE0_NEW_LATE_FIXTURES,
            catalog_by_name=catalog_by_name,
        )
        for fixture in fixtures:
            role = (
                "canonical_late"
                if fixture.fixture_name in WAVE0_CANONICAL_LATE_FIXTURES
                else "expanded_late"
            )
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier="long_eval",
                stage="exact_trace_wave0_baseline_long_eval",
                fixture=fixture,
                batch=no_cache_spec["batch"],
                chunk=no_cache_spec["chunk"],
                cache_gib=no_cache_spec["cache_gib"],
                resource_profile=RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
                scratch_root=scratch_root,
                label="wave0_late",
                extra={
                    "long_eval_group": "wave0_baseline",
                    "wave": "wave0",
                    "wave0_role": role,
                    **_select_exact_mode_knobs(no_cache_spec),
                },
            )
        return payload

    raise ValueError(f"Unknown Wave 0 tier '{tier}'")


def wave0_scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_wave0_baseline_{tier}_{cluster}_scenarios.json"


def write_wave0_baseline_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave0_scenario_file_name(tier=tier, cluster=cluster)
    write_json(output_path, payload)
    return output_path


def build_wave2a_phase1_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    baseline_registry: Path = DEFAULT_WAVE0_BASELINE_REGISTRY,
) -> dict[str, Any]:
    """Build Wave 2A Phase-1 trace-batch policy sweep scenarios."""
    _require_cluster(cluster)
    if tier not in WAVE2A_PHASE1_TIERS:
        raise ValueError(f"Unknown Wave 2A Phase-1 tier '{tier}'")

    cfg = CLUSTER_SETTINGS[cluster][tier]
    fixture_names = (
        WAVE0_CANONICAL_FAST_FIXTURES
        if tier == "fast"
        else WAVE0_CANONICAL_ANOMALY_FIXTURES
    )
    fixtures = _resolve_required_fixtures(
        fixture_names,
        catalog_by_name=catalog_by_name,
    )
    stage = f"exact_trace_wave2a_phase1_{tier}"
    payload = _base_payload(
        cluster=cluster,
        tier=tier,
        notes=[
            "Wave 2A Phase-1 trace-batch policy sweep over canonical sentinels.",
            "Self-scored in metrics mode against same-cluster Wave 0 fp32 baselines.",
            "Uses Wave 1 locked batch/chunk/cache settings for this cluster and tier.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )
    payload["metadata"].update(
        {
            "stage": stage,
            "wave": "wave2a",
            "sweep_family": "phase1_trace_batch",
            "baseline_registry": str(baseline_registry),
            "run_name": "Wave 2A Phase-1 trace-batch sweep",
            "run_goal": "Measure Phase-1 trace-batch policy effects against Wave 0 metrics baselines.",
        }
    )

    for fixture in fixtures:
        for variant in WAVE2A_PHASE1_VARIANTS:
            label = variant["label"]
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier=tier,
                stage=stage,
                fixture=fixture,
                batch=cfg["batch"],
                chunk=cfg["chunk"],
                cache_gib=cfg["cache_gib"],
                resource_profile=RESOURCE_PROFILE_STANDARD,
                scratch_root=scratch_root,
                label=f"wave2a_phase1_{label}",
                extra={
                    "wave": "wave2a",
                    "sweep_family": "phase1_trace_batch",
                    "phase1_variant": label,
                    "baseline_check": {
                        "enabled": True,
                        "mode": "metrics",
                        "registry_key": f"wave0/{fixture.fixture_name}/{cluster}/{tier}/fp32_default",
                        "baseline_required": True,
                        "thresholds": None,
                    },
                    **{key: value for key, value in variant.items() if key != "label"},
                    **_select_exact_mode_knobs(cfg),
                },
            )

    return payload


def wave2a_phase1_scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_wave2a_phase1_{tier}_{cluster}_scenarios.json"


def write_wave2a_phase1_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave2a_phase1_scenario_file_name(
        tier=tier,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path


def build_wave2b_phase4_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    baseline_registry: Path = DEFAULT_WAVE0_BASELINE_REGISTRY,
) -> dict[str, Any]:
    """Build Wave 2B Phase-4 scheduler/refresh/ranker/executor sweep scenarios."""
    _require_cluster(cluster)
    if tier not in WAVE2B_PHASE4_TIERS:
        raise ValueError(f"Unknown Wave 2B Phase-4 tier '{tier}'")

    cfg = CLUSTER_SETTINGS[cluster][tier]
    fixture_names = (
        WAVE0_CANONICAL_FAST_FIXTURES
        if tier == "fast"
        else WAVE0_CANONICAL_ANOMALY_FIXTURES
    )
    fixtures = _resolve_required_fixtures(
        fixture_names,
        catalog_by_name=catalog_by_name,
    )
    stage = f"exact_trace_wave2b_phase4_{tier}"
    payload = _base_payload(
        cluster=cluster,
        tier=tier,
        notes=[
            "Wave 2B Phase-4 scheduler/refresh/ranker/executor sweep over canonical sentinels.",
            "Self-scored in metrics mode against same-cluster Wave 0 fp32 baselines.",
            "Uses Wave 1 locked batch/chunk/cache settings for this cluster and tier.",
            "Phase-1 remains legacy_default from the Wave 2A decision: phase1_trace_batch_policy='legacy' and phase1_trace_batch_size_max omitted.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )
    payload["metadata"].update(
        {
            "stage": stage,
            "wave": "wave2b",
            "sweep_family": "phase4_scheduler_refresh_ranker_executor",
            "baseline_registry": str(baseline_registry),
            "run_name": "Wave 2B Phase-4 scheduler/refresh/ranker/executor sweep",
            "run_goal": "Measure curated Phase-4 scheduler, refresh, ranker, and row-executor variants against Wave 0 metrics baselines while keeping Phase-1 legacy_default.",
        }
    )

    for fixture in fixtures:
        for variant in WAVE2B_PHASE4_VARIANTS:
            label = variant["label"]
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier=tier,
                stage=stage,
                fixture=fixture,
                batch=cfg["batch"],
                chunk=cfg["chunk"],
                cache_gib=cfg["cache_gib"],
                resource_profile=RESOURCE_PROFILE_STANDARD,
                scratch_root=scratch_root,
                label=f"wave2b_phase4_{label}",
                extra={
                    "wave": "wave2b",
                    "sweep_family": "phase4_scheduler_refresh_ranker_executor",
                    "phase4_variant": label,
                    "phase1_trace_batch_policy": "legacy",
                    "baseline_check": {
                        "enabled": True,
                        "mode": "metrics",
                        "registry_key": f"wave0/{fixture.fixture_name}/{cluster}/{tier}/fp32_default",
                        "baseline_required": True,
                        "thresholds": None,
                    },
                    **{key: value for key, value in variant.items() if key != "label"},
                    **_select_exact_mode_knobs(cfg),
                },
            )

    return payload


def wave2b_phase4_scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_wave2b_phase4_{tier}_{cluster}_scenarios.json"


def write_wave2b_phase4_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave2b_phase4_scenario_file_name(
        tier=tier,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path


def build_wave2c_row_encoder_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    baseline_registry: Path = DEFAULT_WAVE0_BASELINE_REGISTRY,
) -> dict[str, Any]:
    """Build Wave 2C row-store/encoder/staging/planner sweep scenarios."""
    _require_cluster(cluster)
    if tier not in WAVE2C_ROW_ENCODER_TIERS:
        raise ValueError(f"Unknown Wave 2C row/encoder tier '{tier}'")

    cfg = CLUSTER_SETTINGS[cluster][tier]
    fixture_names = (
        WAVE0_CANONICAL_FAST_FIXTURES
        if tier == "fast"
        else WAVE0_CANONICAL_ANOMALY_FIXTURES
    )
    fixtures = _resolve_required_fixtures(
        fixture_names,
        catalog_by_name=catalog_by_name,
    )
    stage = f"exact_trace_wave2c_row_encoder_{tier}"
    payload = _base_payload(
        cluster=cluster,
        tier=tier,
        notes=[
            "Wave 2C row/encoder/staging/planner sweep over canonical sentinels.",
            "Self-scored in metrics mode against same-cluster Wave 0 fp32 baselines.",
            "Uses Wave 1 locked batch/chunk/cache settings for this cluster and tier.",
            "Legacy Phase-1 and legacy Phase-4 are intentionally retained to avoid cross-family interactions before Wave 3.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )
    payload["metadata"].update(
        {
            "stage": stage,
            "wave": "wave2c",
            "sweep_family": "row_encoder_staging_planner",
            "baseline_registry": str(baseline_registry),
            "run_name": "Wave 2C row/encoder/staging/planner sweep",
            "run_goal": "Measure curated row-store, encoder residency, CPU staging, row-subchunk, and feature-batch planner variants against Wave 0 metrics baselines while keeping legacy Phase-1 and legacy Phase-4 defaults.",
        }
    )

    for fixture in fixtures:
        for variant in WAVE2C_ROW_ENCODER_VARIANTS:
            label = variant["label"]
            knobs = {
                **WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS,
                **{key: value for key, value in variant.items() if key != "label"},
            }
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier=tier,
                stage=stage,
                fixture=fixture,
                batch=cfg["batch"],
                chunk=cfg["chunk"],
                cache_gib=cfg["cache_gib"],
                resource_profile=RESOURCE_PROFILE_STANDARD,
                scratch_root=scratch_root,
                label=f"wave2c_row_encoder_{label}",
                extra={
                    "wave": "wave2c",
                    "sweep_family": "row_encoder_staging_planner",
                    "row_encoder_variant": label,
                    "baseline_check": {
                        "enabled": True,
                        "mode": "metrics",
                        "registry_key": f"wave0/{fixture.fixture_name}/{cluster}/{tier}/fp32_default",
                        "baseline_required": True,
                        "thresholds": None,
                    },
                    **knobs,
                    **_select_exact_mode_knobs(cfg),
                },
            )

    return payload


def wave2c_row_encoder_scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_wave2c_row_encoder_{tier}_{cluster}_scenarios.json"


def write_wave2c_row_encoder_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave2c_row_encoder_scenario_file_name(
        tier=tier,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path


def build_wave3_interaction_confirmation_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    baseline_registry: Path = DEFAULT_WAVE0_BASELINE_REGISTRY,
    include_optional_speed_interaction: bool = False,
) -> dict[str, Any]:
    """Build Wave 3 interaction-confirmation scenarios."""
    _require_cluster(cluster)
    if tier not in WAVE3_INTERACTION_CONFIRMATION_TIERS:
        raise ValueError(f"Unknown Wave 3 interaction-confirmation tier '{tier}'")

    cfg = CLUSTER_SETTINGS[cluster][tier]
    fixture_names = (
        WAVE0_CANONICAL_FAST_FIXTURES
        if tier == "fast"
        else WAVE0_CANONICAL_ANOMALY_FIXTURES
    )
    fixtures = _resolve_required_fixtures(
        fixture_names,
        catalog_by_name=catalog_by_name,
    )
    variants = WAVE3_INTERACTION_CONFIRMATION_VARIANTS
    if include_optional_speed_interaction:
        variants = variants + WAVE3_OPTIONAL_SPEED_INTERACTION_VARIANTS

    stage = f"exact_trace_wave3_interaction_confirmation_{tier}"
    payload = _base_payload(
        cluster=cluster,
        tier=tier,
        notes=[
            "Wave 3 interaction-confirmation sweep over canonical sentinels.",
            "Self-scored in metrics mode against same-cluster Wave 0 fp32 baselines.",
            "Confirms selected interactions among deferred refresh, row subchunking, and feature-batch planning.",
            "Pins legacy Phase-1/Phase-4, row-store, and encoder defaults outside the interaction under test.",
            "Optional speed interaction with streaming_v1 is excluded unless requested.",
        ],
        scratch_root=scratch_root,
        resource_profile=RESOURCE_PROFILE_STANDARD,
    )
    payload["metadata"].update(
        {
            "stage": stage,
            "wave": "wave3",
            "sweep_family": "interaction_confirmation",
            "baseline_registry": str(baseline_registry),
            "include_optional_speed_interaction": include_optional_speed_interaction,
            "run_name": "Wave 3 interaction-confirmation sweep",
            "run_goal": "Measure candidate interactions among deferred refresh, row subchunking, and feature-batch planning against Wave 0 metrics baselines while pinning legacy defaults outside the interaction under test.",
        }
    )

    for fixture in fixtures:
        for variant in variants:
            label = variant["label"]
            _append_exact_scenario(
                payload,
                cluster=cluster,
                tier=tier,
                stage=stage,
                fixture=fixture,
                batch=cfg["batch"],
                chunk=cfg["chunk"],
                cache_gib=cfg["cache_gib"],
                resource_profile=RESOURCE_PROFILE_STANDARD,
                scratch_root=scratch_root,
                label=f"wave3_interaction_{label}",
                extra={
                    "wave": "wave3",
                    "sweep_family": "interaction_confirmation",
                    "interaction_variant": label,
                    "baseline_check": {
                        "enabled": True,
                        "mode": "metrics",
                        "registry_key": f"wave0/{fixture.fixture_name}/{cluster}/{tier}/fp32_default",
                        "baseline_required": True,
                        "thresholds": None,
                    },
                    **WAVE3_INTERACTION_CONFIRMATION_LEGACY_DEFAULTS,
                    **{key: value for key, value in variant.items() if key != "label"},
                    **_select_exact_mode_knobs(cfg),
                },
            )

    return payload


def wave3_interaction_confirmation_scenario_file_name(
    *,
    tier: str,
    cluster: str,
) -> str:
    return f"exact_trace_wave3_interaction_confirmation_{tier}_{cluster}_scenarios.json"


def write_wave3_interaction_confirmation_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave3_interaction_confirmation_scenario_file_name(
        tier=tier,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path


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
