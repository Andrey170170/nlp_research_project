from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_GENERATED_DIR,
    DEFAULT_SCRATCH_ROOT,
    DEFAULT_WAVE0_BASELINE_REGISTRY,
)
from ..io_utils import ensure_dir, write_json
from .base import (
    CLUSTER_SETTINGS,
    RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
    RESOURCE_PROFILE_STANDARD,
    WAVE0_CANONICAL_ANOMALY_FIXTURES,
    WAVE0_CANONICAL_FAST_FIXTURES,
    WAVE0_CANONICAL_LATE_FIXTURES,
    WAVE0_NEW_BASE_FIXTURES,
    WAVE0_NEW_LATE_FIXTURES,
    WAVE2A_PHASE1_TIERS,
    WAVE2A_PHASE1_VARIANTS,
    WAVE2B_PHASE4_TIERS,
    WAVE2B_PHASE4_VARIANTS,
    WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS,
    WAVE2C_ROW_ENCODER_TIERS,
    WAVE2C_ROW_ENCODER_VARIANTS,
    WAVE3_INTERACTION_CONFIRMATION_LEGACY_DEFAULTS,
    WAVE3_INTERACTION_CONFIRMATION_TIERS,
    WAVE3_INTERACTION_CONFIRMATION_VARIANTS,
    WAVE3_OPTIONAL_SPEED_INTERACTION_VARIANTS,
    WAVE4_GENERALIZATION_TIERS,
    WAVE4_GENERALIZATION_VARIANTS,
    _append_exact_scenario,
    _base_payload,
    _require_cluster,
    _resolve_required_fixtures,
    _select_exact_mode_knobs,
)


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


def _wave4_fixture_names(tier: str) -> tuple[str, ...]:
    if tier == "fast":
        return WAVE0_CANONICAL_FAST_FIXTURES + WAVE0_NEW_BASE_FIXTURES
    if tier == "anomaly":
        return WAVE0_CANONICAL_ANOMALY_FIXTURES
    if tier == "long_eval":
        return WAVE0_CANONICAL_LATE_FIXTURES + WAVE0_NEW_LATE_FIXTURES
    raise ValueError(f"Unknown Wave 4 generalization tier '{tier}'")


def build_wave4_generalization_config(
    *,
    tier: str,
    cluster: str,
    catalog_by_name: dict[str, dict[str, Any]],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    baseline_registry: Path = DEFAULT_WAVE0_BASELINE_REGISTRY,
) -> dict[str, Any]:
    """Build Wave 4 prompt-generalization/finalist-validation scenarios."""
    _require_cluster(cluster)
    if tier not in WAVE4_GENERALIZATION_TIERS:
        raise ValueError(f"Unknown Wave 4 generalization tier '{tier}'")

    if tier == "long_eval":
        cfg = CLUSTER_SETTINGS[cluster]["long_eval"]["runs"][0]
        resource_profile = RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM
    else:
        cfg = CLUSTER_SETTINGS[cluster][tier]
        resource_profile = RESOURCE_PROFILE_STANDARD

    fixture_names = _wave4_fixture_names(tier)
    fixtures = _resolve_required_fixtures(
        fixture_names,
        catalog_by_name=catalog_by_name,
    )
    stage = f"exact_trace_wave4_generalization_{tier}"
    payload = _base_payload(
        cluster=cluster,
        tier=tier,
        notes=[
            "Wave 4 prompt-generalization/finalist-validation over broad Wave 0 coverage.",
            "Self-scored in metrics mode against same-cluster Wave 0 fp32 baselines.",
            "Validates primary speed/resource finalist row_subchunk_size=512 and conservative memory/planning finalist plan_feature_batch_size=true.",
            "Pins legacy Phase-1/Phase-4, row-store, and encoder defaults outside tested dimensions.",
        ],
        scratch_root=scratch_root,
        resource_profile=resource_profile,
    )
    payload["metadata"].update(
        {
            "stage": stage,
            "wave": "wave4",
            "sweep_family": "prompt_generalization",
            "baseline_registry": str(baseline_registry),
            "run_name": "Wave 4 prompt-generalization finalist validation",
            "run_goal": "Validate row_subchunk_size=512 and plan_feature_batch_size=true finalists against broad Wave 0 prompt coverage using metrics-mode baseline checks.",
        }
    )

    for fixture in fixtures:
        for variant in WAVE4_GENERALIZATION_VARIANTS:
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
                resource_profile=resource_profile,
                scratch_root=scratch_root,
                label=f"wave4_generalization_{label}",
                extra={
                    "wave": "wave4",
                    "sweep_family": "prompt_generalization",
                    "generalization_variant": label,
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


def wave4_generalization_scenario_file_name(*, tier: str, cluster: str) -> str:
    return f"exact_trace_wave4_generalization_{tier}_{cluster}_scenarios.json"


def write_wave4_generalization_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    tier: str,
    cluster: str,
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / wave4_generalization_scenario_file_name(
        tier=tier,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path
