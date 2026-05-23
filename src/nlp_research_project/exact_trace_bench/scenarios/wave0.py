from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_GENERATED_DIR, DEFAULT_SCRATCH_ROOT
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
    WAVE0_REPEAT_COUNT,
    _append_exact_scenario,
    _base_payload,
    _require_cluster,
    _resolve_required_fixtures,
    _select_exact_mode_knobs,
)


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
