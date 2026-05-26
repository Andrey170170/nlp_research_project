from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_SCRATCH_ROOT
from ..fixtures import resolve_tier_fixtures
from .base import (
    CLUSTER_SETTINGS,
    RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
    RESOURCE_PROFILE_STANDARD,
    _append_exact_scenario,
    _base_payload,
    _require_cluster,
    _select_exact_mode_knobs,
)


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
