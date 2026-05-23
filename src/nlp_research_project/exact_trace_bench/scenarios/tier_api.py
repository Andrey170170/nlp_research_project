from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    DEFAULT_SCRATCH_ROOT,
)
from ..fixtures import load_fixture_catalog
from ..io_utils import ensure_dir, write_json
from .base import SCENARIO_TIERS
from .canonical import build_anomaly_tier, build_fast_tier, build_long_eval_tier


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
