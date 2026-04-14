from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    DEFAULT_SCRATCH_ROOT,
    REPO_ROOT,
)
from .fixtures import load_fixture_catalog
from .jobs import render_launch_plan
from .scenarios import build_tier_config, write_tier_config
from .workspace import (
    DEFAULT_SNAPSHOT_ROOT,
    create_workspace_snapshot,
    make_snapshot_read_only,
)


PRESET_DEFS: dict[str, dict[str, tuple[str, ...]]] = {
    "fast-ascend": {"clusters": ("ascend",), "tiers": ("fast", "anomaly")},
    "fast-cardinal": {"clusters": ("cardinal",), "tiers": ("fast", "anomaly")},
    "full-ascend": {
        "clusters": ("ascend",),
        "tiers": ("fast", "anomaly", "long_eval"),
    },
    "full-cardinal": {
        "clusters": ("cardinal",),
        "tiers": ("fast", "anomaly", "long_eval"),
    },
    "fast-all": {
        "clusters": ("ascend", "cardinal"),
        "tiers": ("fast", "anomaly"),
    },
    "full-all": {
        "clusters": ("ascend", "cardinal"),
        "tiers": ("fast", "anomaly", "long_eval"),
    },
}


def preset_names() -> list[str]:
    return sorted(PRESET_DEFS)


def run_preset(
    *,
    preset: str,
    generated_dir: Path = DEFAULT_GENERATED_DIR,
    fixture_catalog: Path = DEFAULT_FIXTURE_CATALOG,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    immutable_workspace: bool = True,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label_prefix: str | None = None,
    walltime: str | None = None,
    print_only: bool = False,
) -> list[dict[str, Any]]:
    if preset not in PRESET_DEFS:
        raise ValueError(f"Unknown preset '{preset}'")

    fixture_catalog_by_name = {}
    if fixture_catalog.exists():
        fixture_catalog_by_name = load_fixture_catalog(fixture_catalog)

    plans: list[dict[str, Any]] = []
    definition = PRESET_DEFS[preset]
    launch_source_root = source_root
    launch_immutable = immutable_workspace
    effective_generated_dir = generated_dir
    if immutable_workspace:
        snapshot_label = workspace_label_prefix or preset.replace("-", "_")
        launch_source_root = create_workspace_snapshot(
            snapshot_root=snapshot_root,
            source_root=source_root,
            label=snapshot_label,
            read_only=False,
        )
        launch_immutable = False
        try:
            effective_generated_dir = (
                launch_source_root
                / generated_dir.resolve().relative_to(source_root.resolve())
            )
        except ValueError:
            effective_generated_dir = (
                launch_source_root / "experiments" / "generated" / "exact_trace_bench"
            )

    for cluster in definition["clusters"]:
        for tier in definition["tiers"]:
            payload = build_tier_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=fixture_catalog_by_name,
                scratch_root=scratch_root,
            )
            scenarios_file = write_tier_config(
                payload,
                output_dir=effective_generated_dir,
                tier=tier,
                cluster=cluster,
            )
            plan = render_launch_plan(
                cluster=cluster,
                scenarios_file=scenarios_file,
                immutable_workspace=launch_immutable,
                snapshot_root=snapshot_root,
                source_root=launch_source_root,
                workspace_label=None,
                walltime=walltime,
            )
            if immutable_workspace:
                script_path = Path(plan["sbatch_script"]).resolve()
                try:
                    snapshot_script_path = launch_source_root / script_path.relative_to(
                        source_root.resolve()
                    )
                    plan["sbatch_script"] = str(snapshot_script_path)
                    plan["sbatch_argv"][-1] = str(snapshot_script_path)
                    plan["sbatch_command"] = " ".join(plan["sbatch_argv"])
                except ValueError:
                    pass
                plan["immutable_workspace"] = True
            plans.append(plan)
            if not print_only:
                subprocess.run(plan["sbatch_argv"], check=True)

    if immutable_workspace:
        make_snapshot_read_only(launch_source_root)

    return plans
