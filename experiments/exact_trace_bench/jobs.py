from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT, recommended_output_root
from .io_utils import read_json
from .scenarios import (
    RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
    RESOURCE_PROFILE_STANDARD,
)
from .workspace import (
    DEFAULT_SNAPSHOT_ROOT,
    resolve_launch_workspace,
    sibling_library_root,
)


SBATCH_SCRIPTS: dict[tuple[str, str], Path] = {
    ("ascend", RESOURCE_PROFILE_STANDARD): REPO_ROOT
    / "scripts"
    / "trace_weekend_exact_chunked.ascend.sbatch",
    ("ascend", RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM): REPO_ROOT
    / "scripts"
    / "trace_weekend_exact_chunked_long_eval.ascend.sbatch",
    ("cardinal", RESOURCE_PROFILE_STANDARD): REPO_ROOT
    / "scripts"
    / "trace_weekend_exact_chunked.cardinal.sbatch",
    ("cardinal", RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM): REPO_ROOT
    / "scripts"
    / "trace_weekend_exact_chunked_long_eval.cardinal.sbatch",
}


def _resolve_resource_profile(scenarios_file: Path) -> str:
    payload = read_json(scenarios_file)
    metadata = payload.get("metadata") or {}
    metadata_profile = metadata.get("resource_profile")
    if metadata_profile:
        return str(metadata_profile)

    scenario_profiles = {
        str(scenario.get("resource_profile"))
        for scenario in payload.get("scenarios", [])
        if scenario.get("resource_profile")
    }
    if not scenario_profiles:
        return RESOURCE_PROFILE_STANDARD
    if len(scenario_profiles) > 1:
        raise ValueError(
            "Scenarios file mixes multiple resource profiles; split it or set metadata.resource_profile"
        )
    return next(iter(scenario_profiles))


def _scenario_count(scenarios_file: Path) -> int:
    payload = read_json(scenarios_file)
    scenarios = payload.get("scenarios") or []
    count = len(scenarios)
    if count <= 0:
        raise ValueError(f"No scenarios found in {scenarios_file}")
    return count


def _default_output_root(
    *,
    cluster: str,
    scenarios_file: Path,
) -> Path:
    payload = read_json(scenarios_file)
    metadata = payload.get("metadata") or {}
    stage = metadata.get("stage")
    recommended = metadata.get("recommended_output_root")
    if recommended:
        return Path(str(recommended))
    tier = metadata.get("tier")
    if tier:
        return recommended_output_root(
            cluster=cluster,
            tier=str(tier),
            scratch_root=DEFAULT_SCRATCH_ROOT,
        )
    if stage:
        return DEFAULT_SCRATCH_ROOT / cluster / str(stage)
    return DEFAULT_SCRATCH_ROOT / cluster / scenarios_file.stem


def render_launch_plan(
    *,
    cluster: str,
    scenarios_file: Path,
    output_root: Path | None = None,
    immutable_workspace: bool = False,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    walltime: str | None = None,
) -> dict[str, Any]:
    scenarios_file = scenarios_file.resolve()
    resource_profile = _resolve_resource_profile(scenarios_file)
    script_key = (cluster, resource_profile)
    if script_key not in SBATCH_SCRIPTS:
        raise ValueError(
            f"Unsupported launch profile for cluster={cluster!r}, resource_profile={resource_profile!r}"
        )

    resolved_output_root = (
        output_root
        or _default_output_root(
            cluster=cluster,
            scenarios_file=scenarios_file,
        )
    ).resolve()
    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)

    script_path = SBATCH_SCRIPTS[script_key].resolve()
    launch_scenarios_file = scenarios_file
    launch_script_path = script_path
    if immutable_workspace:
        resolved_source_root = source_root.resolve()
        try:
            launch_scenarios_file = workspace / scenarios_file.relative_to(
                resolved_source_root
            )
        except ValueError:
            launch_scenarios_file = scenarios_file
        try:
            launch_script_path = workspace / script_path.relative_to(
                resolved_source_root
            )
        except ValueError:
            launch_script_path = script_path

    scenario_count = _scenario_count(launch_scenarios_file)
    array_range = f"0-{scenario_count - 1}"
    export_blob = (
        "ALL,"
        f"SCENARIOS_FILE={launch_scenarios_file},"
        f"OUTPUT_ROOT={resolved_output_root},"
        f"WORKSPACE_ROOT={workspace},"
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}"
    )
    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--array={array_range}",
        f"--export={export_blob}",
        str(launch_script_path),
    ]

    return {
        "cluster": cluster,
        "scenarios_file": str(launch_scenarios_file),
        "output_root": str(resolved_output_root),
        "resource_profile": resource_profile,
        "scenario_count": scenario_count,
        "sbatch_argv": command_parts,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_command": " ".join(command_parts),
    }
