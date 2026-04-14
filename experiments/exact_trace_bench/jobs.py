from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT, recommended_output_root
from .io_utils import read_json
from .workspace import DEFAULT_SNAPSHOT_ROOT, resolve_launch_workspace


SBATCH_SCRIPTS: dict[str, Path] = {
    "ascend": REPO_ROOT / "scripts" / "trace_weekend_exact_chunked.ascend.sbatch",
    "cardinal": REPO_ROOT / "scripts" / "trace_weekend_exact_chunked.cardinal.sbatch",
}


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
    if cluster not in SBATCH_SCRIPTS:
        raise ValueError(f"Unsupported cluster '{cluster}'")

    scenarios_file = scenarios_file.resolve()
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

    script_path = SBATCH_SCRIPTS[cluster].resolve()
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

    count_expr = (
        "$(($(uv run python experiments/print_scenario_count.py --scenarios-file "
        f"{shlex.quote(str(launch_scenarios_file))})-1))"
    )
    export_blob = (
        "ALL,"
        f"SCENARIOS_FILE={launch_scenarios_file},"
        f"OUTPUT_ROOT={resolved_output_root},"
        f"WORKSPACE_ROOT={workspace}"
    )
    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--array=0-{count_expr}",
        f"--export={export_blob}",
        str(launch_script_path),
    ]

    return {
        "cluster": cluster,
        "scenarios_file": str(launch_scenarios_file),
        "output_root": str(resolved_output_root),
        "workspace_root": str(workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_command": " ".join(command_parts),
    }
