from __future__ import annotations

import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_SCRATCH_ROOT,
    DEFAULT_WAVE0_FIXTURE_OUTPUT_DIR,
    DEFAULT_WAVE0_FIXTURE_TARGET_SPEC,
    REPO_ROOT,
    recommended_output_root,
)
from .full_answer.schemas import load_shards
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
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked.ascend.sbatch",
    ("ascend", RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked_long_eval.ascend.sbatch",
    ("cardinal", RESOURCE_PROFILE_STANDARD): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked.cardinal.sbatch",
    ("cardinal", RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked_long_eval.cardinal.sbatch",
}

SBATCH_FIXTURE_PREP_SCRIPTS: dict[str, Path] = {
    "ascend": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "prepare_weekend_prefix_fixtures.ascend.sbatch",
    "cardinal": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "prepare_weekend_prefix_fixtures.cardinal.sbatch",
}

SBATCH_FULL_ANSWER_TRAJECTORY_SCRIPTS: dict[str, Path] = {
    "ascend": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_prepare.ascend.sbatch",
    "cardinal": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_prepare.cardinal.sbatch",
}

SBATCH_FULL_ANSWER_TRACE_SCRIPTS: dict[str, Path] = {
    "ascend": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace.ascend.sbatch",
    "cardinal": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace.cardinal.sbatch",
}

GOAL_BY_TIER: dict[str, str] = {
    "fast": "Quick sanity sweep across base fixtures.",
    "anomaly": "Reproduce and monitor anomaly-focused fixtures.",
    "long_eval": "Stress-test long-eval fixtures with high-memory resources.",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_RUN_ID_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify_run_name(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "launch"
    slug = _SLUG_RE.sub("-", text).strip("-")
    return slug[:80] or "launch"


def _normalize_run_id(value: str) -> str:
    cleaned = _RUN_ID_PART_RE.sub("-", value.strip()).strip("-._")
    return cleaned or "launch"


def _normalize_free_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _default_run_name(
    *,
    cluster: str,
    scenarios_file: Path,
    metadata: dict[str, Any],
) -> str:
    configured = _normalize_free_text(metadata.get("run_name") or metadata.get("name"))
    if configured:
        return configured

    tier = metadata.get("tier")
    if tier:
        return f"exact_trace_bench_{cluster}_{tier}"

    stage = metadata.get("stage")
    if stage:
        return str(stage)

    return scenarios_file.stem


def _default_run_description(
    *,
    cluster: str,
    scenarios_file: Path,
    metadata: dict[str, Any],
) -> str:
    configured = _normalize_free_text(
        metadata.get("run_description") or metadata.get("description")
    )
    if configured:
        return configured

    notes = metadata.get("notes")
    if isinstance(notes, list):
        for note in notes:
            normalized_note = _normalize_free_text(str(note))
            if normalized_note:
                return normalized_note

    tier = metadata.get("tier")
    if tier:
        return f"exact_trace_bench {cluster}/{tier} launch"
    return f"exact_trace_bench launch for {scenarios_file.name}"


def _default_run_goal(metadata: dict[str, Any]) -> str:
    configured = _normalize_free_text(metadata.get("run_goal") or metadata.get("goal"))
    if configured:
        return configured
    tier = metadata.get("tier")
    return GOAL_BY_TIER.get(str(tier), "Execute exact trace benchmark scenarios.")


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
    run_id: str | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
    immutable_workspace: bool = False,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    walltime: str | None = None,
    baseline_registry: Path | None = None,
    fail_on_baseline_missing: bool = False,
    fail_on_validation_fail: bool = False,
) -> dict[str, Any]:
    scenarios_file = scenarios_file.resolve()
    resource_profile = _resolve_resource_profile(scenarios_file)
    script_key = (cluster, resource_profile)
    if script_key not in SBATCH_SCRIPTS:
        raise ValueError(
            f"Unsupported launch profile for cluster={cluster!r}, resource_profile={resource_profile!r}"
        )

    scenarios_payload = read_json(scenarios_file)
    scenarios_metadata = scenarios_payload.get("metadata") or {}

    base_output_root = (
        output_root
        or _default_output_root(
            cluster=cluster,
            scenarios_file=scenarios_file,
        )
    ).resolve()
    resolved_run_name = _normalize_free_text(run_name) or _default_run_name(
        cluster=cluster,
        scenarios_file=scenarios_file,
        metadata=scenarios_metadata,
    )
    resolved_run_description = _normalize_free_text(
        run_description
    ) or _default_run_description(
        cluster=cluster,
        scenarios_file=scenarios_file,
        metadata=scenarios_metadata,
    )
    resolved_run_goal = _normalize_free_text(run_goal) or _default_run_goal(
        scenarios_metadata
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    generated_run_id = f"{timestamp}_{_slugify_run_name(resolved_run_name)}"
    resolved_run_id = (
        _normalize_run_id(run_id) if run_id is not None else generated_run_id
    )
    resolved_output_root = (base_output_root / resolved_run_id).resolve()
    if resolved_output_root.exists():
        raise ValueError(
            f"Launch output root already exists: {resolved_output_root}. "
            "Use a different --run-id or base output root to avoid mixing artifacts."
        )

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
    export_parts = [
        "ALL",
        f"SCENARIOS_FILE={launch_scenarios_file}",
        f"OUTPUT_ROOT={resolved_output_root}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
    ]
    if baseline_registry is not None:
        export_parts.append(f"BASELINE_REGISTRY={baseline_registry.resolve()}")
    if fail_on_baseline_missing:
        export_parts.append("FAIL_ON_BASELINE_MISSING=1")
    if fail_on_validation_fail:
        export_parts.append("FAIL_ON_VALIDATION_FAIL=1")
    export_blob = ",".join(export_parts)
    script_args: list[str] = [
        "--run-id",
        resolved_run_id,
        "--run-name",
        resolved_run_name,
    ]
    if resolved_run_description is not None:
        script_args.extend(["--run-description", resolved_run_description])
    if resolved_run_goal is not None:
        script_args.extend(["--run-goal", resolved_run_goal])

    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--array={array_range}",
        f"--export={export_blob}",
        str(launch_script_path),
        *script_args,
    ]

    return {
        "cluster": cluster,
        "scenarios_file": str(launch_scenarios_file),
        "output_base_root": str(base_output_root),
        "output_root": str(resolved_output_root),
        "resource_profile": resource_profile,
        "scenario_count": scenario_count,
        "run_id": resolved_run_id,
        "launch_id": resolved_run_id,
        "run_name": resolved_run_name,
        "run_description": resolved_run_description,
        "run_goal": resolved_run_goal,
        "baseline_registry": None
        if baseline_registry is None
        else str(baseline_registry.resolve()),
        "fail_on_baseline_missing": fail_on_baseline_missing,
        "fail_on_validation_fail": fail_on_validation_fail,
        "sbatch_argv": command_parts,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_command": shlex.join(command_parts),
    }


def _path_in_workspace(path: Path, *, workspace: Path, source_root: Path) -> Path:
    resolved_path = path.resolve()
    try:
        return workspace / resolved_path.relative_to(source_root.resolve())
    except ValueError:
        return resolved_path


def render_fixture_prep_plan(
    *,
    cluster: str,
    target_spec_file: Path = DEFAULT_WAVE0_FIXTURE_TARGET_SPEC,
    output_dir: Path = DEFAULT_WAVE0_FIXTURE_OUTPUT_DIR,
    decoder_chunk_size: int = 256,
    cross_batch_decoder_cache_bytes: int | None = None,
    immutable_workspace: bool = False,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    walltime: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    if cluster not in SBATCH_FIXTURE_PREP_SCRIPTS:
        raise ValueError(f"Unsupported fixture prep cluster: {cluster!r}")
    if decoder_chunk_size <= 0:
        raise ValueError("decoder_chunk_size must be positive")
    if (
        cross_batch_decoder_cache_bytes is not None
        and cross_batch_decoder_cache_bytes < 0
    ):
        raise ValueError("cross_batch_decoder_cache_bytes must be non-negative")

    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)

    source_root = source_root.resolve()
    script_path = SBATCH_FIXTURE_PREP_SCRIPTS[cluster].resolve()
    launch_script_path = _path_in_workspace(
        script_path,
        workspace=workspace,
        source_root=source_root,
    )
    launch_target_spec = _path_in_workspace(
        target_spec_file,
        workspace=workspace,
        source_root=source_root,
    )
    resolved_output_dir = output_dir.resolve()
    resolved_run_name = (
        _normalize_free_text(run_name) or f"wave0 fixture prep {cluster}"
    )

    export_parts = [
        "ALL",
        f"TARGET_SPEC_FILE={launch_target_spec}",
        f"OUTPUT_DIR={resolved_output_dir}",
        f"DECODER_CHUNK_SIZE={decoder_chunk_size}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
    ]
    if cross_batch_decoder_cache_bytes is not None:
        export_parts.append(
            f"CROSS_BATCH_DECODER_CACHE_BYTES={cross_batch_decoder_cache_bytes}"
        )
    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--job-name={_slugify_run_name(resolved_run_name)}",
        f"--export={','.join(export_parts)}",
        str(launch_script_path),
    ]
    return {
        "cluster": cluster,
        "target_spec_file": str(launch_target_spec),
        "output_dir": str(resolved_output_dir),
        "decoder_chunk_size": decoder_chunk_size,
        "cross_batch_decoder_cache_bytes": cross_batch_decoder_cache_bytes,
        "run_name": resolved_run_name,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }


def render_full_answer_trajectory_plan(
    *,
    cluster: str,
    output: Path,
    max_new_tokens: int,
    prompt_path: Path | None = None,
    fixture_catalog: Path | None = None,
    fixture_name: str | None = None,
    trajectory_id: str | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    include_prompt_text: bool = False,
    immutable_workspace: bool = True,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    walltime: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    if cluster not in SBATCH_FULL_ANSWER_TRAJECTORY_SCRIPTS:
        raise ValueError(f"Unsupported full-answer trajectory cluster: {cluster!r}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt_path is None and (fixture_catalog is None or fixture_name is None):
        raise ValueError("provide prompt_path or fixture_catalog + fixture_name")

    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)
    source_root = source_root.resolve()
    script_path = SBATCH_FULL_ANSWER_TRAJECTORY_SCRIPTS[cluster].resolve()
    launch_script_path = _path_in_workspace(
        script_path,
        workspace=workspace,
        source_root=source_root,
    )

    export_parts = [
        "ALL",
        f"OUTPUT_TRAJECTORY={output.resolve()}",
        f"MAX_NEW_TOKENS={max_new_tokens}",
        f"TEMPERATURE={temperature}",
        f"INCLUDE_PROMPT_TEXT={1 if include_prompt_text else 0}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
    ]
    if prompt_path is not None:
        export_parts.append(
            f"PROMPT_PATH={_path_in_workspace(prompt_path, workspace=workspace, source_root=source_root)}"
        )
    if fixture_catalog is not None:
        export_parts.append(
            f"FIXTURE_CATALOG={_path_in_workspace(fixture_catalog, workspace=workspace, source_root=source_root)}"
        )
    if fixture_name is not None:
        export_parts.append(f"FIXTURE_NAME={fixture_name}")
    if trajectory_id is not None:
        export_parts.append(f"TRAJECTORY_ID={trajectory_id}")
    if seed is not None:
        export_parts.append(f"SEED={seed}")
    resolved_run_name = (
        _normalize_free_text(run_name) or f"full answer trajectory {cluster}"
    )
    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--job-name={_slugify_run_name(resolved_run_name)}",
        f"--export={','.join(export_parts)}",
        str(launch_script_path),
    ]
    return {
        "cluster": cluster,
        "output": str(output.resolve()),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "include_prompt_text": include_prompt_text,
        "prompt_path": None if prompt_path is None else str(prompt_path),
        "fixture_catalog": None if fixture_catalog is None else str(fixture_catalog),
        "fixture_name": fixture_name,
        "trajectory_id": trajectory_id,
        "run_name": resolved_run_name,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }


def render_full_answer_shard_plan(
    *,
    cluster: str,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    output_root: Path,
    immutable_workspace: bool = True,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    walltime: str | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
) -> dict[str, Any]:
    if cluster not in SBATCH_FULL_ANSWER_TRACE_SCRIPTS:
        raise ValueError(f"Unsupported full-answer trace cluster: {cluster!r}")
    shards_payload = load_shards(shards_path)
    shards = shards_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"No shards found in {shards_path}")
    shard_count = len(shards)
    array_range = f"0-{shard_count - 1}"

    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)
    source_root = source_root.resolve()
    script_path = SBATCH_FULL_ANSWER_TRACE_SCRIPTS[cluster].resolve()
    launch_script_path = _path_in_workspace(
        script_path, workspace=workspace, source_root=source_root
    )
    launch_trajectory = _path_in_workspace(
        trajectory_path, workspace=workspace, source_root=source_root
    )
    launch_trace_specs = _path_in_workspace(
        trace_specs_path, workspace=workspace, source_root=source_root
    )
    launch_shards = _path_in_workspace(
        shards_path, workspace=workspace, source_root=source_root
    )

    resolved_run_name = _normalize_free_text(run_name) or f"full answer trace {cluster}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    resolved_run_id = (
        _normalize_run_id(run_id)
        if run_id
        else f"{timestamp}_{_slugify_run_name(resolved_run_name)}"
    )
    resolved_output_root = (output_root.resolve() / resolved_run_id).resolve()
    if resolved_output_root.exists():
        raise ValueError(
            f"Launch output root already exists: {resolved_output_root}. "
            "Resume mode is not implemented."
        )
    export_parts = [
        "ALL",
        f"TRAJECTORY_PATH={launch_trajectory}",
        f"TRACE_SPECS_PATH={launch_trace_specs}",
        f"SHARDS_PATH={launch_shards}",
        f"OUTPUT_ROOT={resolved_output_root}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
    ]
    script_args: list[str] = [
        "--run-id",
        resolved_run_id,
        "--run-name",
        resolved_run_name,
    ]
    if run_description:
        script_args.extend(
            ["--run-description", _normalize_free_text(run_description) or ""]
        )
    if run_goal:
        script_args.extend(["--run-goal", _normalize_free_text(run_goal) or ""])
    command_parts = [
        "sbatch",
        *([f"--time={walltime}"] if walltime else []),
        f"--job-name={_slugify_run_name(resolved_run_name)}",
        f"--array={array_range}",
        f"--export={','.join(export_parts)}",
        str(launch_script_path),
        *script_args,
    ]
    return {
        "cluster": cluster,
        "trajectory_path": str(launch_trajectory),
        "trace_specs_path": str(launch_trace_specs),
        "shards_path": str(launch_shards),
        "output_root": str(resolved_output_root),
        "shard_count": shard_count,
        "array_range": array_range,
        "run_id": resolved_run_id,
        "run_name": resolved_run_name,
        "run_description": _normalize_free_text(run_description),
        "run_goal": _normalize_free_text(run_goal),
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }
