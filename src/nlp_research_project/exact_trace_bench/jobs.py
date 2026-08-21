from __future__ import annotations

import os
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
from .full_answer.launch_spec import build_full_answer_launch_spec
from .full_answer.schemas import load_shards, load_trace_specs
from .io_utils import read_json
from .scenarios import (
    CHPC_BASELINE_RESOURCE_PROFILE,
    RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM,
    RESOURCE_PROFILE_STANDARD,
)
from .workspace import (
    DEFAULT_SNAPSHOT_ROOT,
    repo_state,
    resolve_launch_workspace,
    sibling_library_root,
    validate_launch_snapshot,
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
    ("granite", RESOURCE_PROFILE_STANDARD): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked.granite.sbatch",
    ("granite", RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_weekend_exact_chunked_long_eval.granite.sbatch",
    ("granite", CHPC_BASELINE_RESOURCE_PROFILE): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_baseline_h200.granite.sbatch",
    ("granite", "governor_calibration_h200"): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "trace_baseline_h200.granite.sbatch",
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
    "granite": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "prepare_weekend_prefix_fixtures.granite.sbatch",
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
    "granite": REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_prepare.granite.sbatch",
}

FULL_ANSWER_TRACE_RESOURCE_PROFILES: tuple[str, ...] = ("standard", "quad")

SBATCH_FULL_ANSWER_TRACE_SCRIPTS: dict[tuple[str, str], Path] = {
    ("ascend", "standard"): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace.ascend.sbatch",
    ("ascend", "quad"): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace_quad.ascend.sbatch",
    ("cardinal", "standard"): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace.cardinal.sbatch",
    ("granite", "standard"): REPO_ROOT
    / "slurm"
    / "exact_trace_bench"
    / "full_answer_trace.granite.sbatch",
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


def _external_uv_environment() -> Path:
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    return Path(configured).resolve() if configured else (REPO_ROOT / ".venv").resolve()


def _external_env_file() -> Path:
    configured = os.environ.get("ENV_FILE")
    return Path(configured).resolve() if configured else (REPO_ROOT / ".env").resolve()


def _validated_workspace_provenance(
    *,
    workspace: Path,
    library_workspace: Path | None,
    immutable_workspace: bool,
    live_workspace_rationale: str | None,
) -> dict[str, Any]:
    if library_workspace is None:
        raise ValueError(f"Sibling library workspace does not exist for {workspace}")
    if immutable_workspace:
        return validate_launch_snapshot(
            workspace_root=workspace,
            library_root=library_workspace,
            import_roots=(workspace / "src", workspace, library_workspace),
        )
    rationale = _normalize_free_text(live_workspace_rationale)
    if rationale is None:
        raise ValueError("Live workspace launches require a non-empty rationale")
    return {
        "workspace_mode": "live",
        "manifest_path": None,
        "workspace_root": str(workspace),
        "library_workspace_root": str(library_workspace.resolve()),
        "live_workspace_rationale": rationale,
        "project_repo_state": repo_state(workspace),
        "library_repo_state": repo_state(library_workspace),
    }


def _append_workspace_exports(
    export_parts: list[str], provenance: dict[str, Any]
) -> None:
    export_parts.extend(
        [
            f"EXACT_TRACE_WORKSPACE_MODE={provenance['workspace_mode']}",
            f"EXACT_TRACE_WORKSPACE_MANIFEST={provenance['manifest_path'] or ''}",
        ]
    )
    if provenance["workspace_mode"] == "live":
        export_parts.extend(
            [
                "EXACT_TRACE_ALLOW_LIVE_WORKSPACE=1",
                "EXACT_TRACE_LIVE_WORKSPACE_RATIONALE="
                f"{provenance['live_workspace_rationale']}",
            ]
        )


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


def _validate_array_range(array_range: str, *, item_count: int) -> str:
    normalized = array_range.strip()
    if not normalized:
        raise ValueError("array_range cannot be empty")
    if item_count <= 0:
        raise ValueError("item_count must be positive")

    for part in normalized.split(","):
        range_part = part.strip()
        if not range_part:
            raise ValueError(f"Invalid empty array range component in {array_range!r}")
        if "%" in range_part:
            range_part, throttle = range_part.split("%", maxsplit=1)
            if not throttle.isdigit() or int(throttle) <= 0:
                raise ValueError(f"Invalid array throttle in {array_range!r}")
        if "-" in range_part:
            start_text, end_text = range_part.split("-", maxsplit=1)
        else:
            start_text = end_text = range_part
        if not start_text.isdigit() or not end_text.isdigit():
            raise ValueError(f"Invalid array range component {part!r}")
        start = int(start_text)
        end = int(end_text)
        if start > end:
            raise ValueError(f"Array range start exceeds end in {part!r}")
        if start < 0 or end >= item_count:
            raise ValueError(
                f"Array range {part!r} is outside available shard ids 0-{item_count - 1}"
            )
    return normalized


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
    immutable_workspace: bool = True,
    existing_workspace: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    live_workspace_rationale: str | None = None,
    _pending_snapshot_freeze: bool = False,
    walltime: str | None = None,
    mem: str | None = None,
    baseline_registry: Path | None = None,
    fail_on_baseline_missing: bool = False,
    fail_on_validation_fail: bool = False,
) -> dict[str, Any]:
    normalized_live_rationale = _normalize_free_text(live_workspace_rationale)
    if existing_workspace is not None and not immutable_workspace:
        raise ValueError("existing_workspace requires immutable_workspace")
    if not immutable_workspace and not _pending_snapshot_freeze:
        if normalized_live_rationale is None:
            raise ValueError("Live workspace launches require a non-empty rationale")
    scenarios_file = scenarios_file.resolve()
    resource_profile = _resolve_resource_profile(scenarios_file)
    script_key = (cluster, resource_profile)
    if script_key not in SBATCH_SCRIPTS:
        raise ValueError(
            f"Unsupported launch profile for cluster={cluster!r}, resource_profile={resource_profile!r}"
        )

    scenarios_payload = read_json(scenarios_file)
    scenarios_metadata = scenarios_payload.get("metadata") or {}
    if baseline_registry is None:
        metadata_registry = scenarios_metadata.get("baseline_registry")
        if isinstance(metadata_registry, str) and metadata_registry:
            baseline_registry = Path(metadata_registry)
    fail_on_baseline_missing = fail_on_baseline_missing or bool(
        scenarios_metadata.get("fail_on_baseline_missing", False)
    )
    fail_on_validation_fail = fail_on_validation_fail or bool(
        scenarios_metadata.get("fail_on_validation_fail", False)
    )
    slurm_metadata = scenarios_metadata.get("slurm") or {}
    if not isinstance(slurm_metadata, dict):
        raise ValueError("metadata.slurm must be an object")

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

    workspace = (
        existing_workspace.resolve()
        if existing_workspace is not None
        else resolve_launch_workspace(
            immutable=immutable_workspace,
            snapshot_root=snapshot_root,
            source_root=source_root,
            label=workspace_label,
        ).resolve()
    )
    library_workspace = sibling_library_root(workspace)
    if library_workspace is None:
        raise ValueError(f"Sibling library workspace does not exist for {workspace}")

    if immutable_workspace:
        workspace_provenance = validate_launch_snapshot(
            workspace_root=workspace,
            library_root=library_workspace,
            import_roots=(workspace / "src", workspace, library_workspace),
        )
    elif _pending_snapshot_freeze:
        workspace_provenance = {
            "workspace_mode": "immutable",
            "manifest_path": str(
                (workspace.parent / ".exact_trace_bench_snapshot.json").resolve()
            ),
            "workspace_root": str(workspace),
            "library_workspace_root": str(library_workspace.resolve()),
            "read_only": False,
            "pending_freeze": True,
        }
    else:
        workspace_provenance = {
            "workspace_mode": "live",
            "manifest_path": None,
            "workspace_root": str(workspace),
            "library_workspace_root": str(library_workspace.resolve()),
            "live_workspace_rationale": normalized_live_rationale,
            "project_repo_state": repo_state(workspace),
            "library_repo_state": repo_state(library_workspace),
        }

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
    array_concurrency = scenarios_metadata.get("array_concurrency")
    if array_concurrency is not None:
        if not isinstance(array_concurrency, int) or array_concurrency <= 0:
            raise ValueError("metadata.array_concurrency must be a positive integer")
        array_range = f"{array_range}%{array_concurrency}"
    export_parts = [
        "ALL",
        f"SCENARIOS_FILE={launch_scenarios_file}",
        f"OUTPUT_ROOT={resolved_output_root}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
        f"UV_PROJECT_ENVIRONMENT={_external_uv_environment()}",
        f"ENV_FILE={_external_env_file()}",
        f"EXACT_TRACE_WORKSPACE_MODE={workspace_provenance['workspace_mode']}",
        f"EXACT_TRACE_WORKSPACE_MANIFEST={workspace_provenance['manifest_path'] or ''}",
    ]
    if workspace_provenance["workspace_mode"] == "live":
        export_parts.extend(
            [
                "EXACT_TRACE_ALLOW_LIVE_WORKSPACE=1",
                f"EXACT_TRACE_LIVE_WORKSPACE_RATIONALE={normalized_live_rationale}",
            ]
        )
    launch_baseline_registry = (
        None
        if baseline_registry is None
        else _path_in_workspace(
            baseline_registry, workspace=workspace, source_root=source_root
        )
    )
    if launch_baseline_registry is not None:
        export_parts.append(f"BASELINE_REGISTRY={launch_baseline_registry}")
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

    resolved_walltime = walltime or slurm_metadata.get("time")
    resolved_mem = mem or slurm_metadata.get("mem")
    scheduler_args = [
        *(
            [f"--account={slurm_metadata['account']}"]
            if slurm_metadata.get("account")
            else []
        ),
        *(
            [f"--partition={slurm_metadata['partition']}"]
            if slurm_metadata.get("partition")
            else []
        ),
        *([f"--qos={slurm_metadata['qos']}"] if slurm_metadata.get("qos") else []),
        *([f"--gres={slurm_metadata['gres']}"] if slurm_metadata.get("gres") else []),
        *(
            [f"--cpus-per-task={slurm_metadata['cpus_per_task']}"]
            if slurm_metadata.get("cpus_per_task")
            else []
        ),
    ]
    command_parts = [
        "sbatch",
        *scheduler_args,
        *([f"--time={resolved_walltime}"] if resolved_walltime else []),
        *([f"--mem={resolved_mem}"] if resolved_mem else []),
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
        "array_range": array_range,
        "run_id": resolved_run_id,
        "launch_id": resolved_run_id,
        "run_name": resolved_run_name,
        "run_description": resolved_run_description,
        "run_goal": resolved_run_goal,
        "mem": resolved_mem,
        "walltime": resolved_walltime,
        "baseline_registry": None
        if launch_baseline_registry is None
        else str(launch_baseline_registry),
        "fail_on_baseline_missing": fail_on_baseline_missing,
        "fail_on_validation_fail": fail_on_validation_fail,
        "sbatch_argv": command_parts,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "workspace_provenance": workspace_provenance,
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
    immutable_workspace: bool = True,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    live_workspace_rationale: str | None = None,
    walltime: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    normalized_live_rationale = _normalize_free_text(live_workspace_rationale)
    if not immutable_workspace and normalized_live_rationale is None:
        raise ValueError("Live workspace launches require a non-empty rationale")
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
    if library_workspace is None:
        raise ValueError(f"Sibling library workspace does not exist for {workspace}")
    if immutable_workspace:
        workspace_provenance = validate_launch_snapshot(
            workspace_root=workspace,
            library_root=library_workspace,
            import_roots=(workspace / "src", workspace, library_workspace),
        )
    else:
        workspace_provenance = {
            "workspace_mode": "live",
            "manifest_path": None,
            "workspace_root": str(workspace),
            "library_workspace_root": str(library_workspace.resolve()),
            "live_workspace_rationale": normalized_live_rationale,
            "project_repo_state": repo_state(workspace),
            "library_repo_state": repo_state(library_workspace),
        }

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
        f"UV_PROJECT_ENVIRONMENT={_external_uv_environment()}",
        f"ENV_FILE={_external_env_file()}",
        f"EXACT_TRACE_WORKSPACE_MODE={workspace_provenance['workspace_mode']}",
        f"EXACT_TRACE_WORKSPACE_MANIFEST={workspace_provenance['manifest_path'] or ''}",
    ]
    if workspace_provenance["workspace_mode"] == "live":
        export_parts.extend(
            [
                "EXACT_TRACE_ALLOW_LIVE_WORKSPACE=1",
                f"EXACT_TRACE_LIVE_WORKSPACE_RATIONALE={normalized_live_rationale}",
            ]
        )
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
        "workspace_provenance": workspace_provenance,
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }


def render_full_answer_trajectory_plan(
    *,
    cluster: str,
    output: Path,
    max_new_tokens: int,
    sample_until_success: bool = False,
    collect_all: bool = False,
    expected_answer: str | None = None,
    max_success_tokens: int | None = None,
    max_attempts: int | None = None,
    time_limit_seconds: float | None = None,
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
    live_workspace_rationale: str | None = None,
    walltime: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    if cluster not in SBATCH_FULL_ANSWER_TRAJECTORY_SCRIPTS:
        raise ValueError(f"Unsupported full-answer trajectory cluster: {cluster!r}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if sample_until_success:
        if expected_answer is None:
            raise ValueError("expected_answer is required for sampling")
        if max_success_tokens is None or max_success_tokens <= 0:
            raise ValueError("max_success_tokens must be positive for sampling")
        if max_attempts is None or max_attempts <= 0:
            raise ValueError("max_attempts must be positive for sampling")
    if prompt_path is None and (fixture_catalog is None or fixture_name is None):
        raise ValueError("provide prompt_path or fixture_catalog + fixture_name")

    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)
    workspace_provenance = _validated_workspace_provenance(
        workspace=workspace,
        library_workspace=library_workspace,
        immutable_workspace=immutable_workspace,
        live_workspace_rationale=live_workspace_rationale,
    )
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
        f"OUTPUT_DIR={output.resolve()}",
        f"SAMPLE_UNTIL_SUCCESS={1 if sample_until_success else 0}",
        f"COLLECT_ALL={1 if collect_all else 0}",
        f"MAX_NEW_TOKENS={max_new_tokens}",
        f"TEMPERATURE={temperature}",
        f"INCLUDE_PROMPT_TEXT={1 if include_prompt_text else 0}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
        f"UV_PROJECT_ENVIRONMENT={_external_uv_environment()}",
        f"ENV_FILE={_external_env_file()}",
    ]
    _append_workspace_exports(export_parts, workspace_provenance)
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
    if expected_answer is not None:
        export_parts.append(f"EXPECTED_ANSWER={expected_answer}")
    if max_success_tokens is not None:
        export_parts.append(f"MAX_SUCCESS_TOKENS={max_success_tokens}")
    if max_attempts is not None:
        export_parts.append(f"MAX_ATTEMPTS={max_attempts}")
    if time_limit_seconds is not None:
        export_parts.append(f"TIME_LIMIT_SECONDS={time_limit_seconds}")
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
        "sample_until_success": sample_until_success,
        "collect_all": collect_all,
        "expected_answer": expected_answer,
        "max_success_tokens": max_success_tokens,
        "max_attempts": max_attempts,
        "time_limit_seconds": time_limit_seconds,
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
        "workspace_provenance": workspace_provenance,
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }


def render_full_answer_shard_plan(
    *,
    cluster: str,
    resource_profile: str = "standard",
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    output_root: Path,
    array_range: str | None = None,
    immutable_workspace: bool = True,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    source_root: Path = REPO_ROOT,
    workspace_label: str | None = None,
    live_workspace_rationale: str | None = None,
    walltime: str | None = None,
    mem: str | None = None,
    partition: str | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
) -> dict[str, Any]:
    script_key = (cluster, resource_profile)
    if script_key not in SBATCH_FULL_ANSWER_TRACE_SCRIPTS:
        raise ValueError(
            "Unsupported full-answer trace launch profile: "
            f"cluster={cluster!r}, resource_profile={resource_profile!r}"
        )
    shards_payload = load_shards(shards_path)
    shards = shards_payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"No shards found in {shards_path}")
    shard_count = len(shards)
    resolved_array_range = (
        _validate_array_range(array_range, item_count=shard_count)
        if array_range is not None
        else f"0-{shard_count - 1}"
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

    workspace = resolve_launch_workspace(
        immutable=immutable_workspace,
        snapshot_root=snapshot_root,
        source_root=source_root,
        label=workspace_label,
    ).resolve()
    library_workspace = sibling_library_root(workspace)
    workspace_provenance = _validated_workspace_provenance(
        workspace=workspace,
        library_workspace=library_workspace,
        immutable_workspace=immutable_workspace,
        live_workspace_rationale=live_workspace_rationale,
    )
    source_root = source_root.resolve()
    script_path = SBATCH_FULL_ANSWER_TRACE_SCRIPTS[script_key].resolve()
    sbatch_defaults: dict[str, str] = {}
    for line in script_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH --") or "=" not in stripped:
            continue
        key, value = stripped.removeprefix("#SBATCH --").split("=", maxsplit=1)
        sbatch_defaults[key] = value
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

    scheduler_request = {
        "cluster": cluster,
        "resource_profile": resource_profile,
        "account": sbatch_defaults.get("account"),
        "partition": partition or sbatch_defaults.get("partition"),
        "qos": sbatch_defaults.get("qos"),
        "gpus_per_task": sbatch_defaults.get("gpus-per-task"),
        "cpus_per_task": sbatch_defaults.get("cpus-per-task"),
        "memory": mem or sbatch_defaults.get("mem"),
        "walltime": walltime or sbatch_defaults.get("time"),
    }
    specs = load_trace_specs(trace_specs_path)
    launch_spec = None
    if specs:
        first_knobs = specs[0]["graph_knobs"]
        launch_spec = build_full_answer_launch_spec(
            trajectory_path=launch_trajectory,
            trace_specs_path=launch_trace_specs,
            shards_path=launch_shards,
            output_root=resolved_output_root,
            shard_selection=resolved_array_range,
            run={
                "run_id": resolved_run_id,
                "run_name": resolved_run_name,
                "run_description": _normalize_free_text(run_description),
                "run_goal": _normalize_free_text(run_goal),
            },
            specs=specs,
            planning_envelope=first_knobs.get("resource_planning_envelope", {}),
            scheduler_request=scheduler_request,
            runtime_resource_policy=first_knobs.get("runtime_resource_policy", "off"),
            runtime_resource_override_rationale=None,
            preheat={"policy": "none", "paths": []},
            workspace={
                "policy": (
                    "immutable_snapshot" if immutable_workspace else "live_override"
                ),
                "project_root": str(workspace),
                "library_root": (
                    None if library_workspace is None else str(library_workspace)
                ),
                "provenance": workspace_provenance,
            },
            monitoring={
                "gpu_sampler": False,
                "runtime_resource_samples": (
                    first_knobs.get("runtime_resource_policy", "off") != "off"
                ),
                "incremental_trace_telemetry": bool(
                    first_knobs.get("incremental_telemetry_jsonl", False)
                ),
                "postrun_mechanism_validation": True,
            },
        )

    export_parts = [
        "ALL",
        f"TRAJECTORY_PATH={launch_trajectory}",
        f"TRACE_SPECS_PATH={launch_trace_specs}",
        f"SHARDS_PATH={launch_shards}",
        f"OUTPUT_ROOT={resolved_output_root}",
        f"WORKSPACE_ROOT={workspace}",
        f"LIB_WORKSPACE_ROOT={library_workspace or ''}",
        f"UV_PROJECT_ENVIRONMENT={_external_uv_environment()}",
        f"ENV_FILE={_external_env_file()}",
        f"EXACT_TRACE_REQUESTED_CLUSTER={cluster}",
        f"EXACT_TRACE_REQUESTED_RESOURCE_PROFILE={resource_profile}",
        f"EXACT_TRACE_REQUESTED_ACCOUNT={scheduler_request['account'] or ''}",
        f"EXACT_TRACE_REQUESTED_PARTITION={scheduler_request['partition'] or ''}",
        f"EXACT_TRACE_REQUESTED_QOS={scheduler_request['qos'] or ''}",
        f"EXACT_TRACE_REQUESTED_GPUS_PER_TASK={scheduler_request['gpus_per_task'] or ''}",
        f"EXACT_TRACE_REQUESTED_CPUS={scheduler_request['cpus_per_task'] or ''}",
        f"EXACT_TRACE_REQUESTED_MEM={scheduler_request['memory'] or ''}",
        f"EXACT_TRACE_REQUESTED_WALLTIME={scheduler_request['walltime'] or ''}",
    ]
    _append_workspace_exports(export_parts, workspace_provenance)
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
        *([f"--mem={mem}"] if mem else []),
        *([f"--partition={partition}"] if partition else []),
        f"--job-name={_slugify_run_name(resolved_run_name)}",
        f"--array={resolved_array_range}",
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
        "array_range": resolved_array_range,
        "resource_profile": resource_profile,
        "run_id": resolved_run_id,
        "run_name": resolved_run_name,
        "run_description": _normalize_free_text(run_description),
        "run_goal": _normalize_free_text(run_goal),
        "mem": mem,
        "partition": partition,
        "workspace_root": str(workspace),
        "library_workspace_root": None
        if library_workspace is None
        else str(library_workspace),
        "immutable_workspace": immutable_workspace,
        "workspace_provenance": workspace_provenance,
        "launch_spec": None if launch_spec is None else launch_spec.to_record(),
        "sbatch_script": str(launch_script_path),
        "sbatch_argv": command_parts,
        "sbatch_command": shlex.join(command_parts),
    }
