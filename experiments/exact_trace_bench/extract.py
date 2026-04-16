from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import DEFAULT_EXTRACTED_DIR, DEFAULT_LOGS_DIR, DEFAULT_SCRATCH_ROOT
from .io_utils import (
    ensure_dir,
    flatten_dict,
    parse_memory_value_to_gib,
    read_json,
    safe_stem,
    write_csv,
    write_jsonl,
)


MEMORY_RE = re.compile(
    r"rss=(?P<rss>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_alloc=(?P<cuda_alloc>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_reserved=(?P<cuda_reserved>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_peak_alloc=(?P<cuda_peak_alloc>n/a|\d+(?:\.\d+)?) GiB, "
    r"cuda_peak_reserved=(?P<cuda_peak_reserved>n/a|\d+(?:\.\d+)?) GiB"
)
PHASE0_ENCODE_DONE_RE = re.compile(
    r"TRACE phase0\.encode_sparse\.done \| "
    r"total_active_features=(?P<active_features>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE0_RECON_DONE_RE = re.compile(
    r"TRACE phase0\.reconstruction\.done \| total_chunks=(?P<chunks>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE3_DONE_RE = re.compile(
    r"(?P<count>\d+) logit attribution\(s\) completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE4_DONE_RE = re.compile(
    r"Feature attributions completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
ATTRIBUTION_DONE_RE = re.compile(
    r"Attribution completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE4_BATCH_RE = re.compile(
    r"Phase 4 batch (?P<batch_idx>\d+)/(?P<total_batches>\d+) in "
    r"(?P<seconds>\d+(?:\.\d+)?)s"
)
CACHE_EVENT_RE = re.compile(
    r"TRACE decoder\.cache\.(?P<event>hit|miss|eviction) \| .*?"
    r"(?P<counter_name>hit_count|miss_count|eviction_count)=(?P<count>\d+)"
    r"(?:, resident_bytes=(?P<resident_bytes>\d+))?"
)
CUDA_OOM_RE = re.compile(
    r"torch\.OutOfMemoryError: CUDA out of memory\. Tried to allocate "
    r"(?P<requested>[0-9.]+\s(?:GiB|MiB|KiB))\. "
    r"GPU 0 has a total capacity of (?P<total>[0-9.]+\s(?:GiB|MiB|KiB)) "
    r"of which (?P<free>[0-9.]+\s(?:GiB|MiB|KiB)) is free\. "
    r"Including non-PyTorch memory, this process has "
    r"(?P<in_use>[0-9.]+\s(?:GiB|MiB|KiB)) memory in use\."
)

JOB_ID_RE = re.compile(r"Job ID: (?P<job_id>\d+)")
ARRAY_TASK_RE = re.compile(r"Array task: (?P<array_task>\S+)")
NODE_RE = re.compile(r"Node: (?P<node>.+)")
CLUSTER_RE = re.compile(r"Cluster: (?P<cluster>.+)")
SCENARIOS_FILE_RE = re.compile(r"Scenarios file: (?P<scenarios_file>.+)")
OUTPUT_ROOT_RE = re.compile(r"Output root: (?P<output_root>.+)")
RUN_ID_RE = re.compile(r"Run ID: (?P<run_id>.+)")
RUN_NAME_RE = re.compile(r"Run name: (?P<run_name>.+)")
RUN_DESCRIPTION_RE = re.compile(r"Run description: (?P<run_description>.+)")
RUN_GOAL_RE = re.compile(r"Run goal: (?P<run_goal>.+)")
WRITING_DIR_RE = re.compile(r"Writing experiment results to (?P<scenario_root>.+)")
RUNNING_SCENARIO_RE = re.compile(r"Running scenario: (?P<scenario_name>.+)")
OOM_KILL_RE = re.compile(r"Detected (?P<count>\d+) oom_kill events")
TIMEOUT_RE = re.compile(r"time limit|timed out", re.IGNORECASE)


def _max_or_none(
    current: float | int | None,
    candidate: float | int | None,
) -> float | int | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _infer_cluster(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
    return None


def _load_prompt_meta(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta_files = sorted(artifact_dir.glob("prompt_*/prompt_meta.json"))
    if not prompt_meta_files:
        return {}
    return read_json(prompt_meta_files[0])


def _load_completion_manifests(artifact_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("prompt_*/completion_*/completion.json")):
        manifests.append(read_json(path))
    return manifests


def _load_run_config(artifact_dir: Path) -> dict[str, Any]:
    run_config_path = artifact_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    return read_json(run_config_path)


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_optional_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "unset":
            return None
        return normalized
    return value


def _load_summary_run_metadata(scenario_root: Path) -> dict[str, Any]:
    candidates = [scenario_root / "summary.json", scenario_root.parent / "summary.json"]
    for summary_path in candidates:
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        run_metadata = summary.get("run_metadata")
        if isinstance(run_metadata, dict):
            return run_metadata
        if any(
            key in summary
            for key in (
                "run_id",
                "launch_id",
                "run_name",
                "run_description",
                "run_goal",
            )
        ):
            return {
                "run_id": summary.get("run_id") or summary.get("launch_id"),
                "run_name": summary.get("run_name"),
                "run_description": summary.get("run_description"),
                "run_goal": summary.get("run_goal"),
            }
    return {}


def _summarize_artifacts(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta = _load_prompt_meta(artifact_dir)
    manifests = _load_completion_manifests(artifact_dir)
    steps = [
        step
        for manifest in manifests
        for step in manifest.get("steps", [])
        if isinstance(step, dict)
    ]
    diagnostics = [
        step.get("transcoder_diagnostics", {})
        for step in steps
        if isinstance(step.get("transcoder_diagnostics"), dict)
    ]

    first_manifest = manifests[0] if manifests else {}
    resource_snapshot = first_manifest.get("resource_snapshot")

    return {
        "prompt_count": len(list(artifact_dir.glob("prompt_*"))),
        "completion_count": len(manifests),
        "gsm8k_index": prompt_meta.get("gsm8k_index"),
        "prompt_source": first_manifest.get(
            "prompt_source", prompt_meta.get("prompt_source")
        ),
        "fixture_name": first_manifest.get(
            "fixture_name", prompt_meta.get("fixture_name")
        ),
        "fixture_kind": first_manifest.get(
            "fixture_kind", prompt_meta.get("fixture_kind")
        ),
        "prepared_prompt_file": prompt_meta.get("prepared_prompt_file"),
        "prepared_prompt_meta_file": prompt_meta.get("prepared_prompt_meta_file"),
        "prompt_token_count": first_manifest.get(
            "prompt_token_count", prompt_meta.get("prompt_token_count")
        ),
        "initial_input_token_count": first_manifest.get(
            "initial_input_token_count",
            prompt_meta.get("initial_input_token_count"),
        ),
        "generated_token_count": first_manifest.get("generated_token_count"),
        "completion_duration_seconds": first_manifest.get("duration_seconds"),
        "n_steps_traced": first_manifest.get("n_steps_traced"),
        "max_active_features": max(
            (
                step.get("n_active_features")
                for step in steps
                if step.get("n_active_features") is not None
            ),
            default=None,
        ),
        "max_edges_retained": max(
            (
                step.get("n_edges_retained")
                for step in steps
                if step.get("n_edges_retained") is not None
            ),
            default=None,
        ),
        "decoder_cache_hit_count": max(
            (
                diag.get("decoder_cache_hit_count")
                for diag in diagnostics
                if diag.get("decoder_cache_hit_count") is not None
            ),
            default=None,
        ),
        "decoder_cache_miss_count": max(
            (
                diag.get("decoder_cache_miss_count")
                for diag in diagnostics
                if diag.get("decoder_cache_miss_count") is not None
            ),
            default=None,
        ),
        "decoder_cache_eviction_count": max(
            (
                diag.get("decoder_cache_eviction_count")
                for diag in diagnostics
                if diag.get("decoder_cache_eviction_count") is not None
            ),
            default=None,
        ),
        **flatten_dict(resource_snapshot, prefix="resource_snapshot_"),
    }


def build_benchmark_index_row(result_path: Path) -> dict[str, Any]:
    scenario_root = result_path.parent
    result = read_json(result_path)
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.exists() else {}
    artifact_dir = Path(result.get("output_dir") or scenario_root / "artifacts")
    run_config = _load_run_config(artifact_dir)
    summary_run_metadata = _load_summary_run_metadata(scenario_root)
    scenario_run_metadata_raw = scenario.get("run_metadata")
    scenario_run_metadata = (
        scenario_run_metadata_raw if isinstance(scenario_run_metadata_raw, dict) else {}
    )
    result_run_metadata_raw = result.get("run_metadata")
    result_run_metadata = (
        result_run_metadata_raw if isinstance(result_run_metadata_raw, dict) else {}
    )
    run_id = _first_non_null(
        result.get("run_id"),
        result.get("launch_id"),
        result_run_metadata.get("run_id"),
        result_run_metadata.get("launch_id"),
        scenario_run_metadata.get("run_id"),
        scenario_run_metadata.get("launch_id"),
        summary_run_metadata.get("run_id"),
        summary_run_metadata.get("launch_id"),
    )
    run_name = _first_non_null(
        result.get("run_name"),
        result_run_metadata.get("run_name"),
        scenario_run_metadata.get("run_name"),
        summary_run_metadata.get("run_name"),
    )
    run_description = _first_non_null(
        result.get("run_description"),
        result_run_metadata.get("run_description"),
        scenario_run_metadata.get("run_description"),
        summary_run_metadata.get("run_description"),
    )
    run_goal = _first_non_null(
        result.get("run_goal"),
        result_run_metadata.get("run_goal"),
        scenario_run_metadata.get("run_goal"),
        summary_run_metadata.get("run_goal"),
    )

    profiling = result.get("profiling_summary", {})
    cache_bytes = run_config.get(
        "cross_batch_decoder_cache_bytes",
        scenario.get("cross_batch_decoder_cache_bytes"),
    )

    return {
        "scenario_root": str(scenario_root),
        "scenario_name": result.get("name")
        or scenario.get("name")
        or scenario_root.name,
        "stage": result.get("stage") or scenario.get("stage"),
        "cluster": _infer_cluster(scenario_root, scenario),
        "method": result.get("method") or scenario.get("method"),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "timeout_minutes": result.get("timeout_minutes"),
        "scenario_file": str(scenario_path) if scenario_path.exists() else None,
        "result_file": str(result_path),
        "run_log_path": result.get("log_path") or str(scenario_root / "run.log"),
        "artifacts_dir": str(artifact_dir),
        "run_id": run_id,
        "launch_id": run_id,
        "run_name": run_name,
        "run_description": run_description,
        "run_goal": run_goal,
        "attribution_batch_size": run_config.get(
            "attribution_batch_size", scenario.get("attribution_batch_size")
        ),
        "feature_batch_size": run_config.get(
            "feature_batch_size", scenario.get("feature_batch_size")
        ),
        "logit_batch_size": run_config.get(
            "logit_batch_size", scenario.get("logit_batch_size")
        ),
        "decoder_chunk_size": run_config.get(
            "decoder_chunk_size", scenario.get("decoder_chunk_size")
        ),
        "chunked_feature_replay_window": run_config.get(
            "chunked_feature_replay_window",
            scenario.get("chunked_feature_replay_window"),
        ),
        "error_vector_prefetch_lookahead": run_config.get(
            "error_vector_prefetch_lookahead",
            scenario.get("error_vector_prefetch_lookahead"),
        ),
        "stage_encoder_vecs_on_cpu": run_config.get(
            "stage_encoder_vecs_on_cpu", scenario.get("stage_encoder_vecs_on_cpu")
        ),
        "stage_error_vectors_on_cpu": run_config.get(
            "stage_error_vectors_on_cpu",
            scenario.get("stage_error_vectors_on_cpu"),
        ),
        "row_subchunk_size": run_config.get(
            "row_subchunk_size", scenario.get("row_subchunk_size")
        ),
        "auto_scale_feature_batch_size": run_config.get(
            "auto_scale_feature_batch_size",
            scenario.get("auto_scale_feature_batch_size"),
        ),
        "feature_batch_size_max": run_config.get(
            "feature_batch_size_max", scenario.get("feature_batch_size_max")
        ),
        "decoder_cache_bytes": cache_bytes,
        "decoder_cache_gib": None if cache_bytes is None else cache_bytes / (1024**3),
        "max_feature_nodes": scenario.get("max_feature_nodes"),
        "max_edges": scenario.get("max_edges"),
        "max_steps": scenario.get("max_steps"),
        **_summarize_artifacts(artifact_dir),
        **profiling,
    }


def extract_benchmark_index(input_root: Path) -> list[dict[str, Any]]:
    return [
        build_benchmark_index_row(path)
        for path in sorted(input_root.glob("**/result.json"))
    ]


def _guess_failure_stage(
    summary: dict[str, Any], result_status: str | None
) -> str | None:
    if result_status == "success":
        return None
    if summary.get("cuda_oom_requested_gib") is not None:
        if summary.get("phase0_encode_total_active_features") is None:
            return "phase0_encode"
        if summary.get("phase0_reconstruction_seconds") is None:
            return "phase0_reconstruction"
        if summary.get("phase3_logit_attribution_seconds") is None:
            return "phase3"
        if summary.get("phase4_feature_attribution_seconds") is None:
            return "phase4"
    if (
        summary.get("phase4_batches_observed", 0) > 0
        and summary.get("phase4_feature_attribution_seconds") is None
    ):
        return "phase4"
    if (
        summary.get("phase3_logit_attribution_seconds") is None
        and summary.get("phase0_reconstruction_seconds") is not None
    ):
        return "phase3"
    if (
        summary.get("phase0_reconstruction_seconds") is None
        and summary.get("phase0_encode_total_active_features") is not None
    ):
        return "phase0_reconstruction"
    if summary.get("phase0_encode_total_active_features") is None:
        return "phase0_encode"
    return "unknown"


def parse_run_log_summary(
    *,
    scenario_root: Path,
    scenario_name: str,
    stage: str | None,
    cluster: str | None,
    result_status: str | None,
    log_path: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scenario_root": str(scenario_root),
        "scenario_name": scenario_name,
        "stage": stage,
        "cluster": cluster,
        "result_status": result_status,
        "log_path": str(log_path),
        "phase0_encode_seconds": None,
        "phase0_encode_total_active_features": None,
        "phase0_reconstruction_seconds": None,
        "phase0_reconstruction_total_chunks": None,
        "phase3_logit_count": None,
        "phase3_logit_attribution_seconds": None,
        "phase4_feature_attribution_seconds": None,
        "attribution_total_seconds": None,
        "phase4_batches_observed": 0,
        "phase4_batch_seconds_mean": None,
        "phase4_batch_seconds_max": None,
        "log_peak_rss_gib": None,
        "log_peak_cuda_allocated_gib": None,
        "log_peak_cuda_reserved_gib": None,
        "log_peak_cuda_peak_allocated_gib": None,
        "log_peak_cuda_peak_reserved_gib": None,
        "log_max_decoder_cache_hit_count": None,
        "log_max_decoder_cache_miss_count": None,
        "log_max_decoder_cache_eviction_count": None,
        "log_max_decoder_cache_resident_bytes": None,
        "cuda_oom_requested_gib": None,
        "cuda_oom_total_gib": None,
        "cuda_oom_free_gib": None,
        "cuda_oom_in_use_gib": None,
    }

    phase4_batch_seconds: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            match = PHASE0_ENCODE_DONE_RE.search(stripped)
            if match:
                summary["phase0_encode_total_active_features"] = int(
                    match.group("active_features")
                )
                summary["phase0_encode_seconds"] = float(match.group("seconds"))

            match = PHASE0_RECON_DONE_RE.search(stripped)
            if match:
                summary["phase0_reconstruction_total_chunks"] = int(
                    match.group("chunks")
                )
                summary["phase0_reconstruction_seconds"] = float(match.group("seconds"))

            match = PHASE3_DONE_RE.search(stripped)
            if match:
                summary["phase3_logit_count"] = int(match.group("count"))
                summary["phase3_logit_attribution_seconds"] = float(
                    match.group("seconds")
                )

            match = PHASE4_DONE_RE.search(stripped)
            if match:
                summary["phase4_feature_attribution_seconds"] = float(
                    match.group("seconds")
                )

            match = ATTRIBUTION_DONE_RE.search(stripped)
            if match:
                summary["attribution_total_seconds"] = float(match.group("seconds"))

            match = PHASE4_BATCH_RE.search(stripped)
            if match:
                phase4_batch_seconds.append(float(match.group("seconds")))

            match = CACHE_EVENT_RE.search(stripped)
            if match:
                event = match.group("event")
                count = int(match.group("count"))
                if event == "hit":
                    summary["log_max_decoder_cache_hit_count"] = _max_or_none(
                        summary["log_max_decoder_cache_hit_count"],
                        count,
                    )
                elif event == "miss":
                    summary["log_max_decoder_cache_miss_count"] = _max_or_none(
                        summary["log_max_decoder_cache_miss_count"],
                        count,
                    )
                elif event == "eviction":
                    summary["log_max_decoder_cache_eviction_count"] = _max_or_none(
                        summary["log_max_decoder_cache_eviction_count"],
                        count,
                    )

                resident_bytes = match.group("resident_bytes")
                if resident_bytes is not None:
                    summary["log_max_decoder_cache_resident_bytes"] = _max_or_none(
                        summary["log_max_decoder_cache_resident_bytes"],
                        int(resident_bytes),
                    )

            match = MEMORY_RE.search(stripped)
            if match:
                summary["log_peak_rss_gib"] = _max_or_none(
                    summary["log_peak_rss_gib"],
                    parse_memory_value_to_gib(match.group("rss")),
                )
                summary["log_peak_cuda_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_allocated_gib"],
                    parse_memory_value_to_gib(match.group("cuda_alloc")),
                )
                summary["log_peak_cuda_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_reserved_gib"],
                    parse_memory_value_to_gib(match.group("cuda_reserved")),
                )
                summary["log_peak_cuda_peak_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_allocated_gib"],
                    parse_memory_value_to_gib(match.group("cuda_peak_alloc")),
                )
                summary["log_peak_cuda_peak_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_reserved_gib"],
                    parse_memory_value_to_gib(match.group("cuda_peak_reserved")),
                )

            match = CUDA_OOM_RE.search(stripped)
            if match:
                summary["cuda_oom_requested_gib"] = parse_memory_value_to_gib(
                    match.group("requested")
                )
                summary["cuda_oom_total_gib"] = parse_memory_value_to_gib(
                    match.group("total")
                )
                summary["cuda_oom_free_gib"] = parse_memory_value_to_gib(
                    match.group("free")
                )
                summary["cuda_oom_in_use_gib"] = parse_memory_value_to_gib(
                    match.group("in_use")
                )

    if phase4_batch_seconds:
        summary["phase4_batches_observed"] = len(phase4_batch_seconds)
        summary["phase4_batch_seconds_mean"] = mean(phase4_batch_seconds)
        summary["phase4_batch_seconds_max"] = max(phase4_batch_seconds)

    summary["failure_stage_guess"] = _guess_failure_stage(summary, result_status)
    return summary


def extract_runlog_summaries(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(input_root.glob("**/result.json")):
        scenario_root = result_path.parent
        result = read_json(result_path)
        scenario_path = scenario_root / "scenario.json"
        scenario = read_json(scenario_path) if scenario_path.exists() else {}
        log_path = Path(result.get("log_path") or scenario_root / "run.log")
        if not log_path.exists():
            continue
        rows.append(
            parse_run_log_summary(
                scenario_root=scenario_root,
                scenario_name=result.get("name")
                or scenario.get("name")
                or scenario_root.name,
                stage=result.get("stage") or scenario.get("stage"),
                cluster=_infer_cluster(scenario_root, scenario),
                result_status=result.get("status"),
                log_path=log_path,
            )
        )
    return rows


def _classify_err_text(text: str) -> tuple[str | None, int | None]:
    oom_kill_match = OOM_KILL_RE.search(text)
    if oom_kill_match:
        return "ram_oom", int(oom_kill_match.group("count"))
    if "Exceeded step memory limit" in text:
        return "ram_oom", None
    if "OOM Killed" in text or "oom_kill" in text:
        return "ram_oom", None
    if "torch.OutOfMemoryError: CUDA out of memory" in text:
        return "cuda_oom", None
    if TIMEOUT_RE.search(text):
        return "timeout", None
    if text.strip():
        return "other_error", None
    return None, None


def _parse_out_metadata(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {}
    metadata: dict[str, Any] = {"out_file": str(out_path)}
    line_map = [
        (JOB_ID_RE, "job_id"),
        (ARRAY_TASK_RE, "array_task"),
        (NODE_RE, "node"),
        (CLUSTER_RE, "cluster"),
        (SCENARIOS_FILE_RE, "scenarios_file"),
        (OUTPUT_ROOT_RE, "output_root"),
        (RUN_ID_RE, "run_id"),
        (RUN_NAME_RE, "run_name"),
        (RUN_DESCRIPTION_RE, "run_description"),
        (RUN_GOAL_RE, "run_goal"),
        (WRITING_DIR_RE, "scenario_root"),
        (RUNNING_SCENARIO_RE, "scenario_name"),
    ]
    for raw_line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for regex, key in line_map:
            match = regex.search(line)
            if match:
                metadata[key] = _normalize_optional_metadata_value(match.group(key))
    return metadata


def build_slurm_row(err_path: Path, benchmark_root: Path) -> dict[str, Any]:
    err_text = err_path.read_text(encoding="utf-8", errors="replace")
    failure_family, oom_kill_count = _classify_err_text(err_text)
    out_path = err_path.with_suffix(".out")
    out_metadata = _parse_out_metadata(out_path)
    scenario_root = out_metadata.get("scenario_root")
    result_path = Path(scenario_root) / "result.json" if scenario_root else None
    result_status = None
    if result_path is not None and result_path.exists():
        result_status = read_json(result_path).get("status")

    return {
        "err_file": str(err_path),
        "out_file": out_metadata.get("out_file"),
        "slurm_stem": safe_stem(err_path),
        "job_id": out_metadata.get("job_id"),
        "array_task": out_metadata.get("array_task"),
        "node": out_metadata.get("node"),
        "cluster": out_metadata.get("cluster"),
        "scenarios_file": out_metadata.get("scenarios_file"),
        "output_root": out_metadata.get("output_root"),
        "run_id": out_metadata.get("run_id"),
        "launch_id": out_metadata.get("run_id"),
        "run_name": out_metadata.get("run_name"),
        "run_description": out_metadata.get("run_description"),
        "run_goal": out_metadata.get("run_goal"),
        "scenario_root": scenario_root,
        "scenario_name": out_metadata.get("scenario_name"),
        "failure_family": failure_family,
        "oom_kill_event_count": oom_kill_count,
        "result_json_exists": result_path.exists()
        if result_path is not None
        else False,
        "result_status": result_status,
        "matches_benchmark_root": bool(
            scenario_root and str(scenario_root).startswith(str(benchmark_root))
        ),
        "err_excerpt": err_text.strip()[:500] if err_text.strip() else None,
    }


def extract_slurm_err_summary(
    logs_dir: Path, benchmark_root: Path
) -> list[dict[str, Any]]:
    return [
        build_slurm_row(path, benchmark_root) for path in sorted(logs_dir.glob("*.err"))
    ]


def _to_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in {None, "", "None"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def build_slurm_lookup(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_root = row.get("scenario_root") or ""
        if scenario_root and row.get("matches_benchmark_root"):
            by_root[scenario_root].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        families = sorted(
            {row.get("failure_family") for row in group if row.get("failure_family")}
        )
        return {
            "slurm_err_file_count": len(group),
            "slurm_failure_families": "|".join(families) if families else None,
            "slurm_any_ram_oom": any(
                row.get("failure_family") == "ram_oom" for row in group
            ),
            "slurm_any_cuda_oom": any(
                row.get("failure_family") == "cuda_oom" for row in group
            ),
            "slurm_any_timeout": any(
                row.get("failure_family") == "timeout" for row in group
            ),
            "slurm_oom_kill_event_count": sum(
                _to_int(row.get("oom_kill_event_count")) or 0 for row in group
            ),
            "slurm_err_excerpt": next(
                (row.get("err_excerpt") for row in group if row.get("err_excerpt")),
                None,
            ),
        }

    return {key: summarize(group) for key, group in by_root.items()}


def merge_benchmark_tables(
    benchmark_rows: list[dict[str, Any]],
    runlog_rows: list[dict[str, Any]],
    slurm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runlog_lookup = {
        row["scenario_root"]: row for row in runlog_rows if row.get("scenario_root")
    }
    slurm_by_root = build_slurm_lookup(slurm_rows)

    merged_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        merged = dict(row)
        scenario_root = row.get("scenario_root") or ""
        runlog = runlog_lookup.get(scenario_root, {})
        slurm = slurm_by_root.get(scenario_root, {})

        for key, value in runlog.items():
            if key in {
                "scenario_root",
                "scenario_name",
                "stage",
                "cluster",
                "result_status",
            }:
                continue
            merged[key] = value
        merged.update(slurm)

        max_active_features = _to_float(merged.get("max_active_features"))
        duration_seconds = _to_float(merged.get("duration_seconds"))
        merged["runtime_per_million_active_features"] = (
            None
            if duration_seconds is None or max_active_features in {None, 0.0}
            else duration_seconds / (max_active_features / 1_000_000.0)
        )

        status = merged.get("status")
        slurm_any_ram_oom = _to_bool(merged.get("slurm_any_ram_oom"))
        slurm_any_timeout = _to_bool(merged.get("slurm_any_timeout"))
        cuda_oom_seen = _to_float(merged.get("cuda_oom_requested_gib")) is not None
        if status == "success":
            failure_family = "success"
        elif slurm_any_ram_oom:
            failure_family = "ram_oom"
        elif status == "oom" or cuda_oom_seen:
            failure_family = "cuda_oom"
        elif status == "timeout" or slurm_any_timeout:
            failure_family = "timeout"
        else:
            failure_family = "other_fail"
        merged["failure_family_final"] = failure_family
        merged_rows.append(merged)

    return merged_rows


def run_full_extraction(
    *,
    input_root: Path = DEFAULT_SCRATCH_ROOT,
    output_dir: Path = DEFAULT_EXTRACTED_DIR,
    logs_dir: Path | None = DEFAULT_LOGS_DIR,
) -> dict[str, int]:
    ensure_dir(output_dir)

    benchmark_rows = extract_benchmark_index(input_root)
    runlog_rows = extract_runlog_summaries(input_root)
    slurm_rows = extract_slurm_err_summary(logs_dir, input_root) if logs_dir else []
    merged_rows = merge_benchmark_tables(benchmark_rows, runlog_rows, slurm_rows)

    write_csv(output_dir / "benchmark_index.csv", benchmark_rows)
    write_jsonl(output_dir / "benchmark_index.jsonl", benchmark_rows)
    write_csv(output_dir / "runlog_summary.csv", runlog_rows)
    if logs_dir is not None:
        write_csv(output_dir / "slurm_err_summary.csv", slurm_rows)
    write_csv(output_dir / "benchmark_enriched.csv", merged_rows)
    write_jsonl(output_dir / "benchmark_enriched.jsonl", merged_rows)

    return {
        "benchmark_rows": len(benchmark_rows),
        "runlog_rows": len(runlog_rows),
        "slurm_rows": len(slurm_rows),
        "merged_rows": len(merged_rows),
    }
