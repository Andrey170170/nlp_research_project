from __future__ import annotations

from pathlib import Path
import json
import os
import time
from typing import Any, Mapping, cast

from ..io_utils import ensure_dir, write_json, write_jsonl
from .schemas import TraceSpec, load_shards, load_trace_specs, load_trajectory


def _shard_dir(output_root: Path, shard_id: int) -> Path:
    return output_root / "shards" / f"shard_{shard_id:03d}"


def select_shard(shards: Mapping[str, Any], shard_id: int) -> dict[str, Any]:
    shard_rows = shards.get("shards")
    if not isinstance(shard_rows, list):
        raise ValueError("shards payload missing shards list")
    for shard in shard_rows:
        if isinstance(shard, dict) and shard.get("shard_id") == shard_id:
            return cast(dict[str, Any], shard)
    raise ValueError(f"shard_id not found: {shard_id}")


def assigned_specs(specs: list[TraceSpec], shard: Mapping[str, Any]) -> list[TraceSpec]:
    selected: list[TraceSpec] = []
    for index in shard.get("spec_indices", []):
        if not isinstance(index, int) or index < 0 or index >= len(specs):
            raise ValueError(f"spec index out of bounds for shard: {index}")
        selected.append(specs[index])
    return selected


def load_shard_inputs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> tuple[dict[str, Any], list[TraceSpec], dict[str, Any]]:
    trajectory = load_trajectory(trajectory_path)
    specs = load_trace_specs(trace_specs_path)
    shards = load_shards(shards_path)
    shard = select_shard(shards, shard_id)
    selected = assigned_specs(specs, shard)
    trajectory_id = trajectory["trajectory_id"]
    for spec in selected:
        if spec["trajectory_id"] != trajectory_id:
            raise ValueError("trace spec trajectory_id does not match trajectory")
    return cast(dict[str, Any], trajectory), selected, shard


def list_shard_specs(
    *, trajectory_path: Path, trace_specs_path: Path, shards_path: Path, shard_id: int
) -> list[dict[str, Any]]:
    _, specs, _ = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    return [_summary_row(spec, shard_id=shard_id) for spec in specs]


def print_shard_specs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"shard={row['shard_id']} generated_index={row['generated_index']} "
            f"trace_id={row['trace_id']} target={row['target_token_id']} "
            f"text={row['target_token_text']!r} cost={row['estimated_cost']}"
        )


def dry_run_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
) -> dict[str, Any]:
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    shard_payload = {
        "schema_version": 1,
        "status": "dry_run",
        "shard": shard,
        "trajectory_id": trajectory["trajectory_id"],
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": len(specs),
    }
    write_json(root / "shard.json", shard_payload)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        trace = _trace_payload(spec, shard_id=shard_id)
        token_dir = root / f"token_{spec['generated_index']:06d}"
        write_json(token_dir / "trace.json", trace)
        rows.append(trace)
    write_jsonl(root / "trace_results.jsonl", rows)
    return {"shard_dir": str(root), "token_count": len(rows), "status": "dry_run"}


def reconstruct_prefix_token_ids(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> list[int]:
    """Return frozen prompt + generated prefix immediately before target token."""
    generated_index = spec["generated_index"]
    prompt_token_ids = trajectory.get("prompt_token_ids")
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_token_ids
    ):
        raise ValueError("trajectory prompt_token_ids must be a list of ints")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    if generated_index < 0 or generated_index >= len(generated_tokens):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    prefix_generated = [
        int(token["token_id"])
        for token in generated_tokens[:generated_index]
        if isinstance(token, Mapping) and isinstance(token.get("token_id"), int)
    ]
    if len(prefix_generated) != generated_index:
        raise ValueError("generated prefix contains malformed token rows")
    prefix = [int(token_id) for token_id in prompt_token_ids] + prefix_generated
    if spec["prefix_token_count"] != len(prefix):
        raise ValueError(
            "trace spec prefix_token_count does not match reconstructed prefix "
            f"({spec['prefix_token_count']} != {len(prefix)})"
        )
    return prefix


def forced_target_payload(spec: TraceSpec) -> dict[str, Any]:
    return {
        "token_id": spec["target_token_id"],
        "token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "attribution_targets": [spec["target_token_id"]],
    }


def _runtime_metadata(
    *,
    run_id: str | None,
    run_name: str | None,
    run_description: str | None,
    run_goal: str | None,
) -> dict[str, Any]:
    workspace_root = os.environ.get("WORKSPACE_ROOT")
    library_workspace_root = os.environ.get("LIB_WORKSPACE_ROOT")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return {
        "run_id": run_id or os.environ.get("RUN_ID") or None,
        "run_name": run_name or os.environ.get("RUN_NAME") or None,
        "run_description": run_description or os.environ.get("RUN_DESCRIPTION") or None,
        "run_goal": run_goal or os.environ.get("RUN_GOAL") or None,
        "workspace_root": workspace_root,
        "library_workspace_root": library_workspace_root,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
    }


def _shard_record(
    *,
    status: str,
    shard: Mapping[str, Any],
    trajectory_id: str,
    trace_specs_path: Path,
    shards_path: Path,
    token_count: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "shard": dict(shard),
        "trajectory_id": trajectory_id,
        "trace_specs_file": str(trace_specs_path),
        "shards_file": str(shards_path),
        "token_count": token_count,
    }
    payload.update(metadata)
    return payload


def _trace_runtime_payload(
    *,
    spec: TraceSpec,
    shard_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _trace_payload(spec, shard_id=shard_id)
    payload.update(metadata)
    return payload


def _target_token_metadata(
    trajectory: Mapping[str, Any], spec: TraceSpec
) -> dict[str, Any]:
    generated_tokens = trajectory.get("generated_tokens")
    if not isinstance(generated_tokens, list):
        raise ValueError("trajectory generated_tokens must be a list")
    generated_index = spec["generated_index"]
    token = generated_tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    return {
        "target_logprob": token.get("logprob"),
        "target_probability": token.get("probability"),
        "target_rank": token.get("rank"),
    }


def _graph_summary(
    step: Mapping[str, Any] | Any, graph_path: Path | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {"saved": graph_path is not None}
    if graph_path is not None:
        summary["graph_path"] = str(graph_path)
    if isinstance(step, Mapping):
        for key in (
            "node_count",
            "edge_count",
            "feature_node_count",
            "max_edges",
            "max_feature_nodes",
            "active_feature_count",
        ):
            if key in step:
                summary[key] = step[key]
        nested = step.get("summary")
        if isinstance(nested, Mapping):
            for key in ("node_count", "edge_count", "feature_node_count"):
                if key in nested and key not in summary:
                    summary[key] = nested[key]
    return summary


def _model_load_knobs(specs: list[TraceSpec]) -> dict[str, Any]:
    decoder_chunk_size: int | None = None
    cross_batch_decoder_cache_bytes: int | None = None
    for spec in specs:
        knobs = spec["graph_knobs"]
        if not isinstance(knobs, Mapping):
            raise ValueError("trace spec graph_knobs must be an object")
        spec_decoder_chunk_size = knobs.get("decoder_chunk_size")
        spec_cache_bytes = knobs.get("cross_batch_decoder_cache_bytes")
        if spec_decoder_chunk_size is not None:
            if (
                not isinstance(spec_decoder_chunk_size, int)
                or spec_decoder_chunk_size <= 0
            ):
                raise ValueError("decoder_chunk_size must be a positive int")
            if (
                decoder_chunk_size is not None
                and spec_decoder_chunk_size != decoder_chunk_size
            ):
                raise ValueError("all specs in a shard must share decoder_chunk_size")
            decoder_chunk_size = spec_decoder_chunk_size
        if spec_cache_bytes is not None:
            if not isinstance(spec_cache_bytes, int) or spec_cache_bytes < 0:
                raise ValueError(
                    "cross_batch_decoder_cache_bytes must be a non-negative int"
                )
            if (
                cross_batch_decoder_cache_bytes is not None
                and spec_cache_bytes != cross_batch_decoder_cache_bytes
            ):
                raise ValueError(
                    "all specs in a shard must share cross_batch_decoder_cache_bytes"
                )
            cross_batch_decoder_cache_bytes = spec_cache_bytes
    return {
        "decoder_chunk_size": decoder_chunk_size or 256,
        "cross_batch_decoder_cache_bytes": cross_batch_decoder_cache_bytes,
    }


def run_real_shard(
    *,
    trajectory_path: Path,
    trace_specs_path: Path,
    shards_path: Path,
    shard_id: int,
    output_root: Path,
    run_id: str | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
    run_goal: str | None = None,
) -> dict[str, Any]:
    """SLURM-only real tracing path; imports model stack only inside this function."""
    trajectory, specs, shard = load_shard_inputs(
        trajectory_path=trajectory_path,
        trace_specs_path=trace_specs_path,
        shards_path=shards_path,
        shard_id=shard_id,
    )
    metadata = _runtime_metadata(
        run_id=run_id,
        run_name=run_name,
        run_description=run_description,
        run_goal=run_goal,
    )
    if (
        os.environ.get("SLURM_JOB_ID") is None
        and os.environ.get("EXACT_TRACE_ALLOW_LOCAL_GPU") != "1"
    ):
        raise RuntimeError(
            "run-full-answer-shard requires SLURM_JOB_ID; set EXACT_TRACE_ALLOW_LOCAL_GPU=1 "
            "to override for local GPU debugging before importing model code"
        )
    root = _shard_dir(output_root, shard_id)
    ensure_dir(root)
    trace_results_path = root / "trace_results.jsonl"
    trace_results_path.write_text("", encoding="utf-8")
    write_json(
        root / "shard.json",
        _shard_record(
            status="running",
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata=metadata,
        ),
    )

    import importlib

    import torch

    import circuit_utils
    import trace_pipeline as base
    from trace_pipeline_chunked import (
        compact_result_to_step_data,
        resolve_internal_precision,
    )

    attribute_module = importlib.import_module(
        "circuit_tracer.attribution.attribute_nnsight"
    )
    attribute_nnsight = getattr(attribute_module, "attribute")
    model_load_knobs = _model_load_knobs(specs)
    model = base.load_model(
        exact_chunked_decoder=True,
        decoder_chunk_size=model_load_knobs["decoder_chunk_size"],
        cross_batch_decoder_cache_bytes=model_load_knobs[
            "cross_batch_decoder_cache_bytes"
        ],
    )
    offload = "cpu"
    rows: list[dict[str, Any]] = []
    for spec in specs:
        token_dir = root / f"token_{spec['generated_index']:06d}"
        graph_path = token_dir / "graph.npz"
        trace = _trace_runtime_payload(spec=spec, shard_id=shard_id, metadata=metadata)
        trace["status"] = "running"
        trace["forced_target"] = forced_target_payload(spec)
        trace["graph_path"] = str(graph_path)
        trace.update(_target_token_metadata(trajectory, spec))
        started = time.perf_counter()
        try:
            prefix_token_ids = reconstruct_prefix_token_ids(trajectory, spec)
            knobs = spec["graph_knobs"]
            compact_result = attribute_nnsight(
                prompt=torch.tensor(prefix_token_ids, dtype=torch.long),
                model=model,
                attribution_targets=torch.tensor(
                    [spec["target_token_id"]], dtype=torch.long
                ),
                max_n_logits=1,
                desired_logit_prob=1.0,
                batch_size=int(knobs.get("attribution_batch_size", 256)),
                feature_batch_size=knobs.get("feature_batch_size"),
                logit_batch_size=knobs.get("logit_batch_size"),
                max_feature_nodes=int(knobs.get("max_feature_nodes", 8192)),
                offload=offload,
                verbose=bool(knobs.get("verbose_attribution", True)),
                update_interval=int(knobs.get("attribution_update_interval", 4)),
                profile=bool(knobs.get("profile_attribution", True)),
                profile_log_interval=int(knobs.get("profile_log_interval", 1)),
                internal_precision=resolve_internal_precision(
                    str(knobs.get("exact_trace_internal_dtype", "fp32"))
                ),
                exact_trace_internal_dtype=str(
                    knobs.get("exact_trace_internal_dtype", "fp32")
                ),
                compact_output=True,
            )
            step = compact_result_to_step_data(
                compact_result,
                spec["generated_index"],
                token_text=spec["target_token_text"],
                max_edges=int(knobs.get("max_edges", 20000)),
            )
            circuit_utils.save_compact(step, graph_path)
            trace.update(
                {
                    "status": "ok",
                    "error": None,
                    "timings": {
                        "trace_seconds": time.perf_counter() - started,
                    },
                    "graph_summary": _graph_summary(step, graph_path),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised only in SLURM real mode
            trace.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "timings": {
                        "trace_seconds": time.perf_counter() - started,
                    },
                }
            )
        write_json(token_dir / "trace.json", trace)
        rows.append(trace)
        with trace_results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    final_status = "complete" if all(r["status"] == "ok" for r in rows) else "error"
    write_json(
        root / "shard.json",
        _shard_record(
            status=final_status,
            shard=shard,
            trajectory_id=trajectory["trajectory_id"],
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            token_count=len(specs),
            metadata=metadata,
        ),
    )
    return {"shard_dir": str(root), "token_count": len(rows), "status": final_status}


def _summary_row(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    return {
        "shard_id": shard_id,
        "trace_id": spec["trace_id"],
        "trajectory_id": spec["trajectory_id"],
        "generated_index": spec["generated_index"],
        "prefix_token_count": spec["prefix_token_count"],
        "target_token_id": spec["target_token_id"],
        "target_token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
        "estimated_cost": spec["estimated_cost"],
        "selection_reasons": spec["selection_reasons"],
    }


def _trace_payload(spec: TraceSpec, *, shard_id: int) -> dict[str, Any]:
    forced_target = forced_target_payload(spec)
    payload = _summary_row(spec, shard_id=shard_id)
    payload.update(
        {
            "schema_version": 1,
            "status": "dry_run",
            "graph_path": None,
            "graph_knobs": spec["graph_knobs"],
            "forced_target": forced_target,
            "error": None,
        }
    )
    return payload
