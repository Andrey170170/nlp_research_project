from __future__ import annotations

from pathlib import Path
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
    forced_target = {
        "token_id": spec["target_token_id"],
        "token_text": spec["target_token_text"],
        "target_mode": spec["target_mode"],
    }
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
