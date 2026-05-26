from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io_utils import iter_jsonl, write_csv, write_json, write_jsonl


PREFERRED_HEADERS = [
    "shard_id",
    "trace_id",
    "trajectory_id",
    "generated_index",
    "status",
    "graph_path",
    "target_token_id",
    "target_token_text",
    "target_mode",
    "prefix_token_count",
    "estimated_cost",
    "error",
]


def aggregate_shards(run_root: Path) -> dict[str, Any]:
    rows = _load_rows(run_root)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    aggregate = {
        "schema_version": 1,
        "run_root": str(run_root),
        "token_count": len(rows),
        "status_counts": status_counts,
        "shard_count": len({row.get("shard_id") for row in rows}),
    }
    write_json(run_root / "aggregate.json", aggregate)
    write_jsonl(run_root / "per_token_metrics.jsonl", rows)
    write_csv(
        run_root / "per_token_metrics.csv", rows, preferred_headers=PREFERRED_HEADERS
    )
    return aggregate


def _load_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shards_root = run_root / "shards"
    if not shards_root.exists():
        return rows
    for results_path in sorted(shards_root.glob("shard_*/trace_results.jsonl")):
        for row in iter_jsonl(results_path):
            rows.append(_metric_row(row))
    rows.sort(
        key=lambda row: (
            int(row.get("shard_id", -1)),
            int(row.get("generated_index", -1)),
        )
    )
    return rows


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    forced_raw = row.get("forced_target")
    forced: dict[str, Any] = forced_raw if isinstance(forced_raw, dict) else {}
    return {
        "schema_version": 1,
        "shard_id": row.get("shard_id"),
        "trace_id": row.get("trace_id"),
        "trajectory_id": row.get("trajectory_id"),
        "generated_index": row.get("generated_index"),
        "status": row.get("status"),
        "graph_path": row.get("graph_path"),
        "target_token_id": row.get("target_token_id", forced.get("token_id")),
        "target_token_text": row.get("target_token_text", forced.get("token_text")),
        "target_mode": row.get("target_mode", forced.get("target_mode")),
        "prefix_token_count": row.get("prefix_token_count"),
        "estimated_cost": row.get("estimated_cost"),
        "selection_reasons": row.get("selection_reasons", []),
        "error": row.get("error"),
    }
