from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..io_utils import ensure_dir, read_json, write_json, write_jsonl
from ..typed_compact_graph import load_typed_compact_graph
from .runner import prefix_view_metadata, reconstruct_prefix_token_ids
from .schemas import TraceSpec, load_trajectory


AUDIT_SCHEMA_VERSION = 1


def audit_prefix_views(
    *, trajectory_path: Path, run_root: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Audit existing full-answer independent-prefix artifacts without models.

    This is the first Phase 2 artifact-level scaffold: it validates the frozen
    trajectory contract, stored prefix-view metadata, and compact graph positions
    for already-produced independent-prefix traces. It intentionally does not
    implement shared Phase-0 reuse, full-sequence cache reuse, or scheduling.
    """
    trajectory = load_trajectory(trajectory_path)
    rows = [
        _audit_trace(trajectory, trace_path, run_root)
        for trace_path in _trace_paths(run_root)
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("shard_id", -1)),
            int(row.get("generated_index", -1)),
        )
    )

    counts = {
        "total": len(rows),
        "ok": sum(1 for row in rows if row["audit_status"] == "ok"),
        "error_or_non_ok_traces": sum(1 for row in rows if row["trace_status"] != "ok"),
        "metadata_mismatches": sum(1 for row in rows if row["metadata_mismatch"]),
        "target_mismatches": sum(1 for row in rows if row["target_mismatch"]),
        "future_position_violations": sum(
            1 for row in rows if row["future_position_violation"]
        ),
        "missing_graph": sum(1 for row in rows if row["graph_missing"]),
        "invalid_graph": sum(1 for row in rows if row["graph_error"] is not None),
        "missing_metadata": sum(1 for row in rows if row["metadata_missing"]),
    }
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_kind": "full_answer_independent_prefix_artifact_audit",
        "run_root": str(run_root),
        "trajectory": str(trajectory_path),
        "trajectory_id": trajectory["trajectory_id"],
        "counts": counts,
    }
    out = output_dir or run_root
    ensure_dir(out)
    write_json(out / "prefix_view_audit.json", summary)
    write_jsonl(out / "prefix_view_audit.jsonl", rows)
    return summary


def _trace_paths(run_root: Path) -> list[Path]:
    return sorted((run_root / "shards").glob("shard_*/token_*/trace.json"))


def _audit_trace(
    trajectory: Mapping[str, Any], trace_path: Path, run_root: Path
) -> dict[str, Any]:
    trace = read_json(trace_path)
    if not isinstance(trace, dict):
        raise ValueError(f"trace artifact must be a JSON object: {trace_path}")
    row: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "trace_path": str(trace_path),
        "trace_id": trace.get("trace_id"),
        "shard_id": trace.get("shard_id"),
        "generated_index": trace.get("generated_index"),
        "target_position": trace.get("target_position"),
        "trace_status": trace.get("status"),
        "metadata_missing": not isinstance(trace.get("prefix_view_metadata"), dict),
        "metadata_mismatch": False,
        "target_mismatch": False,
        "future_position_violation": False,
        "graph_missing": False,
        "graph_error": None,
        "errors": [],
    }
    errors = row["errors"]
    try:
        spec = _spec_from_trace(trace)
        token = _trajectory_token(trajectory, spec["generated_index"])
        expected_position = (
            int(trajectory["prompt_token_count"]) + spec["generated_index"]
        )
        target_mismatches = []
        if spec["target_position"] != expected_position:
            target_mismatches.append("target_position")
        if token.get("absolute_token_position") != expected_position:
            target_mismatches.append("trajectory_absolute_token_position")
        if spec["target_token_id"] != token.get("token_id"):
            target_mismatches.append("target_token_id")
        if spec["target_token_text"] != token.get("token_text"):
            target_mismatches.append("target_token_text")
        row["target_mismatch_fields"] = target_mismatches
        row["target_mismatch"] = bool(target_mismatches)

        prefix = reconstruct_prefix_token_ids(trajectory, spec)
        expected_metadata = prefix_view_metadata(trajectory, spec, prefix)
        stored_metadata = trace.get("prefix_view_metadata")
        if isinstance(stored_metadata, dict):
            mismatched = [
                key
                for key, expected in expected_metadata.items()
                if stored_metadata.get(key) != expected
            ]
            row["metadata_mismatch_fields"] = mismatched
            row["metadata_mismatch"] = bool(mismatched)
            row["expected_prefix_token_ids_sha256"] = expected_metadata[
                "prefix_token_ids_sha256"
            ]
            row["stored_prefix_token_ids_sha256"] = stored_metadata.get(
                "prefix_token_ids_sha256"
            )
        else:
            errors.append("missing prefix_view_metadata")

        graph_path = _resolve_graph_path(trace.get("graph_path"), trace_path, run_root)
        row["graph_path"] = str(graph_path) if graph_path is not None else None
        if graph_path is None or not graph_path.exists():
            row["graph_missing"] = True
        else:
            graph_result = inspect_compact_graph_positions(
                graph_path,
                target_position=spec["target_position"],
                expected_step_idx=spec["generated_index"],
            )
            row.update(graph_result)
    except Exception as exc:
        errors.append(str(exc))
    row["audit_status"] = "ok" if _row_ok(row) else "error"
    return row


def inspect_compact_graph_positions(
    graph_path: Path, *, target_position: int, expected_step_idx: int | None = None
) -> dict[str, Any]:
    """Inspect compact graph feature positions for prefix-view leakage.

    Independent-prefix traces for generated token y_k pass only positions
    ``0..target_position-1`` into attribution. Feature nodes at
    ``target_position`` or later therefore indicate target/future-position
    leakage into a per-token graph.
    """
    try:
        graph = load_typed_compact_graph(
            graph_path, expected_step_idx=expected_step_idx
        )
        positions = graph.feature_ids[:, 1]
        future_count = int(np.count_nonzero(positions >= int(target_position)))
        typed_endpoint_count = sum(
            int(np.count_nonzero(graph.bucket_ids == bucket_id))
            * (int(name.startswith("feature<-")) + int(name.endswith("<-feature")))
            for bucket_id, name in enumerate(graph.bucket_names)
        )
        return {
            "graph_feature_count": graph.n_features,
            "graph_typed_feature_endpoint_count": typed_endpoint_count,
            "graph_max_feature_position": (
                int(positions.max()) if positions.size else None
            ),
            "future_position_violation": future_count > 0,
            "future_position_count": future_count,
            "retention_policy_id": graph.retention_policy_id,
            "retention_policy_fingerprint": graph.retention_policy_fingerprint,
            "graph_fingerprint": graph.graph_fingerprint,
        }
    except Exception as exc:  # pragma: no cover - defensive malformed npz path
        return {"graph_error": repr(exc), "future_position_violation": False}


def _row_ok(row: Mapping[str, Any]) -> bool:
    return (
        row.get("trace_status") == "ok"
        and not row.get("metadata_missing")
        and not row.get("metadata_mismatch")
        and not row.get("target_mismatch")
        and not row.get("graph_missing")
        and not row.get("future_position_violation")
        and not row.get("graph_error")
        and not row.get("errors")
    )


def _spec_from_trace(trace: Mapping[str, Any]) -> TraceSpec:
    return {
        "schema_version": 1,
        "trace_id": str(trace["trace_id"]),
        "trajectory_id": str(trace["trajectory_id"]),
        "generated_index": int(trace["generated_index"]),
        "target_position": int(trace["target_position"]),
        "prefix_token_count": int(trace["prefix_token_count"]),
        "target_token_id": int(trace["target_token_id"]),
        "target_token_text": str(trace["target_token_text"]),
        "target_mode": str(trace["target_mode"]),
        "selection_reasons": list(trace.get("selection_reasons", [])),
        "graph_knobs": dict(trace.get("graph_knobs", {})),
        "estimated_cost": int(trace.get("estimated_cost", 0)),
    }


def _trajectory_token(
    trajectory: Mapping[str, Any], generated_index: int
) -> Mapping[str, Any]:
    tokens = trajectory.get("generated_tokens")
    if (
        not isinstance(tokens, list)
        or generated_index < 0
        or generated_index >= len(tokens)
    ):
        raise ValueError(f"generated_index out of bounds: {generated_index}")
    token = tokens[generated_index]
    if not isinstance(token, Mapping):
        raise ValueError("trajectory generated token row must be an object")
    return token


def _resolve_graph_path(value: Any, trace_path: Path, run_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    for base in (trace_path.parent, run_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return run_root / path
