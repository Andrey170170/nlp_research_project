from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from nlp_research_project.exact_trace_bench.full_answer.audit import audit_prefix_views
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    prefix_view_metadata,
    reconstruct_prefix_token_ids,
)
from nlp_research_project.exact_trace_bench.full_answer.schemas import TraceSpec


def _trajectory() -> dict:
    return {
        "schema_version": 1,
        "trajectory_id": "traj_audit",
        "prompt_token_count": 2,
        "prompt_token_ids": [101, 102],
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 2,
                "token_id": 201,
                "token_text": "A",
                "is_stop": False,
            },
            {
                "generated_index": 1,
                "absolute_token_position": 3,
                "token_id": 202,
                "token_text": "B",
                "is_stop": False,
            },
        ],
    }


def _trace(trajectory: dict, *, generated_index: int, graph_path: Path | None) -> dict:
    token = trajectory["generated_tokens"][generated_index]
    spec = cast(
        TraceSpec,
        {
            "schema_version": 1,
            "trace_id": f"traj_audit_tok{generated_index:06d}",
            "trajectory_id": trajectory["trajectory_id"],
            "generated_index": generated_index,
            "target_position": trajectory["prompt_token_count"] + generated_index,
            "prefix_token_count": trajectory["prompt_token_count"] + generated_index,
            "target_token_id": token["token_id"],
            "target_token_text": token["token_text"],
            "target_mode": "frozen_target_only",
            "selection_reasons": ["unit"],
            "graph_knobs": {},
            "estimated_cost": trajectory["prompt_token_count"] + generated_index,
        },
    )
    prefix = reconstruct_prefix_token_ids(trajectory, spec)
    return {
        **spec,
        "shard_id": 0,
        "status": "ok",
        "graph_path": str(graph_path) if graph_path is not None else None,
        "forced_target": {
            "token_id": token["token_id"],
            "token_text": token["token_text"],
        },
        "prefix_view_metadata": prefix_view_metadata(trajectory, spec, prefix),
        "error": None,
    }


def _write_trace(run_root: Path, trace: dict) -> Path:
    path = (
        run_root
        / "shards"
        / "shard_000"
        / f"token_{trace['generated_index']:06d}"
        / "trace.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def _write_graph(path: Path, *, step_idx: int, feature_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_features = int(feature_ids.shape[0])
    np.savez(
        path,
        row_idx=np.asarray([], dtype=np.int32),
        col_idx=np.asarray([], dtype=np.int32),
        weights=np.asarray([], dtype=np.float32),
        feature_ids=feature_ids,
        token_text=np.asarray("token"),
        logprob=np.asarray(np.nan),
        n_features=np.asarray(n_features, dtype=np.int32),
        step_idx=np.asarray(step_idx, dtype=np.int32),
    )


def test_audit_prefix_views_accepts_matching_artifacts(tmp_path: Path) -> None:
    trajectory = _trajectory()
    trajectory_path = tmp_path / "trajectory.json"
    run_root = tmp_path / "run"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    graph_path = run_root / "shards" / "shard_000" / "token_000001" / "graph.npz"
    _write_graph(
        graph_path,
        step_idx=1,
        feature_ids=np.asarray([[0, 0, 10], [1, 1, 11], [2, 2, 12]], dtype=np.int64),
    )
    _write_trace(run_root, _trace(trajectory, generated_index=1, graph_path=graph_path))

    summary = audit_prefix_views(trajectory_path=trajectory_path, run_root=run_root)

    assert summary["counts"]["total"] == 1
    assert summary["counts"]["ok"] == 1
    row = json.loads((run_root / "prefix_view_audit.jsonl").read_text().splitlines()[0])
    assert row["audit_status"] == "ok"
    assert row["graph_max_feature_position"] == 2


def test_audit_prefix_views_flags_mismatches_and_future_positions(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory()
    trajectory_path = tmp_path / "trajectory.json"
    run_root = tmp_path / "run"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    graph_path = run_root / "shards" / "shard_000" / "token_000000" / "graph.npz"
    _write_graph(
        graph_path,
        step_idx=0,
        feature_ids=np.asarray([[0, 2, 10]], dtype=np.int64),
    )
    trace = _trace(trajectory, generated_index=0, graph_path=graph_path)
    trace["target_token_id"] = 999
    trace["prefix_view_metadata"]["prefix_token_ids_sha256"] = "bad"
    _write_trace(run_root, trace)

    summary = audit_prefix_views(trajectory_path=trajectory_path, run_root=run_root)

    assert summary["counts"]["ok"] == 0
    assert summary["counts"]["metadata_mismatches"] == 1
    assert summary["counts"]["target_mismatches"] == 1
    assert summary["counts"]["future_position_violations"] == 1


def test_audit_prefix_views_rejects_structurally_invalid_graph(tmp_path: Path) -> None:
    trajectory = _trajectory()
    trajectory_path = tmp_path / "trajectory.json"
    run_root = tmp_path / "run"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    graph_path = run_root / "shards" / "shard_000" / "token_000000" / "graph.npz"
    _write_graph(
        graph_path,
        step_idx=0,
        feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
    )
    with np.load(graph_path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload["row_idx"] = np.asarray([0], dtype=np.int32)
    np.savez(graph_path, **payload)
    _write_trace(run_root, _trace(trajectory, generated_index=0, graph_path=graph_path))

    summary = audit_prefix_views(trajectory_path=trajectory_path, run_root=run_root)

    assert summary["counts"]["ok"] == 0
    assert summary["counts"]["invalid_graph"] == 1
    row = json.loads((run_root / "prefix_view_audit.jsonl").read_text().splitlines()[0])
    assert row["audit_status"] == "error"
    assert "mismatched lengths" in row["graph_error"]
