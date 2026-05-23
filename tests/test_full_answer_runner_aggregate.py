from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nlp_research_project.exact_trace_bench.full_answer.aggregate import (
    aggregate_shards,
)
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    dry_run_shard,
    list_shard_specs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _write_tiny_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    trajectory = {
        "schema_version": 1,
        "trajectory_id": "traj_runner",
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
                "token_text": "7",
                "is_stop": False,
            },
        ],
    }
    specs = [
        {
            "schema_version": 1,
            "trace_id": "traj_runner_tok000000",
            "trajectory_id": "traj_runner",
            "generated_index": 0,
            "prefix_token_count": 2,
            "target_token_id": 201,
            "target_token_text": "A",
            "target_mode": "frozen_target_only",
            "selection_reasons": ["explicit"],
            "graph_knobs": {"max_edges": 10},
            "estimated_cost": 2,
        },
        {
            "schema_version": 1,
            "trace_id": "traj_runner_tok000001",
            "trajectory_id": "traj_runner",
            "generated_index": 1,
            "prefix_token_count": 3,
            "target_token_id": 202,
            "target_token_text": "7",
            "target_mode": "frozen_target_only",
            "selection_reasons": ["numeric"],
            "graph_knobs": {"max_edges": 10},
            "estimated_cost": 3,
        },
    ]
    shards = {
        "schema_version": 1,
        "trace_specs_file": "trace_specs.jsonl",
        "cost_model": "prefix_token_count_lpt_v1",
        "shards": [
            {"shard_id": 0, "estimated_cost_sum": 3, "spec_indices": [1]},
            {"shard_id": 1, "estimated_cost_sum": 2, "spec_indices": [0]},
        ],
    }
    trajectory_path = tmp_path / "trajectory.json"
    specs_path = tmp_path / "trace_specs.jsonl"
    shards_path = tmp_path / "shards.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    specs_path.write_text(
        "\n".join(json.dumps(spec) for spec in specs) + "\n", encoding="utf-8"
    )
    shards_path.write_text(json.dumps(shards), encoding="utf-8")
    return trajectory_path, specs_path, shards_path


def test_dry_run_shard_writes_expected_files_and_metadata(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    result = dry_run_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=tmp_path / "run",
    )
    assert result["status"] == "dry_run"
    shard_dir = tmp_path / "run" / "shards" / "shard_000"
    assert (shard_dir / "shard.json").exists()
    trace_path = shard_dir / "token_000001" / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "dry_run"
    assert trace["graph_path"] is None
    assert trace["forced_target"] == {
        "token_id": 202,
        "token_text": "7",
        "target_mode": "frozen_target_only",
    }
    assert trace["selection_reasons"] == ["numeric"]
    assert trace["prefix_token_count"] == 3
    assert (shard_dir / "trace_results.jsonl").exists()


def test_list_mode_returns_specs_without_writing_token_dirs(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    rows = list_shard_specs(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=1,
    )
    assert [row["generated_index"] for row in rows] == [0]
    assert not (tmp_path / "shards").exists()


def test_aggregate_handles_dry_run_rows(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    run_root = tmp_path / "run"
    dry_run_shard(
        trajectory_path=trajectory_path,
        trace_specs_path=specs_path,
        shards_path=shards_path,
        shard_id=0,
        output_root=run_root,
    )
    aggregate = aggregate_shards(run_root)
    assert aggregate["status_counts"] == {"dry_run": 1}
    assert (run_root / "aggregate.json").exists()
    assert (run_root / "per_token_metrics.jsonl").exists()
    assert "target_token_id" in (run_root / "per_token_metrics.csv").read_text(
        encoding="utf-8"
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(SRC_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "nlp_research_project.exact_trace_bench", *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_cli_dry_run_and_aggregate(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    run_root = tmp_path / "run"
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "0",
        "--output-root",
        str(run_root),
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli("aggregate-full-answer-shards", "--run-root", str(run_root))
    assert proc.returncode == 0, proc.stderr
    assert (
        json.loads((run_root / "aggregate.json").read_text(encoding="utf-8"))[
            "token_count"
        ]
        == 1
    )


def test_cli_list_does_not_require_output_root(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "1",
        "--list",
    )
    assert proc.returncode == 0, proc.stderr
    assert "generated_index=0" in proc.stdout


def test_cli_dry_run_requires_output_root(tmp_path: Path) -> None:
    trajectory_path, specs_path, shards_path = _write_tiny_inputs(tmp_path)
    proc = _run_cli(
        "run-full-answer-shard",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--shard-id",
        "0",
        "--dry-run",
    )
    assert proc.returncode != 0
    assert "--output-root" in proc.stderr
