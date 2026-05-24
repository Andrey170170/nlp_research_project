from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nlp_research_project.exact_trace_bench.jobs import render_full_answer_shard_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def tiny_trajectory() -> dict:
    return {
        "schema_version": 1,
        "trajectory_id": "traj_cli",
        "prompt_token_count": 4,
        "prompt_token_ids": [1, 2, 3, 4],
        "generated_tokens": [
            {
                "generated_index": 0,
                "absolute_token_position": 4,
                "token_id": 10,
                "token_text": "A",
                "logprob": -0.1,
                "is_stop": False,
            },
            {
                "generated_index": 1,
                "absolute_token_position": 5,
                "token_id": 11,
                "token_text": " 7",
                "logprob": -3.0,
                "is_stop": False,
            },
            {
                "generated_index": 2,
                "absolute_token_position": 6,
                "token_id": 12,
                "token_text": ".",
                "is_stop": False,
            },
        ],
    }


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(SRC_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "nlp_research_project.exact_trace_bench", *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_full_answer_cli_help_is_login_safe() -> None:
    assert run_cli("build-full-answer-trace-specs", "--help").returncode == 0
    assert run_cli("build-full-answer-shards", "--help").returncode == 0
    assert run_cli("run-full-answer-shard", "--help").returncode == 0
    assert run_cli("aggregate-full-answer-shards", "--help").returncode == 0
    assert run_cli("run-full-answer-trajectory", "--help").returncode == 0
    assert run_cli("submit-full-answer-trajectory", "--help").returncode == 0
    assert run_cli("launch-full-answer-shards", "--help").returncode == 0


def test_full_answer_cli_writes_planning_artifacts(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    out_dir = tmp_path / "out"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")

    proc = run_cli(
        "build-full-answer-trace-specs",
        "--trajectory",
        str(trajectory_path),
        "--select",
        "numeric",
        "--indices",
        "0",
        "--high-surprisal-top-k",
        "1",
        "--output-dir",
        str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr
    selection = json.loads(
        (out_dir / "trace_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_indices"] == [0, 1]
    specs_lines = (
        (out_dir / "trace_specs.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(specs_lines) == 2

    shards_path = tmp_path / "shards.json"
    proc = run_cli(
        "build-full-answer-shards",
        "--trace-specs",
        str(out_dir / "trace_specs.jsonl"),
        "--shard-count",
        "2",
        "--output",
        str(shards_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(shards_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_full_answer_trajectory_print_only_plan_uses_snapshot_template(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question: 1+1?", encoding="utf-8")
    proc = run_cli(
        "submit-full-answer-trajectory",
        "--cluster",
        "ascend",
        "--prompt-path",
        str(prompt_path),
        "--output",
        str(tmp_path / "trajectory.json"),
        "--max-new-tokens",
        "4",
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--workspace-label",
        "test-full-answer",
        "--print-only",
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["immutable_workspace"] is True
    assert "test-full-answer" in plan["workspace_root"]
    assert plan["sbatch_script"].endswith(
        "slurm/exact_trace_bench/full_answer_prepare.ascend.sbatch"
    )
    assert "WORKSPACE_ROOT=" in plan["sbatch_command"]
    assert "LIB_WORKSPACE_ROOT=" in plan["sbatch_command"]


def test_full_answer_shard_print_only_plan_uses_snapshot_paths(tmp_path: Path) -> None:
    input_dir = (
        PROJECT_ROOT
        / "experiments"
        / "generated"
        / "exact_trace_bench"
        / "unit_full_answer"
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = input_dir / "trajectory.json"
    specs_path = input_dir / "trace_specs.jsonl"
    shards_path = input_dir / "shards.json"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")
    specs_path.write_text("", encoding="utf-8")
    shards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_specs_file": str(specs_path),
                "cost_model": "prefix_token_count_lpt_v1",
                "shards": [
                    {"shard_id": 0, "estimated_cost_sum": 0, "spec_indices": []},
                    {"shard_id": 1, "estimated_cost_sum": 0, "spec_indices": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    proc = run_cli(
        "launch-full-answer-shards",
        "--cluster",
        "ascend",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--output-root",
        str(tmp_path / "runs"),
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--workspace-label",
        "test-shards",
        "--run-id",
        "unit-run",
        "--run-description",
        "comma, safe metadata",
        "--run-goal",
        "propagate metadata args",
        "--print-only",
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["array_range"] == "0-1"
    assert "test-shards" in plan["workspace_root"]
    assert plan["sbatch_script"].endswith(
        "slurm/exact_trace_bench/full_answer_trace.ascend.sbatch"
    )
    assert "test-shards" in plan["trajectory_path"]
    assert plan["trajectory_path"].endswith(
        "experiments/generated/exact_trace_bench/unit_full_answer/trajectory.json"
    )
    assert plan["trace_specs_path"].endswith(
        "experiments/generated/exact_trace_bench/unit_full_answer/trace_specs.jsonl"
    )
    assert plan["shards_path"].endswith(
        "experiments/generated/exact_trace_bench/unit_full_answer/shards.json"
    )
    assert plan["output_root"].endswith("/runs/unit-run")
    assert "WORKSPACE_ROOT=" in plan["sbatch_command"]
    assert "LIB_WORKSPACE_ROOT=" in plan["sbatch_command"]
    assert "--run-id unit-run" in plan["sbatch_command"]
    assert "--run-name" in plan["sbatch_command"]
    assert "--run-description" in plan["sbatch_command"]
    assert "--run-goal" in plan["sbatch_command"]


def test_full_answer_shard_plan_supports_quad_partial_array(tmp_path: Path) -> None:
    input_dir = (
        PROJECT_ROOT
        / "experiments"
        / "generated"
        / "exact_trace_bench"
        / "unit_full_answer_quad"
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = input_dir / "trajectory.json"
    specs_path = input_dir / "trace_specs.jsonl"
    shards_path = input_dir / "shards.json"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")
    specs_path.write_text("", encoding="utf-8")
    shards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_specs_file": str(specs_path),
                "cost_model": "prefix_token_count_lpt_v1",
                "shards": [
                    {"shard_id": i, "estimated_cost_sum": 0, "spec_indices": []}
                    for i in range(40)
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = run_cli(
        "launch-full-answer-shards",
        "--cluster",
        "ascend",
        "--trace-resource-profile",
        "quad",
        "--array-range",
        "0-39",
        "--trajectory",
        str(trajectory_path),
        "--trace-specs",
        str(specs_path),
        "--shards",
        str(shards_path),
        "--output-root",
        str(tmp_path / "runs"),
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--workspace-label",
        "test-quad-shards",
        "--run-id",
        "unit-quad-run",
        "--print-only",
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["resource_profile"] == "quad"
    assert plan["array_range"] == "0-39"
    assert plan["sbatch_script"].endswith(
        "slurm/exact_trace_bench/full_answer_trace_quad.ascend.sbatch"
    )
    assert "--array=0-39" in plan["sbatch_command"]


def test_full_answer_shard_plan_rejects_existing_output_root(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    trace_specs_path = tmp_path / "trace_specs.jsonl"
    shards_path = tmp_path / "shards.json"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")
    trace_specs_path.write_text("", encoding="utf-8")
    shards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_specs_file": str(trace_specs_path),
                "cost_model": "prefix_token_count_lpt_v1",
                "shards": [
                    {"shard_id": 0, "estimated_cost_sum": 0, "spec_indices": []}
                ],
            }
        ),
        encoding="utf-8",
    )
    existing_root = tmp_path / "runs" / "existing-run"
    existing_root.mkdir(parents=True)
    try:
        render_full_answer_shard_plan(
            cluster="ascend",
            trajectory_path=trajectory_path,
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            output_root=tmp_path / "runs",
            run_id="existing-run",
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected existing output root to be rejected")


def test_full_answer_shard_plan_rejects_malformed_shard_schema(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    trace_specs_path = tmp_path / "trace_specs.jsonl"
    shards_path = tmp_path / "shards.json"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")
    trace_specs_path.write_text("", encoding="utf-8")
    shards_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_specs_file": str(trace_specs_path),
                "cost_model": "prefix_token_count_lpt_v1",
                "shards": [{"shard_id": 0}],
            }
        ),
        encoding="utf-8",
    )
    try:
        render_full_answer_shard_plan(
            cluster="ascend",
            trajectory_path=trajectory_path,
            trace_specs_path=trace_specs_path,
            shards_path=shards_path,
            output_root=tmp_path / "runs",
        )
    except ValueError as exc:
        assert "estimated_cost_sum" in str(exc)
    else:
        raise AssertionError("expected malformed shards schema to be rejected")
