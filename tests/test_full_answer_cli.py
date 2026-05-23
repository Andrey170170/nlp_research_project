from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
