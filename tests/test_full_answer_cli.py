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
    assert run_cli("audit-full-answer-prefix-views", "--help").returncode == 0
    assert run_cli("run-full-answer-trajectory", "--help").returncode == 0
    assert run_cli("sample-full-answer-trajectories", "--help").returncode == 0
    assert run_cli("submit-full-answer-trajectory", "--help").returncode == 0
    assert run_cli("launch-full-answer-shards", "--help").returncode == 0
    assert run_cli("plot-full-answer-temporal", "--help").returncode == 0
    assert run_cli("classify-full-answer-roles", "--help").returncode == 0
    assert run_cli("build-role-matched-calibration-manifest", "--help").returncode == 0
    assert run_cli("apply-role-classification-reviews", "--help").returncode == 0
    assert run_cli("build-decoder-signature-cache", "--help").returncode == 0
    assert run_cli("download-transcoders", "--help").returncode == 0
    assert run_cli("run-metric-calibration", "--help").returncode == 0
    assert run_cli("plot-metric-calibration", "--help").returncode == 0
    assert run_cli("compare-full-answer-stability", "--help").returncode == 0
    assert run_cli("diagnose-full-answer-stability", "--help").returncode == 0


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
    knobs = json.loads(specs_lines[0])["graph_knobs"]
    assert knobs["verbose_attribution"] is False
    assert knobs["profile_attribution"] is False
    assert knobs["decoder_chunk_size"] == 256
    assert knobs["cross_batch_decoder_cache_bytes"] == 8589934592
    assert knobs["transcoder_architecture"] == "clt"
    assert knobs["transcoder_provider_family"] == "gemmascope2-clt-1b-medium-affine"
    assert knobs["repo_id"] == "google/gemma-scope-2-1b-it"
    assert knobs["clt_subfolder"] == "clt/width_262k_l0_medium_affine"
    assert knobs["phase4_row_reduction"] == "gpu_v1"
    assert knobs["phase4_refresh_optimization"] == "v1"
    assert knobs["phase4_refresh_active_row_accumulation"] == "direct_v1"
    assert knobs["row_store_preallocate"] is True
    assert knobs["phase4_refresh_prepared_chunk_cache_bytes"] == 0

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


def test_full_answer_trace_spec_perf_knob_overrides(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    out_dir = tmp_path / "out"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")

    proc = run_cli(
        "build-full-answer-trace-specs",
        "--trajectory",
        str(trajectory_path),
        "--indices",
        "0",
        "--output-dir",
        str(out_dir),
        "--decoder-chunk-size",
        "128",
        "--cross-batch-decoder-cache-bytes",
        "0",
        "--transcoder-architecture",
        "plt",
        "--transcoder-provider-family",
        "gemmascope2-plt-1b-big-affine",
        "--phase4-refresh-optimization",
        "off",
        "--phase4-refresh-active-row-accumulation",
        "zero_fill",
        "--phase4-row-reduction",
        "off",
        "--no-row-store-preallocate",
        "--phase4-refresh-prepared-chunk-cache-bytes",
        "0",
        "--phase4-row-executor",
        "streaming_v1",
        "--phase4-scheduler-mode",
        "planner_v1",
        "--phase4-scheduler-telemetry-detail",
        "debug",
        "--phase3-frontier-buffer-relative-epsilon",
        "0.01",
        "--phase3-frontier-buffer-max-extra",
        "256",
        "--phase4-frontier-buffer-relative-epsilon",
        "0.02",
        "--phase4-frontier-buffer-max-extra-per-refresh",
        "16",
        "--phase4-frontier-buffer-max-extra-total",
        "128",
        "--cross-cluster-debug",
        "--capture-phase0-donor-bundle",
        "--capture-phase3-seed-bundle",
        "--capture-feature-semantic-descriptors",
        "--semantic-descriptor-top-k",
        "1024",
        "--semantic-descriptor-dim",
        "32",
        "--plan-feature-batch-size",
        "--feature-batch-size-max",
        "64",
        "--row-subchunk-size",
        "32",
        "--input-context-mode",
        "full_sequence",
        "--verbose-attribution",
        "--profile-attribution",
    )
    assert proc.returncode == 0, proc.stderr
    spec = json.loads((out_dir / "trace_specs.jsonl").read_text(encoding="utf-8"))
    knobs = spec["graph_knobs"]
    assert knobs["decoder_chunk_size"] == 128
    assert knobs["cross_batch_decoder_cache_bytes"] == 0
    assert knobs["transcoder_architecture"] == "plt"
    assert knobs["transcoder_provider_family"] == "gemmascope2-plt-1b-big-affine"
    assert knobs["model_name"] == "google/gemma-3-1b-it"
    assert knobs["repo_id"] == "google/gemma-scope-2-1b-it"
    assert knobs["layer_count"] == 26
    assert knobs["phase4_refresh_optimization"] == "off"
    assert knobs["phase4_refresh_active_row_accumulation"] == "zero_fill"
    assert knobs["phase4_row_reduction"] == "off"
    assert knobs["row_store_preallocate"] is False
    assert knobs["phase4_row_executor"] == "streaming_v1"
    assert knobs["phase4_scheduler_mode"] == "planner_v1"
    assert knobs["phase4_scheduler_telemetry_detail"] == "debug"
    assert knobs["phase3_frontier_buffer_relative_epsilon"] == 0.01
    assert knobs["phase3_frontier_buffer_max_extra"] == 256
    assert knobs["phase4_frontier_buffer_relative_epsilon"] == 0.02
    assert knobs["phase4_frontier_buffer_max_extra_per_refresh"] == 16
    assert knobs["phase4_frontier_buffer_max_extra_total"] == 128
    assert knobs["cross_cluster_debug"] is True
    assert knobs["capture_phase0_donor_bundle"] is True
    assert knobs["capture_phase3_seed_bundle"] is True
    assert knobs["capture_feature_semantic_descriptors"] is True
    assert knobs["semantic_descriptor_top_k"] == 1024
    assert knobs["semantic_descriptor_dim"] == 32
    assert knobs["plan_feature_batch_size"] is True
    assert knobs["feature_batch_size_max"] == 64
    assert knobs["row_subchunk_size"] == 32
    assert knobs["input_context_mode"] == "full_sequence"
    assert knobs["verbose_attribution"] is True
    assert knobs["profile_attribution"] is True


def test_full_answer_trace_spec_provider_family_resolves_complete_plt_config(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    out_dir = tmp_path / "out"
    trajectory_path.write_text(json.dumps(tiny_trajectory()), encoding="utf-8")

    proc = run_cli(
        "build-full-answer-trace-specs",
        "--trajectory",
        str(trajectory_path),
        "--indices",
        "0",
        "--output-dir",
        str(out_dir),
        "--transcoder-provider-family",
        "gemmascope2-plt-4b-small-affine",
    )
    assert proc.returncode == 0, proc.stderr
    spec = json.loads((out_dir / "trace_specs.jsonl").read_text(encoding="utf-8"))
    knobs = spec["graph_knobs"]
    assert knobs["transcoder_architecture"] == "plt"
    assert knobs["transcoder_provider_family"] == "gemmascope2-plt-4b-small-affine"
    assert knobs["model_name"] == "google/gemma-3-4b-it"
    assert knobs["repo_id"] == "google/gemma-scope-2-4b-it"
    assert knobs["layer_count"] == 34
    assert knobs["cross_batch_decoder_cache_bytes"] == 0


def test_download_transcoders_dry_run_is_login_safe(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=fake-token-for-dry-run\n", encoding="utf-8")
    proc = run_cli(
        "download-transcoders",
        "--provider-family",
        "gemmascope2-plt-1b-big-affine",
        "--provider-family",
        "gemmascope2-plt-4b-big-affine",
        "--env-file",
        str(env_file),
        "--dry-run",
    )
    assert proc.returncode == 0, proc.stderr
    assert "fake-token-for-dry-run" not in proc.stdout
    assert "fake-token-for-dry-run" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "dry_run"
    assert payload["env_file_loaded"] is True
    downloads = payload["downloads"]
    assert [item["config"]["transcoder_provider_family"] for item in downloads] == [
        "gemmascope2-plt-1b-big-affine",
        "gemmascope2-plt-4b-big-affine",
    ]
    assert downloads[0]["pattern_count"] == 26
    assert downloads[1]["pattern_count"] == 34
    assert downloads[1]["config"]["model_name"] == "google/gemma-3-4b-it"


def test_download_transcoders_rejects_local_non_dry_run(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=fake-token-for-guard\n", encoding="utf-8")
    proc = run_cli(
        "download-transcoders",
        "--provider-family",
        "gemmascope2-plt-1b-big-affine",
        "--env-file",
        str(env_file),
    )
    assert proc.returncode != 0
    assert "run under SLURM" in proc.stderr
    assert "fake-token-for-guard" not in proc.stdout
    assert "fake-token-for-guard" not in proc.stderr


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


def test_full_answer_trajectory_sampling_print_only_plan(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question: 6*7?", encoding="utf-8")
    proc = run_cli(
        "submit-full-answer-trajectory",
        "--cluster",
        "ascend",
        "--prompt-path",
        str(prompt_path),
        "--output",
        str(tmp_path / "samples"),
        "--max-new-tokens",
        "4",
        "--seed",
        "100",
        "--sample-until-success",
        "--expected-answer",
        "42",
        "--max-success-tokens",
        "2",
        "--max-attempts",
        "5",
        "--collect-all",
        "--time-limit-seconds",
        "30",
        "--snapshot-root",
        str(tmp_path / "snapshots"),
        "--print-only",
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["sample_until_success"] is True
    assert plan["collect_all"] is True
    assert plan["max_attempts"] == 5
    assert "SAMPLE_UNTIL_SUCCESS=1" in plan["sbatch_command"]
    assert "COLLECT_ALL=1" in plan["sbatch_command"]
    assert "EXPECTED_ANSWER=42" in plan["sbatch_command"]


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
        "--mem",
        "600G",
        "--partition",
        "gpu",
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
    assert plan["mem"] == "600G"
    assert plan["partition"] == "gpu"
    assert plan["sbatch_script"].endswith(
        "slurm/exact_trace_bench/full_answer_trace_quad.ascend.sbatch"
    )
    assert "--array=0-39" in plan["sbatch_command"]
    assert "--mem=600G" in plan["sbatch_command"]
    assert "--partition=gpu" in plan["sbatch_command"]


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
