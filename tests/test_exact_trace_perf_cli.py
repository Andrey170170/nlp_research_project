from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_sparsification_experiment as experiment_runner  # noqa: E402
from nlp_research_project.exact_trace_bench import perf_cli  # noqa: E402


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int | None = None,
        pid: int = 987_654,
    ) -> None:
        self.returncode = returncode
        self.pid = pid
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        assert self.returncode is not None
        return self.returncode


def test_suites_have_fixed_requested_cases() -> None:
    assert [(case.variant, case.fixture) for case in perf_cli.SUITES["clt-smoke"]] == [
        ("gemma3_1b_clt", "361_base")
    ]
    assert [(case.variant, case.fixture) for case in perf_cli.SUITES["clt-pair"]] == [
        ("gemma3_1b_clt", "828_base"),
        ("gemma3_1b_clt", "361_base"),
    ]
    assert [(case.variant, case.fixture) for case in perf_cli.SUITES["plt-hard"]] == [
        ("gemma3_1b_plt", "361_base")
    ]
    assert perf_cli.SUITES["all"] == (
        *perf_cli.SUITES["clt-pair"],
        *perf_cli.SUITES["plt-hard"],
    )


def test_fidelity_thresholds_are_explicit() -> None:
    assert perf_cli.FIDELITY_THRESHOLDS["bounded"] == {
        "worst_step_feature_jaccard_min": 0.98,
        "worst_step_all_edge_jaccard_min": 0.98,
        "worst_step_all_edge_top256_jaccard_min": 0.98,
        "worst_step_all_edge_weighted_jaccard_min": 0.98,
        "worst_step_target_token_match_min": 1.0,
        "worst_step_all_edge_normalized_l1_deviation_max": 0.02,
    }
    assert perf_cli.FIDELITY_THRESHOLDS["exact"] == {
        "worst_step_feature_jaccard_min": 1.0,
        "worst_step_all_edge_jaccard_min": 1.0,
        "worst_step_all_edge_top256_jaccard_min": 1.0,
        "worst_step_all_edge_weighted_jaccard_min": 0.999999,
        "worst_step_target_token_match_min": 1.0,
        "worst_step_all_edge_normalized_l1_deviation_max": 0.000001,
    }


def test_profiling_parser_accepts_execution_batches_without_total(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Phase 4 batch 1 in 4.00s",
                "Phase 4 batch 2 in 6.00s",
                (
                    "Forward pass completed in 1.50s | rss=3.03 GiB, "
                    "rss_current=1.80 GiB, proc_anon=1.30 GiB, "
                    "proc_file=0.47 GiB, cuda_alloc=6.44 GiB, "
                    "cuda_reserved=24.84 GiB, cuda_peak_alloc=22.73 GiB, "
                    "cuda_peak_reserved=24.84 GiB"
                ),
                ("Attribution completed in 10.00s | phase4_execution_batch_count=2"),
            ]
        )
    )

    summary = experiment_runner._extract_benchmark_metrics(log_path)

    assert summary["phase4_batches_observed"] == 2
    assert summary["phase4_total_batches"] == 2
    assert summary["phase4_avg_batch_seconds"] == 5.0
    assert summary["phase4_projected_total_seconds"] == 10.0
    assert summary["peak_rss_gib"] == 1.8
    assert summary["peak_cuda_allocated_gib"] == 6.44
    assert summary["peak_cuda_reserved_gib"] == 24.84
    assert summary["peak_cuda_peak_allocated_gib"] == 22.73


def test_gpu_resource_summary_reports_useful_utilization(
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "gpu_samples.csv"
    samples_path.write_text(
        "\n".join(
            [
                "2026/07/24 22:00:00.000, 10, 2, 100.0, 20000, 143771",
                "2026/07/24 22:00:01.000, 30, 6, 200.0, 45000, 143771",
            ]
        )
    )

    summary = perf_cli._gpu_resource_summary(samples_path)

    assert summary["gpu_sample_count"] == 2
    assert summary["gpu_sm_utilization_mean_percent"] == 20.0
    assert summary["gpu_sm_utilization_p95_percent"] == 30.0
    assert summary["gpu_memory_utilization_mean_percent"] == 4.0
    assert summary["gpu_power_max_watts"] == 200.0
    assert summary["gpu_framebuffer_peak_mib"] == 45000.0
    assert summary["gpu_framebuffer_peak_fraction"] == pytest.approx(45000 / 143771)


def test_gpu_resource_summary_marks_empty_samples_explicitly(
    tmp_path: Path,
) -> None:
    samples_path = tmp_path / "gpu_samples.csv"
    samples_path.write_text("")

    assert perf_cli._gpu_resource_summary(samples_path) == {"gpu_sample_count": 0}


def test_stream_runner_cleans_up_sampler_when_runner_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _FakeProcess()
    calls = 0

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        nonlocal calls
        calls += 1
        if calls == 1:
            return sampler
        raise OSError("runner startup failed")

    monkeypatch.setattr(perf_cli.subprocess, "Popen", fake_popen)

    with pytest.raises(OSError, match="runner startup failed"):
        perf_cli._stream_runner(["runner"], output_root=tmp_path)

    assert sampler.terminated is True
    assert sampler.waited is True
    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    assert summary["resource_validation_passed"] is False
    assert summary["gpu_sampling_status"] == "no_samples"


def test_stream_runner_cleans_up_both_processes_on_tail_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _FakeProcess()
    runner = _FakeProcess()
    processes = iter((sampler, runner))
    popen_kwargs: list[dict[str, object]] = []

    def fake_popen(*_args: object, **kwargs: object) -> _FakeProcess:
        popen_kwargs.append(kwargs)
        return next(processes)

    monkeypatch.setattr(
        perf_cli.subprocess,
        "Popen",
        fake_popen,
    )
    group_signals: list[tuple[int, int]] = []

    def fake_killpg(pid: int, sig: int) -> None:
        group_signals.append((pid, sig))
        if sig == perf_cli.signal.SIGTERM:
            runner.returncode = -sig

    monkeypatch.setattr(
        perf_cli.os,
        "killpg",
        fake_killpg,
    )

    def fail_tail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tail failed")

    monkeypatch.setattr(perf_cli, "_tail_logs", fail_tail)

    with pytest.raises(RuntimeError, match="tail failed"):
        perf_cli._stream_runner(["runner"], output_root=tmp_path)

    assert popen_kwargs[1]["start_new_session"] is True
    assert group_signals == [
        (runner.pid, perf_cli.signal.SIGTERM),
        (runner.pid, perf_cli.signal.SIGKILL),
    ]
    assert runner.terminated is False
    assert runner.waited is True
    assert sampler.terminated is True
    assert sampler.waited is True


def test_stream_runner_reports_sampler_early_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _FakeProcess(returncode=1)
    runner = _FakeProcess(returncode=0)
    processes = iter((sampler, runner))
    monkeypatch.setattr(
        perf_cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(processes),
    )

    assert perf_cli._stream_runner(["runner"], output_root=tmp_path) == 0

    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    assert summary["resource_validation_passed"] is False
    assert summary["gpu_sampling_status"] == "exited_early"
    assert summary["gpu_sampler_returncode"] == 1


def test_case_scenario_reuses_canonical_builder_and_adds_gate() -> None:
    case = perf_cli.Case("gemma3_1b_plt", "361_base")
    payload = perf_cli._case_scenario(case, "bounded")
    scenario = payload["scenarios"][0]

    assert scenario["fixture_name"] == "361_base"
    assert scenario["transcoder_provider_family"] == "gemmascope2-plt-1b-small"
    assert scenario["model_name"] == "google/gemma-3-1b-it"
    assert payload["defaults"]["method"] == "exact"
    assert scenario["baseline_check"] == {
        "enabled": True,
        "mode": "gate",
        "registry_key": case.key,
        "baseline_required": True,
        "thresholds": perf_cli.FIDELITY_THRESHOLDS["bounded"],
    }


@pytest.mark.parametrize(
    ("profile", "decoder_chunk_size", "decoder_cache_bytes"),
    [
        ("plt-bounded-fast-v1", 32768, 0),
        ("plt-bounded-fast-v2", 65536, 0),
        ("plt-bounded-fast-v3", 65536, 16 * 1024**3),
    ],
)
def test_bounded_fast_profile_changes_only_plt_physical_controls(
    profile: str,
    decoder_chunk_size: int,
    decoder_cache_bytes: int,
) -> None:
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")

    plt_payload = perf_cli._case_scenario(plt_case, "bounded", profile)
    plt = plt_payload["scenarios"][0]
    clt = perf_cli._case_scenario(clt_case, "bounded", profile)["scenarios"][0]

    expected_overrides = {
        "decoder_chunk_size": decoder_chunk_size,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    }
    if decoder_cache_bytes:
        expected_overrides["cross_batch_decoder_cache_bytes"] = decoder_cache_bytes
    assert perf_cli._candidate_overrides(plt_case, profile) == expected_overrides
    assert plt["attribution_batch_size"] == 128
    assert plt["feature_batch_size"] == 128
    assert plt["logit_batch_size"] == 128
    assert plt_payload["defaults"]["attribution_update_interval"] == 4
    assert plt["decoder_chunk_size"] == decoder_chunk_size
    assert plt["cross_batch_decoder_cache_bytes"] == decoder_cache_bytes
    assert plt["phase4_execution_batch_max_rows"] == 256
    assert plt_payload["defaults"]["row_subchunk_size"] is None
    assert clt["decoder_chunk_size"] == 4096
    assert "phase4_execution_batch_max_rows" not in clt


@pytest.mark.parametrize(
    "profile",
    [
        "plt-bounded-fast-v1",
        "plt-bounded-fast-v2",
        "plt-bounded-fast-v3",
    ],
)
def test_bounded_fast_profile_rejects_exact_fidelity(
    tmp_path: Path,
    profile: str,
) -> None:
    with pytest.raises(ValueError, match="bounded-only"):
        perf_cli.main(
            [
                "run",
                "plt-hard",
                "--fidelity",
                "exact",
                "--candidate-profile",
                profile,
                "--dry-run",
                "--output-root",
                str(tmp_path),
            ]
        )


def test_h200_guard_requires_slurm_and_h200() -> None:
    with pytest.raises(RuntimeError, match="SLURM allocation"):
        perf_cli.assert_h200_allocation(
            {"slurm_job_id": None, "gpus": [{"name": "NVIDIA H200"}]},
            allow_test_environment=False,
        )
    with pytest.raises(RuntimeError, match="requires H200"):
        perf_cli.assert_h200_allocation(
            {"slurm_job_id": "123", "gpus": [{"name": "NVIDIA A100"}]},
            allow_test_environment=False,
        )
    perf_cli.assert_h200_allocation(
        {
            "slurm_job_id": "123",
            "slurm_cluster_name": "granite",
            "slurm_job_partition": "rai-gpu-grn",
            "gpus": [{"name": "NVIDIA H200", "memory_total_mib": 141_000}],
        },
        allow_test_environment=False,
    )
    perf_cli.assert_h200_allocation(
        {"slurm_job_id": None, "gpus": []},
        allow_test_environment=True,
    )


@pytest.mark.parametrize(
    ("duration_seconds", "performance_passed", "passed"),
    [
        (600.0, True, True),
        (600.01, False, False),
    ],
)
def test_result_report_enforces_plt_duration_target(
    tmp_path: Path,
    duration_seconds: float,
    performance_passed: bool,
    passed: bool,
) -> None:
    case = perf_cli.Case("gemma3_1b_plt", "361_base")
    scenario_root = tmp_path / "perf_gemma3_1b_plt_361_base"
    scenario_root.mkdir()
    (scenario_root / "result.json").write_text(
        json.dumps(
            {
                "duration_seconds": duration_seconds,
                "status": "success",
                "baseline_check": {"passed": True, "failure_reasons": []},
                "profiling_summary": {
                    "phase3_duration_seconds": 10.0,
                    "phase4_duration_seconds": 500.0,
                },
            }
        )
    )
    (scenario_root / "baseline_compare.json").write_text(
        json.dumps(
            {
                "worst_step_feature_jaccard": 1.0,
                "worst_step_all_edge_jaccard": 0.99,
                "worst_step_all_edge_top256_jaccard": 0.98,
                "worst_step_all_edge_weighted_jaccard": 0.97,
                "worst_step_target_token_match": 1.0,
                "worst_step_all_edge_normalized_l1_deviation": 0.01,
                "worst_step_evidence": {"worst_step_all_edge_jaccard": []},
            }
        )
    )
    (tmp_path / "resource_summary.json").write_text(
        json.dumps(
            {
                "gpu_sm_utilization_mean_percent": 25.0,
                "resource_validation_passed": True,
            }
        )
    )

    report = perf_cli._result_report(
        case,
        candidate_root=tmp_path,
        baseline_entry={"duration_seconds": 1200.0},
    )

    assert report["speedup"] == pytest.approx(1200.0 / duration_seconds)
    assert report["top256_edge_jaccard"] == 0.98
    assert report["edge_magnitude_l1_deviation"] == 0.01
    assert report["performance_target_seconds"] == 600.0
    assert report["performance_stretch_target_seconds"] == 300.0
    assert report["performance_stretch_passed"] is (duration_seconds <= 300.0)
    assert report["parity_passed"] is True
    assert report["performance_passed"] is performance_passed
    assert report["passed"] is passed
    assert report["resource_validation_passed"] is True
    assert report["resource_failure_reasons"] == []
    assert report["profiling_summary"]["phase4_duration_seconds"] == 500.0
    assert report["resource_summary"]["gpu_sm_utilization_mean_percent"] == 25.0


def test_clt_report_has_no_hard_duration_target(tmp_path: Path) -> None:
    case = perf_cli.Case("gemma3_1b_clt", "361_base")
    scenario_root = tmp_path / "perf_gemma3_1b_clt_361_base"
    scenario_root.mkdir()
    (scenario_root / "result.json").write_text(
        json.dumps(
            {
                "duration_seconds": 10_000.0,
                "status": "success",
                "baseline_check": {"passed": True, "failure_reasons": []},
            }
        )
    )

    report = perf_cli._result_report(
        case,
        candidate_root=tmp_path,
        baseline_entry={"duration_seconds": 100.0},
    )

    assert report["performance_target_seconds"] is None
    assert report["performance_passed"] is None
    assert report["passed"] is True


def test_dry_run_skips_allocation_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        perf_cli,
        "gpu_provenance",
        lambda: pytest.fail("dry-run must not inspect GPUs"),
    )

    exit_code = perf_cli.main(
        [
            "run",
            "clt-smoke",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "test-run",
        ]
    )

    assert exit_code == 0
    assert "run_sparsification_experiment.py" in capsys.readouterr().out
    assert not (tmp_path / "test-run").exists()


def test_run_finalizes_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = perf_cli.SUITES["clt-smoke"][0]
    monkeypatch.setattr(perf_cli, "gpu_provenance", lambda: {"gpus": []})
    monkeypatch.setattr(
        perf_cli, "assert_h200_allocation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(perf_cli, "capture_source_state", lambda: {"state": "same"})
    monkeypatch.setattr(
        perf_cli,
        "execution_evidence",
        lambda *_args, **_kwargs: {"timing_evidence_eligible": False},
    )
    monkeypatch.setattr(
        perf_cli,
        "_baseline_entries",
        lambda: {case.key: {"duration_seconds": 1.0}},
    )

    def fail_runner(*_args, **_kwargs) -> int:
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(perf_cli, "_stream_runner", fail_runner)

    with pytest.raises(RuntimeError, match="runner exploded"):
        perf_cli.main(
            [
                "run",
                "clt-smoke",
                "--output-root",
                str(tmp_path),
                "--run-id",
                "failed-run",
            ]
        )

    report = json.loads(
        (tmp_path / "failed-run" / "performance_report.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["current_case"] == case.key
    assert report["error"] == "RuntimeError: runner exploded"
    assert report["source_state_after"] == {"state": "same"}
    assert report["reports"] == []


def test_run_goal_reaches_runner_command_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = perf_cli.SUITES["clt-smoke"][0]
    goal = "Measure the column kernel candidate under bounded parity."
    captured_command: list[str] = []
    monkeypatch.setattr(perf_cli, "gpu_provenance", lambda: {"gpus": []})
    monkeypatch.setattr(
        perf_cli, "assert_h200_allocation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(perf_cli, "capture_source_state", lambda: {"state": "same"})
    monkeypatch.setattr(
        perf_cli,
        "execution_evidence",
        lambda *_args, **_kwargs: {"timing_evidence_eligible": False},
    )
    monkeypatch.setattr(
        perf_cli,
        "_baseline_entries",
        lambda: {case.key: {"duration_seconds": 1.0}},
    )

    def capture_runner(command: list[str], *, output_root: Path) -> int:
        del output_root
        captured_command.extend(command)
        raise RuntimeError("stop after command capture")

    monkeypatch.setattr(perf_cli, "_stream_runner", capture_runner)

    with pytest.raises(RuntimeError, match="stop after command capture"):
        perf_cli.main(
            [
                "run",
                "clt-smoke",
                "--output-root",
                str(tmp_path),
                "--run-id",
                "goal-run",
                "--run-goal",
                goal,
            ]
        )

    manifest = json.loads((tmp_path / "goal-run" / "run_manifest.json").read_text())
    assert manifest["run_goal"] == goal
    assert captured_command[captured_command.index("--run-goal") + 1] == goal


def test_manifest_contract_names_live_workspace_exception() -> None:
    source = Path(perf_cli.__file__).read_text()
    assert '"workspace_mode": "live"' in source
    assert '"live_workspace_rationale"' in source
    assert '"no_edits_during_run_enforced": True' in source


def test_source_hash_changes_when_already_dirty_content_changes(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    tracked.write_text("dirty one\n")
    before = perf_cli.source_content_sha256(tmp_path)

    tracked.write_text("dirty two\n")

    assert perf_cli.source_content_sha256(tmp_path) != before


def test_test_override_is_explicitly_non_evidence() -> None:
    evidence = perf_cli.execution_evidence(
        {"gpus": [{"name": "NVIDIA A100"}]},
        allow_test_environment=True,
    )

    assert evidence["test_environment_override_used"] is True
    assert evidence["timing_evidence_eligible"] is False
    assert evidence["timing_evidence_status"] == "test_override_non_evidence"


def test_print_report_handles_missing_compare_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    perf_cli._print_report(
        {
            "case": "performance/test",
            "candidate_duration_seconds": 1.0,
            "baseline_duration_seconds": 2.0,
            "speedup": 2.0,
            "passed": False,
        }
    )

    output = capsys.readouterr().out
    assert "all_edge=n/a" in output
    assert "gate=FAIL" in output


def test_print_report_includes_compact_phase_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    perf_cli._print_report(
        {
            "case": "performance/test",
            "candidate_duration_seconds": 20.0,
            "baseline_duration_seconds": 40.0,
            "speedup": 2.0,
            "performance_target_seconds": 600.0,
            "profiling_summary": {
                "phase3_duration_seconds": 2.5,
                "phase4_duration_seconds": 15.25,
                "phase4_avg_batch_seconds": 0.5,
                "phase4_batches_observed": 4,
                "phase4_total_batches": 30,
            },
            "parity_passed": True,
            "performance_passed": True,
            "resource_validation_passed": True,
            "passed": True,
        }
    )

    output = capsys.readouterr().out
    assert "phase3=2.50s" in output
    assert "phase4=15.25s" in output
    assert "phase4_batch=0.50s" in output
    assert "batches=4/30" in output
    assert "parity=PASS performance=PASS stretch=MISS resource=PASS gate=PASS" in output


@pytest.mark.parametrize(
    ("reports", "expected"),
    [
        (
            [
                {
                    "parity_passed": True,
                    "performance_passed": None,
                    "runner_returncode": 0,
                }
            ],
            {
                "parity_passed": True,
                "performance_passed": None,
                "passed": True,
            },
        ),
        (
            [
                {
                    "parity_passed": True,
                    "performance_passed": False,
                    "runner_returncode": 0,
                }
            ],
            {
                "parity_passed": True,
                "performance_passed": False,
                "passed": False,
            },
        ),
        (
            [
                {
                    "parity_passed": False,
                    "performance_passed": True,
                    "runner_returncode": 0,
                }
            ],
            {
                "parity_passed": False,
                "performance_passed": True,
                "passed": False,
            },
        ),
    ],
)
def test_gate_summary_keeps_parity_performance_and_overall_separate(
    reports: list[dict[str, object]],
    expected: dict[str, object],
) -> None:
    assert perf_cli._gate_summary(reports) == expected
