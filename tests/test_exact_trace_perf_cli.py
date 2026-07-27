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
    (
        "profile",
        "decoder_chunk_size",
        "decoder_cache_bytes",
        "session_capacity",
        "execution_batch_rows",
        "tape_batch_window",
        "tape_max_bytes",
        "decoder_page_prefetch_depth",
    ),
    [
        ("plt-bounded-fast-v1", 32768, 0, 256, 256, 1, 0, 0),
        ("plt-bounded-fast-v2", 65536, 0, 256, 256, 1, 0, 0),
        ("plt-bounded-fast-v3", 65536, 16 * 1024**3, 256, 256, 1, 0, 0),
        ("plt-bounded-tape-v1", 65536, 0, 256, 256, 2, 12 * 1024**3, 0),
        (
            "plt-bounded-tape-prefetch-v1",
            65536,
            0,
            256,
            256,
            2,
            12 * 1024**3,
            1,
        ),
        ("plt-bounded-frontier-v1", 65536, 0, 512, 512, 1, 0, 0),
    ],
)
def test_bounded_fast_profile_changes_only_plt_physical_controls(
    profile: str,
    decoder_chunk_size: int,
    decoder_cache_bytes: int,
    session_capacity: int,
    execution_batch_rows: int,
    tape_batch_window: int,
    tape_max_bytes: int,
    decoder_page_prefetch_depth: int,
) -> None:
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")

    plt_payload = perf_cli._case_scenario(plt_case, "bounded", profile)
    plt = plt_payload["scenarios"][0]
    clt = perf_cli._case_scenario(clt_case, "bounded", profile)["scenarios"][0]

    expected_overrides = {
        "decoder_chunk_size": decoder_chunk_size,
        "nnsight_session_capacity": session_capacity,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": execution_batch_rows,
    }
    if decoder_cache_bytes:
        expected_overrides["cross_batch_decoder_cache_bytes"] = decoder_cache_bytes
    if tape_batch_window > 1:
        expected_overrides["feature_vjp_tape_batch_window"] = tape_batch_window
        expected_overrides["feature_vjp_tape_max_bytes"] = tape_max_bytes
    if decoder_page_prefetch_depth:
        expected_overrides["decoder_page_prefetch_depth"] = decoder_page_prefetch_depth
    assert perf_cli._candidate_overrides(plt_case, profile) == expected_overrides
    assert plt["attribution_batch_size"] == 128
    assert plt["feature_batch_size"] == 128
    assert plt["logit_batch_size"] == 128
    assert plt_payload["defaults"]["attribution_update_interval"] == 4
    assert plt["decoder_chunk_size"] == decoder_chunk_size
    assert plt["cross_batch_decoder_cache_bytes"] == decoder_cache_bytes
    assert plt["phase4_execution_batch_max_rows"] == execution_batch_rows
    assert plt.get("feature_vjp_tape_batch_window", 1) == tape_batch_window
    assert plt.get("feature_vjp_tape_max_bytes", 0) == tape_max_bytes
    assert plt.get("decoder_page_prefetch_depth", 0) == decoder_page_prefetch_depth
    assert plt_payload["defaults"]["row_subchunk_size"] is None
    assert clt["decoder_chunk_size"] == 4096
    assert "phase4_execution_batch_max_rows" not in clt


@pytest.mark.parametrize(
    "profile",
    [
        "plt-bounded-fast-v1",
        "plt-bounded-fast-v2",
        "plt-bounded-fast-v3",
        "plt-bounded-tape-v1",
        "plt-bounded-tape-prefetch-v1",
        "plt-bounded-frontier-v1",
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


@pytest.mark.parametrize(
    ("profile", "tape_batch_window"),
    [("plt-active-rows-v1", 1), ("plt-active-rows-tape-v1", 2)],
)
def test_active_rows_profiles_match_exact_baseline_and_are_exact_eligible(
    profile: str,
    tape_batch_window: int,
) -> None:
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")
    expected = {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": tape_batch_window,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    }
    if tape_batch_window > 1:
        expected["feature_vjp_tape_max_bytes"] = 12 * 1024**3

    assert perf_cli._candidate_overrides(plt_case, profile) == expected
    assert perf_cli._candidate_overrides(clt_case, profile) == {}
    assert profile not in perf_cli.BOUNDED_ONLY_CANDIDATE_PROFILES
    payload = perf_cli._case_scenario(plt_case, "exact", profile)
    scenario = payload["scenarios"][0]
    assert scenario["decoder_chunk_size"] == 4096
    assert scenario["feature_vjp_tape_batch_window"] == tape_batch_window
    assert scenario["decoder_active_row_residency"] is True
    assert scenario["decoder_active_row_max_bytes"] == 1024**3
    assert (
        scenario["baseline_check"]["thresholds"]
        == perf_cli.FIDELITY_THRESHOLDS["exact"]
    )


def test_large_chunk_active_rows_profile_is_exact_eligible_and_single_knob_variant() -> None:
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")
    incumbent = perf_cli._candidate_overrides(plt_case, "plt-active-rows-v1")
    candidate = perf_cli._candidate_overrides(plt_case, "plt-active-rows-c65536-v1")

    assert candidate == {**incumbent, "decoder_chunk_size": 65536}
    assert {
        key: (incumbent[key], candidate[key])
        for key in incumbent | candidate
        if incumbent.get(key) != candidate.get(key)
    } == {"decoder_chunk_size": (4096, 65536)}
    assert "phase0_decoder_row_ranges" not in candidate
    assert "feature_vjp_tape_max_bytes" not in candidate
    assert candidate["feature_vjp_tape_batch_window"] == 1
    assert perf_cli._candidate_overrides(clt_case, "plt-active-rows-c65536-v1") == {}
    assert "plt-active-rows-c65536-v1" not in perf_cli.BOUNDED_ONLY_CANDIDATE_PROFILES

    scenario = perf_cli._case_scenario(
        plt_case, "exact", "plt-active-rows-c65536-v1"
    )["scenarios"][0]
    assert scenario["decoder_chunk_size"] == 65536
    assert scenario["baseline_check"]["thresholds"] == perf_cli.FIDELITY_THRESHOLDS["exact"]


def test_phase0_coalesced_rows_profile_is_exact_eligible_and_provider_scoped() -> None:
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")
    expected = perf_cli._candidate_overrides(
        plt_case, "plt-phase0-coalesced-rows-v1"
    )

    assert expected == {
        **perf_cli._candidate_overrides(plt_case, "plt-active-rows-v1"),
        "phase0_decoder_row_ranges": True,
    }
    assert (
        perf_cli._candidate_overrides(clt_case, "plt-phase0-coalesced-rows-v1")
        == {}
    )
    assert (
        "plt-phase0-coalesced-rows-v1"
        not in perf_cli.BOUNDED_ONLY_CANDIDATE_PROFILES
    )
    scenario = perf_cli._case_scenario(
        plt_case, "exact", "plt-phase0-coalesced-rows-v1"
    )["scenarios"][0]
    assert scenario["phase0_decoder_row_ranges"] is True
    assert scenario["decoder_active_row_residency"] is True
    assert (
        scenario["baseline_check"]["thresholds"]
        == perf_cli.FIDELITY_THRESHOLDS["exact"]
    )


@pytest.mark.parametrize(
    ("profile", "capacity"),
    [
        ("clt-phase1-cap128-v1", 128),
        ("clt-phase1-cap256-v1", 256),
        ("clt-phase1-cap512-v1", 512),
    ],
)
def test_clt_phase1_cap_profile_is_exact_eligible_and_provider_scoped(
    profile: str,
    capacity: int,
) -> None:
    clt_case = perf_cli.Case("gemma3_1b_clt", "361_base")
    plt_case = perf_cli.Case("gemma3_1b_plt", "361_base")
    expected = {
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": capacity,
        "nnsight_session_capacity": capacity,
        "phase3_compute_microbatch_max_rows": capacity,
        "phase4_execution_batch_max_rows": capacity,
    }

    assert perf_cli._candidate_overrides(clt_case, profile) == expected
    assert perf_cli._candidate_overrides(plt_case, profile) == {}
    assert profile not in perf_cli.BOUNDED_ONLY_CANDIDATE_PROFILES

    scenario = perf_cli._case_scenario(clt_case, "exact", profile)["scenarios"][0]
    assert {key: scenario[key] for key in expected} == expected
    assert scenario["decoder_chunk_size"] == 4096
    assert scenario["attribution_batch_size"] == 1000
    assert scenario["feature_batch_size"] == 1000
    assert scenario["logit_batch_size"] == 1000
    assert scenario["baseline_check"]["thresholds"] == perf_cli.FIDELITY_THRESHOLDS["exact"]


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
    (tmp_path / "resource_summary.json").write_text(
        json.dumps(
            {"resource_validation_passed": True, "gpu_framebuffer_peak_mib": 1024.0}
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


def _active_row_diagnostics(
    *,
    effective: bool = True,
    phase0_ranges: bool = False,
    phase0_ranges_effective: bool = True,
) -> dict[str, object]:
    resident_bytes = perf_cli.ACTIVE_ROW_EXPECTED_BYTES
    diagnostics = {
        "requested": True,
        "effective": effective,
        "fallback_reason": None if effective else "estimated_bytes_exceed_max",
        "max_bytes_requested": 1024**3,
        "max_bytes_effective": 1024**3 if effective else 0,
        "resident": {
            "row_count": 66_158 if effective else 0,
            "bytes": resident_bytes if effective else 0,
            "estimated_bytes": resident_bytes,
            "device": "cuda:0" if effective else None,
            "owner_count": 1 if effective else 0,
        },
        "build": {
            "source": "phase0_fused_seed",
            "count": 1 if effective else 0,
            "seconds": 1.25 if effective else None,
            "traversal_bytes": 0,
            "decoder_page_load_count": 0,
            "decoder_load_bytes": 0,
        },
        "seed": {
            "capture_seconds": 137.0 if effective else None,
            "shared_traversal_bytes": 15_665_725_440 if effective else 0,
            "shared_decoder_page_load_count": 1660 if effective else 0,
            "shared_decoder_load_bytes": 15_665_725_440 if effective else 0,
            "unique_row_count": 42_000 if effective else 0,
            "bytes": 96_768_000 if effective else 0,
            "materialization_seconds": 0.25 if effective else None,
            "materialization_h2d_bytes": resident_bytes if effective else 0,
            "missing_keys": 0,
        },
        "phase4": {
            "decoder_page_load_count_delta": 0,
            "decoder_load_bytes_delta": 0,
        },
    }
    if phase0_ranges:
        if phase0_ranges_effective:
            diagnostics["seed"]["shared_traversal_bytes"] = 100_000_000
            diagnostics["seed"]["shared_decoder_page_load_count"] = 0
            diagnostics["seed"]["shared_decoder_load_bytes"] = 0
        diagnostics["phase0_decoder_row_ranges"] = {
            "requested": True,
            "effective": phase0_ranges_effective,
            "fallback_reason": (
                None
                if phase0_ranges_effective
                else "singleton_range_fraction_exceeds_max"
            ),
            "unique_row_count": 42_000,
            "range_request_count": 2_048 if phase0_ranges_effective else 0,
            "logical_materialized_bytes": (
                100_000_000 if phase0_ranges_effective else 15_665_725_440
            ),
            "baseline_full_page_bytes": 15_665_725_440,
        }
    return diagnostics


def _active_row_report(
    tmp_path: Path,
    *,
    duration_seconds: float,
    framebuffer_peak_mib: float = 26_113.0,
    diagnostics: dict[str, object] | None = None,
    phase0_ranges_requested: bool = False,
) -> dict[str, object]:
    case = perf_cli.Case("gemma3_1b_plt", "361_base")
    scenario_root = tmp_path / "perf_gemma3_1b_plt_361_base"
    scenario_root.mkdir()
    (scenario_root / "scenario.json").write_text(
        json.dumps(
            {
                "decoder_active_row_residency": True,
                "decoder_active_row_max_bytes": 1024**3,
                "phase0_decoder_row_ranges": phase0_ranges_requested,
            }
        )
    )
    (scenario_root / "result.json").write_text(
        json.dumps(
            {
                "duration_seconds": duration_seconds,
                "status": "success",
                "baseline_check": {"passed": True, "failure_reasons": []},
                "artifact_summary": {
                    "decoder_active_row_residency": (
                        diagnostics
                        if diagnostics is not None
                        else _active_row_diagnostics()
                    )
                },
            }
        )
    )
    (tmp_path / "resource_summary.json").write_text(
        json.dumps(
            {
                "resource_validation_passed": True,
                "gpu_framebuffer_peak_mib": framebuffer_peak_mib,
            }
        )
    )
    return perf_cli._result_report(
        case,
        candidate_root=tmp_path,
        baseline_entry={"duration_seconds": 307.34},
    )


@pytest.mark.parametrize("duration_seconds", [65.0, 230.0, 245.0])
def test_active_row_report_passes_reconciled_fused_target(
    tmp_path: Path,
    duration_seconds: float,
) -> None:
    report = _active_row_report(tmp_path, duration_seconds=duration_seconds)

    assert report["passed"] is True
    assert report["mechanism_validation_passed"] is True
    assert report["resource_gate_passed"] is True
    assert report["reconciliation_required"] is False
    assert report["framebuffer_comparison"]["reference"] == (
        perf_cli.ACTIVE_ROW_FRAMEBUFFER_REFERENCE
    )


@pytest.mark.parametrize("duration_seconds", [245.01, 287.26])
def test_active_row_report_requires_reconciliation_above_fused_target(
    tmp_path: Path,
    duration_seconds: float,
) -> None:
    report = _active_row_report(tmp_path, duration_seconds=duration_seconds)

    assert report["passed"] is False
    assert report["performance_passed"] is False
    assert report["reconciliation_required"] is True
    assert "reconciliation is required" in " ".join(report["failure_reasons"])
    assert report["active_row_initial_predicted_duration_seconds"] == [65.0, 90.0]
    assert report["active_row_fused_target_seconds"] == 245.0


def test_active_row_report_fails_explicit_fallback(tmp_path: Path) -> None:
    report = _active_row_report(
        tmp_path,
        duration_seconds=75.0,
        diagnostics=_active_row_diagnostics(effective=False),
    )

    assert report["passed"] is False
    assert report["mechanism_validation_passed"] is False
    assert "estimated_bytes_exceed_max" in " ".join(report["failure_reasons"])


def test_phase0_coalesced_row_report_passes_structured_mechanism_gate(
    tmp_path: Path,
) -> None:
    diagnostics = _active_row_diagnostics(phase0_ranges=True)
    report = _active_row_report(
        tmp_path,
        duration_seconds=75.0,
        diagnostics=diagnostics,
        phase0_ranges_requested=True,
    )

    assert report["passed"] is True
    assert report["mechanism_validation_passed"] is True
    assert report["phase0_decoder_row_ranges_requested"] is True
    assert report["phase0_decoder_row_ranges"] == diagnostics[
        "phase0_decoder_row_ranges"
    ]


def test_phase0_coalesced_row_report_rejects_explicit_exact_fallback(
    tmp_path: Path,
) -> None:
    report = _active_row_report(
        tmp_path,
        duration_seconds=75.0,
        diagnostics=_active_row_diagnostics(
            phase0_ranges=True,
            phase0_ranges_effective=False,
        ),
        phase0_ranges_requested=True,
    )

    assert report["passed"] is False
    assert report["parity_passed"] is True
    assert report["mechanism_validation_passed"] is False
    assert "fell back to exact full-page reads" in " ".join(
        report["mechanism_failure_reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("range_request_count", 30_000, "at most half"),
        ("range_request_count", True, "at most half"),
        (
            "logical_materialized_bytes",
            15_665_725_440,
            "below baseline full-page bytes",
        ),
        (
            "logical_materialized_bytes",
            True,
            "below baseline full-page bytes",
        ),
    ],
)
def test_phase0_coalesced_row_gate_rejects_ineffective_evidence(
    field: str,
    value: int | bool,
    reason: str,
) -> None:
    diagnostics = _active_row_diagnostics(phase0_ranges=True)
    diagnostics["phase0_decoder_row_ranges"][field] = value

    passed, reasons = perf_cli._active_row_mechanism_gate(
        True,
        diagnostics,
        1024**3,
        phase0_ranges_requested=True,
    )

    assert passed is False
    assert reason in " ".join(reasons)


def test_phase0_coalesced_row_gate_rejects_boolean_legacy_counters() -> None:
    diagnostics = _active_row_diagnostics(phase0_ranges=True)
    diagnostics["seed"]["shared_decoder_page_load_count"] = False

    passed, reasons = perf_cli._active_row_mechanism_gate(
        True,
        diagnostics,
        1024**3,
        phase0_ranges_requested=True,
    )

    assert passed is False
    assert "zero legacy decoder page loads" in " ".join(reasons)


@pytest.mark.parametrize(
    ("active_rows", "max_bytes", "reason"),
    [
        (False, 1024**3, "decoder_active_row_residency=true"),
        (True, 0, "positive decoder_active_row_max_bytes"),
    ],
)
def test_phase0_coalesced_row_report_rejects_invalid_scenario_dependencies(
    tmp_path: Path,
    active_rows: bool,
    max_bytes: int,
    reason: str,
) -> None:
    case = perf_cli.Case("gemma3_1b_plt", "361_base")
    scenario_root = tmp_path / "perf_gemma3_1b_plt_361_base"
    scenario_root.mkdir()
    (scenario_root / "scenario.json").write_text(
        json.dumps(
            {
                "decoder_active_row_residency": active_rows,
                "decoder_active_row_max_bytes": max_bytes,
                "phase0_decoder_row_ranges": True,
            }
        )
    )
    (scenario_root / "result.json").write_text(
        json.dumps(
            {
                "duration_seconds": 75.0,
                "status": "success",
                "baseline_check": {"passed": True, "failure_reasons": []},
                "artifact_summary": {},
            }
        )
    )
    (tmp_path / "resource_summary.json").write_text(
        json.dumps(
            {
                "resource_validation_passed": True,
                "gpu_framebuffer_peak_mib": 26_113.0,
            }
        )
    )

    report = perf_cli._result_report(
        case,
        candidate_root=tmp_path,
        baseline_entry={"duration_seconds": 307.34},
    )

    assert report["mechanism_validation_passed"] is False
    assert reason in " ".join(report["mechanism_failure_reasons"])


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("build_count", 0, "build count"),
        ("build_source", "page_scan", "build source"),
        ("resident_bytes", 2 * 1024**3, "configured byte cap"),
        ("seed_missing_keys", 1, "cover every final"),
        ("phase4_load_count", 1, "must both equal zero"),
        ("phase4_load_bytes", 4096, "must both equal zero"),
    ],
)
def test_active_row_mechanism_gate_rejects_invalid_evidence(
    field: str,
    value: int | str,
    reason: str,
) -> None:
    diagnostics = _active_row_diagnostics()
    if field == "build_count":
        diagnostics["build"]["count"] = value
    elif field == "build_source":
        diagnostics["build"]["source"] = value
    elif field == "resident_bytes":
        diagnostics["resident"]["bytes"] = value
    elif field == "seed_missing_keys":
        diagnostics["seed"]["missing_keys"] = value
    elif field == "phase4_load_count":
        diagnostics["phase4"]["decoder_page_load_count_delta"] = value
    else:
        diagnostics["phase4"]["decoder_load_bytes_delta"] = value

    passed, reasons = perf_cli._active_row_mechanism_gate(
        True,
        diagnostics,
        1024**3,
    )

    assert passed is False
    assert reason in " ".join(reasons)


@pytest.mark.parametrize(
    ("peak_mib", "passed"),
    [(26_259.0, True), (26_259.01, False)],
)
def test_active_row_framebuffer_gate_uses_audited_resident_allowance(
    tmp_path: Path,
    peak_mib: float,
    passed: bool,
) -> None:
    report = _active_row_report(
        tmp_path,
        duration_seconds=75.0,
        framebuffer_peak_mib=peak_mib,
    )

    assert report["framebuffer_passed"] is passed
    assert report["resource_gate_passed"] is passed
    assert report["passed"] is passed
    comparison = report["framebuffer_comparison"]
    assert comparison["reference_peak_mib"] == 26_113.0
    assert comparison["expected_resident_bytes"] == 152_428_032
    assert comparison["allowance_mib"] == 146
    assert comparison["allowance_rounding"] == "ceil_bytes_to_mib"
    assert comparison["limit_mib"] == 26_259.0


def test_artifact_summary_keeps_active_row_structured_diagnostics(
    tmp_path: Path,
) -> None:
    prompt_root = tmp_path / "prompt_000"
    completion_root = prompt_root / "completion_000"
    completion_root.mkdir(parents=True)
    (prompt_root / "prompt_meta.json").write_text(
        json.dumps({"fixture_name": "361_base"})
    )
    diagnostics = _active_row_diagnostics()
    (completion_root / "completion.json").write_text(
        json.dumps({"steps": [{"decoder_active_row_residency": diagnostics}]})
    )

    summary = experiment_runner._summarize_artifacts(tmp_path)

    assert summary["decoder_active_row_residency"] == diagnostics


def test_artifact_summary_surfaces_authoritative_completion_timings(
    tmp_path: Path,
) -> None:
    prompt_root = tmp_path / "prompt_000"
    completion_root = prompt_root / "completion_000"
    completion_root.mkdir(parents=True)
    (prompt_root / "prompt_meta.json").write_text(
        json.dumps({"fixture_name": "361_base"})
    )
    timing_summary = {
        "completion_end_to_end_seconds": 84.945389,
        "totals": {"attribution_seconds": 83.363731},
        "step_count": 1,
    }
    (completion_root / "completion.json").write_text(
        json.dumps(
            {
                "timing_summary": timing_summary,
                "steps": [
                    {
                        "telemetry_summary": {
                            "wall_clock_elapsed_ms_by_name_top": {
                                "attribute.done": 83_187.122009,
                                "phase0.precompute": 15_287.529237,
                            }
                        }
                    }
                ],
            }
        )
    )

    summary = experiment_runner._summarize_artifacts(tmp_path)

    assert summary["timing_summary"] == timing_summary
    assert summary["telemetry_durations_seconds"] == {
        "attribution_duration_seconds": 83.187122,
        "phase0_duration_seconds": 15.287529,
    }


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
                "completion_end_to_end_seconds": 18.75,
                "attribution_duration_seconds": 17.5,
                "phase0_duration_seconds": 1.25,
                "phase3_duration_seconds": 2.5,
                "phase4_duration_seconds": 15.25,
                "phase4_avg_batch_seconds": 0.5,
                "phase4_batches_observed": 4,
                "phase4_total_batches": 30,
            },
            "parity_passed": True,
            "performance_passed": True,
            "resource_validation_passed": True,
            "resource_gate_passed": True,
            "passed": True,
        }
    )

    output = capsys.readouterr().out
    assert "completion=18.75s" in output
    assert "attribution=17.50s" in output
    assert "phase0=1.25s" in output
    assert "phase3=2.50s" in output
    assert "phase4=15.25s" in output
    assert "phase4_batch=0.50s" in output
    assert "batches=4/30" in output
    assert (
        "parity=PASS performance=PASS stretch=MISS resource=PASS "
        "mechanism=n/a reconciliation=OK gate=PASS" in output
    )


@pytest.mark.parametrize(
    ("reports", "expected"),
    [
        (
            [
                {
                    "parity_passed": True,
                    "performance_passed": None,
                    "runner_returncode": 0,
                    "resource_gate_passed": True,
                    "mechanism_validation_passed": None,
                    "reconciliation_required": False,
                }
            ],
            {
                "parity_passed": True,
                "performance_passed": None,
                "resource_gate_passed": True,
                "mechanism_validation_passed": None,
                "reconciliation_required": False,
                "passed": True,
            },
        ),
        (
            [
                {
                    "parity_passed": True,
                    "performance_passed": False,
                    "runner_returncode": 0,
                    "resource_gate_passed": True,
                    "mechanism_validation_passed": None,
                    "reconciliation_required": False,
                }
            ],
            {
                "parity_passed": True,
                "performance_passed": False,
                "resource_gate_passed": True,
                "mechanism_validation_passed": None,
                "reconciliation_required": False,
                "passed": False,
            },
        ),
        (
            [
                {
                    "parity_passed": False,
                    "performance_passed": True,
                    "runner_returncode": 0,
                    "resource_gate_passed": True,
                    "mechanism_validation_passed": None,
                    "reconciliation_required": False,
                }
            ],
            {
                "parity_passed": False,
                "performance_passed": True,
                "resource_gate_passed": True,
                "mechanism_validation_passed": None,
                "reconciliation_required": False,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_gate_passed", False),
        ("mechanism_validation_passed", False),
        ("reconciliation_required", True),
    ],
)
def test_gate_summary_rejects_each_active_row_acceptance_failure(
    field: str,
    value: bool,
) -> None:
    report = {
        "parity_passed": True,
        "performance_passed": True,
        "runner_returncode": 0,
        "resource_gate_passed": True,
        "mechanism_validation_passed": True,
        "reconciliation_required": False,
    }
    report[field] = value

    summary = perf_cli._gate_summary([report])

    assert summary["passed"] is False
