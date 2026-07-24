from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli


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


def test_result_report_calculates_speedup_and_parity(tmp_path: Path) -> None:
    case = perf_cli.Case("gemma3_1b_clt", "361_base")
    scenario_root = tmp_path / "perf_gemma3_1b_clt_361_base"
    scenario_root.mkdir()
    (scenario_root / "result.json").write_text(
        json.dumps(
            {
                "duration_seconds": 50.0,
                "status": "success",
                "baseline_check": {"passed": True, "failure_reasons": []},
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

    report = perf_cli._result_report(
        case,
        candidate_root=tmp_path,
        baseline_entry={"duration_seconds": 100.0},
    )

    assert report["speedup"] == 2.0
    assert report["top256_edge_jaccard"] == 0.98
    assert report["edge_magnitude_l1_deviation"] == 0.01
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
    monkeypatch.setattr(perf_cli, "assert_h200_allocation", lambda *_args, **_kwargs: None)
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

    report = json.loads((tmp_path / "failed-run" / "performance_report.json").read_text())
    assert report["status"] == "failed"
    assert report["current_case"] == case.key
    assert report["error"] == "RuntimeError: runner exploded"
    assert report["source_state_after"] == {"state": "same"}
    assert report["reports"] == []


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
