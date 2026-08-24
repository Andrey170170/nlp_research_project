from __future__ import annotations

import json
import inspect
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp_research_project.exact_trace_bench import baselines  # noqa: E402
from nlp_research_project.exact_trace_bench.config import base_trace_defaults  # noqa: E402
from nlp_research_project.exact_trace_bench.jobs import (  # noqa: E402
    render_fixture_prep_plan,
    render_launch_plan,
)
from experiments import run_sparsification_experiment as experiment_runner  # noqa: E402
from experiments.run_sparsification_experiment import main, run_scenario  # noqa: E402


def _overall_typed_exact_metrics(value: float = 1.0) -> dict[str, float]:
    return {
        f"overall_mean_bucket_{name.replace('<-', '_').replace('-', '_')}_exact": value
        for name in baselines.CANONICAL_BUCKET_NAMES
    }


def test_baseline_check_preserves_typed_comparison_scope() -> None:
    scientific = baselines.normalize_baseline_check(
        {"baseline_check": {"enabled": True, "mode": "gate"}}
    )
    mechanism = baselines.normalize_baseline_check(
        {
            "baseline_check": {
                "enabled": True,
                "mode": "gate",
                "scope": "same_regime_mechanism",
            }
        }
    )

    assert scientific["scope"] == "frozen_scientific"
    assert mechanism["scope"] == "same_regime_mechanism"
    with pytest.raises(ValueError, match="scope"):
        baselines.normalize_baseline_check(
            {
                "baseline_check": {
                    "enabled": True,
                    "scope": "mixed_or_unknown",
                }
            }
        )


def test_comparison_contract_pins_retention_policy_not_legacy_max_edges() -> None:
    contract = baselines._comparison_contract(
        {
            "method": "exact",
            "edge_retention_policy_id": "typed_top_p_v1",
            "max_edges": 20_000,
        }
    )

    assert contract["edge_retention_policy_id"] == "typed_top_p_v1"
    assert "max_edges" not in contract


def test_registry_loader_and_resolver_enforce_scope_at_shared_boundary(
    tmp_path: Path,
) -> None:
    scientific_path = tmp_path / "legacy-scientific.json"
    scientific_path.write_text(
        json.dumps(
            {
                "registry_id": "legacy",
                "compact_graph_reference": (
                    baselines.historical_compact_graph_reference_contract()
                ),
                "entries": {"case": {}},
            }
        )
    )
    scientific = baselines.load_baseline_registry(scientific_path)
    assert scientific.scope == "frozen_scientific"
    assert scientific.scope_declared is False
    assert scientific.compact_graph_reference == {
        "format": baselines.HISTORICAL_COMPACT_GRAPH_FORMAT
    }
    status, entry = baselines.resolve_baseline_entry(
        {
            "enabled": True,
            "scope": "frozen_scientific",
            "registry_key": "case",
            "failure_reasons": [],
        },
        registry=scientific,
        registry_path=scientific_path,
    )
    assert status["registry_scope"] == "frozen_scientific"
    assert entry == {
        "compact_graph_reference": {"format": baselines.HISTORICAL_COMPACT_GRAPH_FORMAT}
    }

    missing_graph_contract = tmp_path / "missing-graph-contract.json"
    missing_graph_contract.write_text(
        json.dumps({"registry_id": "bad", "entries": {"case": {}}})
    )
    with pytest.raises(ValueError, match="compact_graph_reference"):
        baselines.load_baseline_registry(missing_graph_contract)

    mechanism_missing_entry_scope = tmp_path / "bad-mechanism.json"
    mechanism_missing_entry_scope.write_text(
        json.dumps(
            {
                "registry_id": "bad",
                "scope": "same_regime_mechanism",
                "entries": {"case": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="must declare scope"):
        baselines.load_baseline_registry(mechanism_missing_entry_scope)

    status, entry = baselines.resolve_baseline_entry(
        {
            "enabled": True,
            "scope": "same_regime_mechanism",
            "registry_key": "case",
            "failure_reasons": [],
        },
        registry={"case": {"scope": "same_regime_mechanism"}},
        registry_path=None,
    )
    assert entry is None
    assert status["status"] == "baseline_invalid"
    assert "explicitly scoped" in " ".join(status["failure_reasons"])


def test_run_scenario_rejects_unscoped_mechanism_registry(tmp_path: Path) -> None:
    scenario = {
        **base_trace_defaults(),
        "name": "mechanism_scope_smoke",
        "stage": "test",
        "method": "exact",
        "gsm8k_indices": [828],
        "attribution_batch_size": 1,
        "feature_batch_size": 1,
        "logit_batch_size": 1,
        "decoder_chunk_size": 256,
        "cross_batch_decoder_cache_bytes": 0,
        "baseline_check": {
            "enabled": True,
            "mode": "gate",
            "scope": "same_regime_mechanism",
            "registry_key": "case",
            "baseline_required": True,
        },
    }
    result = run_scenario(
        tmp_path,
        scenario,
        env={},
        run_metadata={
            "run_id": "run",
            "run_name": "test",
            "run_description": None,
            "run_goal": None,
        },
        baseline_registry={"case": {"scope": "same_regime_mechanism"}},
    )

    assert result["status"] == "baseline_invalid"
    assert result["returncode"] is None
    assert "explicitly scoped" in " ".join(result["baseline_check"]["failure_reasons"])
    assert not (tmp_path / "mechanism_scope_smoke" / "run.log").exists()


def test_historical_baseline_comparison_uses_bucket_drift_not_policy_equality(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import (
        write_historical_tie_cutoff_graph,
        write_typed_graph,
    )

    baseline_artifacts = tmp_path / "baseline" / "artifacts"
    current_artifacts = tmp_path / "current" / "artifacts"
    scenario_root = tmp_path / "current"
    historical_step = (
        baseline_artifacts / "prompt_000" / "completion_000" / "step_000.npz"
    )
    candidate_step = (
        current_artifacts / "prompt_000" / "completion_000" / "step_000.npz"
    )
    write_typed_graph(
        candidate_step,
        bucket_values={"feature<-feature": [[18.0, 1.0], [1.0, 0.0]]},
    )
    write_historical_tie_cutoff_graph(
        historical_step,
        canonical_source=candidate_step,
    )
    baseline_result = tmp_path / "baseline" / "result.json"
    baseline_result.write_text(json.dumps({"status": "success"}))

    status, metrics = baselines.run_baseline_comparison(
        scenario_root=scenario_root,
        current_artifacts=current_artifacts,
        baseline_check={
            "enabled": True,
            "mode": "gate",
            "thresholds": {
                "overall_mean_bucket_feature_feature_weighted_jaccard_min": 0.89
            },
            "failure_reasons": [],
        },
        baseline_entry={
            "artifacts_dir": str(baseline_artifacts),
            "result_json": str(baseline_result),
            "compact_graph_reference": (
                baselines.historical_compact_graph_reference_contract()
            ),
        },
    )

    assert status["status"] == "gate_pass"
    assert status["passed"] is True
    assert metrics["overall_mean_bucket_feature_feature_exact"] == 0.0
    assert metrics["overall_mean_bucket_feature_feature_weighted_jaccard"] >= 0.89
    assert status["historical_reference"] is True
    assert status["policy_compatible"] is False
    assert status["candidate_current_policy_valid"] is True
    assert status["reference_bucket_rules_match_candidate"] is True
    assert (scenario_root / "baseline_compare.json").exists()
    comparison = json.loads((scenario_root / "baseline_compare.json").read_text())
    assert comparison["historical_reference"] is True
    assert comparison["policy_compatible"] is False
    assert comparison["candidate_current_policy_valid"] is True
    assert comparison["reference_bucket_rules_match_candidate"] is True

    row = baselines.build_scenario_metrics_row(
        scenario={"name": "current"},
        result={"status": "success"},
        baseline_status=status,
        comparison_metrics=metrics,
    )
    baselines.write_scenario_metrics(scenario_root, row, baseline_status=status)
    persisted = json.loads(
        (scenario_root / "scenario_metrics.json").read_text(encoding="utf-8")
    )
    assert persisted["metrics"]["overall_mean_bucket_feature_feature_exact"] == 0.0


def test_baseline_comparison_routes_typed_v2_reference_strictly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline_artifacts = tmp_path / "baseline" / "artifacts"
    current_artifacts = tmp_path / "current" / "artifacts"
    baseline_artifacts.mkdir(parents=True)
    current_artifacts.mkdir(parents=True)
    baseline_result = tmp_path / "baseline" / "result.json"
    baseline_result.write_text(json.dumps({"status": "success"}))

    def fake_compare(left: Path, right: Path, *, historical_reference: bool) -> dict:
        assert left == baseline_artifacts
        assert right == current_artifacts
        assert historical_reference is False
        return {
            "shared_completion_count": 1,
            "aligned_completion_count": 1,
            "aligned_step_count": 1,
            "comparison_complete": True,
            "left_only_completion_count": 0,
            "right_only_completion_count": 0,
            "historical_reference": False,
            "policy_compatible": True,
            "candidate_current_policy_valid": True,
            "overall_mean_feature_jaccard": 1.0,
            "overall_mean_target_token_match": 1.0,
            **_overall_typed_exact_metrics(),
        }

    monkeypatch.setattr(baselines, "compare_artifact_dirs", fake_compare)
    status, _metrics = baselines.run_baseline_comparison(
        scenario_root=tmp_path / "current",
        current_artifacts=current_artifacts,
        baseline_check={
            "enabled": True,
            "mode": "metrics",
            "thresholds": {},
            "failure_reasons": [],
        },
        baseline_entry={
            "artifacts_dir": str(baseline_artifacts),
            "result_json": str(baseline_result),
            "compact_graph_reference": (
                baselines.typed_compact_graph_reference_contract()
            ),
        },
    )

    assert status["status"] == "compared"
    assert status["compact_graph_reference"] == (
        baselines.typed_compact_graph_reference_contract()
    )


def test_baseline_entry_rejects_missing_or_incomplete_typed_graph_identity(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "baseline" / "artifacts"
    artifacts.mkdir(parents=True)
    result_json = tmp_path / "baseline" / "result.json"
    result_json.write_text(json.dumps({"status": "success"}))

    missing = baselines.validate_baseline_entry(
        {"artifacts_dir": str(artifacts), "result_json": str(result_json)},
        status={"failure_reasons": []},
    )
    assert missing["status"] == "baseline_invalid"
    assert "compact_graph_reference" in " ".join(missing["failure_reasons"])

    incomplete_typed = baselines.validate_baseline_entry(
        {
            "artifacts_dir": str(artifacts),
            "result_json": str(result_json),
            "compact_graph_reference": {"format": "typed_compact_graph_v2"},
        },
        status={"failure_reasons": []},
    )
    assert incomplete_typed["status"] == "baseline_invalid"
    assert "typed compact graph identity mismatch" in " ".join(
        incomplete_typed["failure_reasons"]
    )

    with pytest.raises(ValueError, match="unsupported compact graph reference"):
        baselines.build_baseline_registry_from_run_roots(
            [],
            registry_id="empty-contract",
            compact_graph_reference={},
        )


def test_baseline_comparison_rejects_incomplete_alignment(
    monkeypatch, tmp_path: Path
) -> None:
    baseline_artifacts = tmp_path / "baseline" / "artifacts"
    current_artifacts = tmp_path / "current" / "artifacts"
    baseline_artifacts.mkdir(parents=True)
    current_artifacts.mkdir(parents=True)
    baseline_result = tmp_path / "baseline" / "result.json"
    baseline_result.write_text(json.dumps({"status": "success"}))
    monkeypatch.setattr(
        baselines,
        "compare_artifact_dirs",
        lambda _left, _right, *, historical_reference: {
            "shared_completion_count": 1,
            "aligned_completion_count": 0,
            "aligned_step_count": 0,
            "comparison_complete": False,
            "historical_reference": historical_reference,
            "policy_compatible": False,
        },
    )

    status, _metrics = baselines.run_baseline_comparison(
        scenario_root=tmp_path / "current",
        current_artifacts=current_artifacts,
        baseline_check={
            "enabled": True,
            "mode": "metrics",
            "thresholds": {},
            "failure_reasons": [],
        },
        baseline_entry={
            "artifacts_dir": str(baseline_artifacts),
            "result_json": str(baseline_result),
            "compact_graph_reference": (
                baselines.historical_compact_graph_reference_contract()
            ),
        },
    )

    assert status["status"] == "compare_error"
    assert status["passed"] is False
    assert "no aligned steps" in " ".join(status["failure_reasons"])


def test_threshold_evaluation_reports_failures() -> None:
    passed, reasons = baselines.evaluate_thresholds(
        {"overall_mean_bucket_feature_feature_exact": 0.0},
        {"overall_mean_bucket_feature_feature_exact_min": 1.0},
    )

    assert passed is False
    assert "overall_mean_bucket_feature_feature_exact" in reasons[0]


def test_run_scenario_skips_required_missing_baseline(tmp_path: Path) -> None:
    scenario = {
        **base_trace_defaults(),
        "name": "missing_baseline_smoke",
        "stage": "test",
        "method": "exact",
        "gsm8k_indices": [828],
        "attribution_batch_size": 1,
        "feature_batch_size": 1,
        "logit_batch_size": 1,
        "decoder_chunk_size": 256,
        "cross_batch_decoder_cache_bytes": 0,
        "baseline_check": {
            "enabled": True,
            "mode": "metrics",
            "registry_key": "missing/key",
            "baseline_required": True,
        },
    }

    result = run_scenario(
        tmp_path,
        scenario,
        env={},
        run_metadata={
            "run_id": "run",
            "run_name": "test",
            "run_description": None,
            "run_goal": None,
        },
    )

    scenario_root = tmp_path / "missing_baseline_smoke"
    assert result["status"] == "baseline_invalid"
    assert (scenario_root / "result.json").exists()
    assert (scenario_root / "scenario_metrics.csv").exists()
    assert not (scenario_root / "run.log").exists()


def test_run_scenario_preserves_probe_and_explicitly_skips_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario = {
        **base_trace_defaults(),
        "name": "transition_probe",
        "stage": "diagnostic",
        "method": "exact",
        "baseline_check": {"enabled": True, "mode": "metrics"},
        "diagnostic_stop_mode": "transition_probe",
        "diagnostic_stop_phase4_batches": 2,
    }

    monkeypatch.setattr(
        experiment_runner,
        "resolve_baseline_entry",
        lambda status, **_kwargs: (status, {"registry_key": "unused"}),
    )
    monkeypatch.setattr(
        experiment_runner,
        "validate_baseline_entry",
        lambda _entry, *, status: status,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_comparison",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic probes must not enter baseline comparison")
        ),
    )

    def fake_run(cmd, **_kwargs):
        artifacts = Path(cmd[-1])
        completion = artifacts / "prompt_000" / "completion_000"
        completion.mkdir(parents=True)
        (completion / "completion.json").write_text(
            json.dumps(
                {
                    "status": "probe_completed",
                    "diagnostic_stop_mode": "transition_probe",
                    "phase4_batches_completed": 2,
                    "steps": [],
                }
            )
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment_runner.subprocess, "run", fake_run)
    result = run_scenario(
        tmp_path,
        scenario,
        env={},
        run_metadata={
            "run_id": "probe-run",
            "run_name": None,
            "run_description": None,
            "run_goal": None,
        },
    )

    assert result["status"] == "probe_completed"
    assert result["artifact_summary"]["completion_statuses"] == ["probe_completed"]
    assert result["baseline_check"]["status"] == "skipped_diagnostic_probe"
    assert result["baseline_check"]["passed"] is None


def test_runner_honors_metadata_failure_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text(
        json.dumps(
            {
                "metadata": {"fail_on_baseline_missing": True},
                "defaults": base_trace_defaults(),
                "scenarios": [
                    {
                        "name": "required_baseline",
                        "stage": "test",
                        "method": "exact",
                        "gsm8k_indices": [828],
                        "baseline_check": {
                            "enabled": True,
                            "mode": "metrics",
                            "registry_key": "missing/key",
                            "baseline_required": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparsification_experiment.py",
            "--scenarios-file",
            str(scenarios_file),
            "--output-root",
            str(tmp_path / "results"),
        ],
    )

    with pytest.raises(RuntimeError, match="Required baseline invalid"):
        main()


def test_fixture_prep_plan_uses_cluster_script_and_exports(tmp_path: Path) -> None:
    plan = render_fixture_prep_plan(
        cluster="cardinal",
        target_spec_file=PROJECT_ROOT
        / "experiments"
        / "exact_trace_wave0_fixture_targets.json",
        output_dir=tmp_path / "fixtures",
        immutable_workspace=False,
        live_workspace_rationale="bounded fixture debugging",
        decoder_chunk_size=512,
        cross_batch_decoder_cache_bytes=1024,
        run_name="wave0 fixture smoke",
    )

    assert plan["cluster"] == "cardinal"
    assert plan["decoder_chunk_size"] == 512
    assert "prepare_weekend_prefix_fixtures.cardinal.sbatch" in plan["sbatch_script"]
    assert "TARGET_SPEC_FILE=" in plan["sbatch_command"]
    assert "OUTPUT_DIR=" in plan["sbatch_command"]
    assert "CROSS_BATCH_DECODER_CACHE_BYTES=1024" in plan["sbatch_command"]
    assert "EXACT_TRACE_WORKSPACE_MODE=live" in plan["sbatch_command"]
    assert "EXACT_TRACE_ALLOW_LIVE_WORKSPACE=1" in plan["sbatch_command"]
    assert "UV_PROJECT_ENVIRONMENT=" in plan["sbatch_command"]
    assert "ENV_FILE=" in plan["sbatch_command"]
    assert plan["workspace_provenance"]["project_repo_state"]["commit"]
    assert plan["workspace_provenance"]["library_repo_state"]["commit"]


def test_fixture_prep_plan_rejects_live_workspace_without_rationale() -> None:
    try:
        render_fixture_prep_plan(cluster="cardinal", immutable_workspace=False)
    except ValueError as exc:
        assert "non-empty rationale" in str(exc)
    else:
        raise AssertionError("Expected live workspace without rationale to fail")


def test_launch_renderers_default_immutable_and_require_live_rationale(
    tmp_path: Path,
) -> None:
    assert (
        inspect.signature(render_launch_plan).parameters["immutable_workspace"].default
        is True
    )
    assert (
        inspect.signature(render_fixture_prep_plan)
        .parameters["immutable_workspace"]
        .default
        is True
    )
    try:
        render_launch_plan(
            cluster="granite",
            scenarios_file=tmp_path / "missing.json",
            immutable_workspace=False,
        )
    except ValueError as exc:
        assert "non-empty rationale" in str(exc)
    else:
        raise AssertionError("Expected live launch plan without rationale to fail")

    try:
        render_launch_plan(
            cluster="granite",
            scenarios_file=tmp_path / "missing.json",
            immutable_workspace=False,
            existing_workspace=tmp_path / "snapshot",
            live_workspace_rationale="invalid mixed mode",
        )
    except ValueError as exc:
        assert "existing_workspace requires immutable_workspace" in str(exc)
    else:
        raise AssertionError("Expected existing snapshot in live mode to fail")


def test_launch_plan_supports_sbatch_memory_override(tmp_path: Path) -> None:
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text(
        json.dumps(
            {
                "metadata": {"resource_profile": "standard"},
                "scenarios": [{"name": "smoke"}],
            }
        ),
        encoding="utf-8",
    )

    plan = render_launch_plan(
        cluster="granite",
        scenarios_file=scenarios_file,
        output_root=tmp_path / "runs",
        run_id="memory-override",
        immutable_workspace=False,
        live_workspace_rationale="test memory override rendering",
        walltime="01:00:00",
        mem="600G",
    )

    assert plan["mem"] == "600G"
    assert "--time=01:00:00" in plan["sbatch_argv"]
    assert "--mem=600G" in plan["sbatch_argv"]


def test_launch_plan_honors_scenario_array_concurrency(tmp_path: Path) -> None:
    scenarios_file = tmp_path / "scenarios.json"
    scenarios_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "resource_profile": "standard",
                    "array_concurrency": 2,
                    "fail_on_baseline_missing": True,
                    "fail_on_validation_fail": True,
                    "slurm": {
                        "account": "rai",
                        "partition": "rai-gpu-grn",
                        "qos": "rai-gpu-grn-short",
                        "gres": "gpu:h200:1",
                        "cpus_per_task": 12,
                        "mem": "200G",
                        "time": "02:00:00",
                    },
                },
                "scenarios": [{"name": f"row-{index}"} for index in range(5)],
            }
        ),
        encoding="utf-8",
    )

    plan = render_launch_plan(
        cluster="granite",
        scenarios_file=scenarios_file,
        output_root=tmp_path / "runs",
        run_id="throttled-array",
        immutable_workspace=False,
        live_workspace_rationale="test array rendering",
    )

    assert plan["array_range"] == "0-4%2"
    assert "--array=0-4%2" in plan["sbatch_argv"]
    assert "--account=rai" in plan["sbatch_argv"]
    assert "--partition=rai-gpu-grn" in plan["sbatch_argv"]
    assert "--qos=rai-gpu-grn-short" in plan["sbatch_argv"]
    assert "--gres=gpu:h200:1" in plan["sbatch_argv"]
    assert "--cpus-per-task=12" in plan["sbatch_argv"]
    assert "--mem=200G" in plan["sbatch_argv"]
    assert "--time=02:00:00" in plan["sbatch_argv"]
    assert "FAIL_ON_BASELINE_MISSING=1" in plan["sbatch_command"]
    assert "FAIL_ON_VALIDATION_FAIL=1" in plan["sbatch_command"]


def test_granite_h200_script_forwards_baseline_controls() -> None:
    script = (
        PROJECT_ROOT / "slurm/exact_trace_bench/trace_baseline_h200.granite.sbatch"
    ).read_text(encoding="utf-8")

    assert 'BASELINE_ARGS+=(--baseline-registry "${BASELINE_REGISTRY}")' in script
    assert "BASELINE_ARGS+=(--fail-on-baseline-missing)" in script
    assert "BASELINE_ARGS+=(--fail-on-validation-fail)" in script
    assert '"${BASELINE_ARGS[@]}"' in script


def test_build_baseline_registry_from_wave0_roots(tmp_path: Path) -> None:
    run_root = tmp_path / "ascend" / "fast" / "wave0"
    scenario_root = run_root / "ascend_fast_wave0_r1_828_base_b128_c2048_cache0g"
    artifacts_dir = scenario_root / "artifacts"
    artifacts_dir.mkdir(parents=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test prompt", encoding="utf-8")
    scenario = {
        **base_trace_defaults(),
        "name": scenario_root.name,
        "stage": "exact_trace_wave0_baseline_fast",
        "cluster": "ascend",
        "resource_profile": "standard",
        "fixture_name": "828_base",
        "fixture_kind": "base",
        "gsm8k_indices": [828],
        "prepared_prompt_file": str(prompt),
        "method": "exact",
        "decoder_chunk_size": 2048,
        "cross_batch_decoder_cache_bytes": 0,
        "attribution_batch_size": 128,
        "feature_batch_size": 128,
        "logit_batch_size": 128,
        "wave": "wave0",
        "wave0_role": "canonical_repeat",
        "wave0_repeat_index": 1,
    }
    (scenario_root / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")
    (scenario_root / "result.json").write_text(
        json.dumps({"status": "success", "run_id": "wave0"}),
        encoding="utf-8",
    )

    registry = baselines.build_baseline_registry_from_run_roots(
        [run_root],
        registry_id="test-registry",
        project_root=PROJECT_ROOT,
        library_root=PROJECT_ROOT,
    )

    default_key = "wave0/828_base/ascend/fast/fp32_default"
    repeat_key = "wave0/828_base/ascend/fast/fp32_default_r1"
    assert default_key in registry["entries"]
    assert repeat_key in registry["entries"]
    assert registry["entries"][default_key]["artifacts_dir"] == str(artifacts_dir)
    assert registry["compact_graph_reference"] == (
        baselines.typed_compact_graph_reference_contract()
    )
    assert registry["entries"][default_key]["compact_graph_reference"] == (
        baselines.typed_compact_graph_reference_contract()
    )
    assert registry["entries"][default_key]["prompt_identity"]["prepared_prompt_sha256"]
