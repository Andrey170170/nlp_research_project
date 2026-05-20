from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exact_trace_bench import baselines  # noqa: E402
from experiments.exact_trace_bench.config import base_trace_defaults  # noqa: E402
from experiments.exact_trace_bench.jobs import render_fixture_prep_plan  # noqa: E402
from experiments.run_sparsification_experiment import run_scenario  # noqa: E402


def test_baseline_comparison_writes_metrics(monkeypatch, tmp_path: Path) -> None:
    baseline_artifacts = tmp_path / "baseline" / "artifacts"
    current_artifacts = tmp_path / "current" / "artifacts"
    scenario_root = tmp_path / "current"
    baseline_artifacts.mkdir(parents=True)
    current_artifacts.mkdir(parents=True)
    baseline_result = tmp_path / "baseline" / "result.json"
    baseline_result.write_text(json.dumps({"status": "success"}))

    def fake_compare(left: Path, right: Path) -> dict:
        assert left == baseline_artifacts
        assert right == current_artifacts
        return {
            "shared_completion_count": 1,
            "left_only_completion_count": 0,
            "right_only_completion_count": 0,
            "overall_mean_feature_jaccard": 1.0,
            "overall_mean_edge_jaccard": 0.99,
            "overall_mean_weighted_edge_jaccard": 0.98,
        }

    monkeypatch.setattr(baselines, "compare_artifact_dirs", fake_compare)
    status, metrics = baselines.run_baseline_comparison(
        scenario_root=scenario_root,
        current_artifacts=current_artifacts,
        baseline_check={
            "enabled": True,
            "mode": "gate",
            "thresholds": {"overall_mean_weighted_edge_jaccard_min": 0.97},
            "failure_reasons": [],
        },
        baseline_entry={
            "artifacts_dir": str(baseline_artifacts),
            "result_json": str(baseline_result),
        },
    )

    assert status["status"] == "gate_pass"
    assert status["passed"] is True
    assert metrics["overall_mean_weighted_edge_jaccard"] == 0.98
    assert (scenario_root / "baseline_compare.json").exists()


def test_threshold_evaluation_reports_failures() -> None:
    passed, reasons = baselines.evaluate_thresholds(
        {"overall_mean_edge_jaccard": 0.9},
        {"overall_mean_edge_jaccard_min": 0.95},
    )

    assert passed is False
    assert "overall_mean_edge_jaccard" in reasons[0]


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


def test_fixture_prep_plan_uses_cluster_script_and_exports(tmp_path: Path) -> None:
    plan = render_fixture_prep_plan(
        cluster="cardinal",
        target_spec_file=PROJECT_ROOT
        / "experiments"
        / "exact_trace_wave0_fixture_targets.json",
        output_dir=tmp_path / "fixtures",
        immutable_workspace=False,
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
    assert registry["entries"][default_key]["prompt_identity"]["prepared_prompt_sha256"]
