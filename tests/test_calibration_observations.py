from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_sparsification_experiment as experiment_runner  # noqa: E402

from nlp_research_project.exact_trace_bench.calibration_observations import (  # noqa: E402
    apply_campaign_reference_to_baseline_check,
    build_calibration_observation,
    parse_fidelity_policy,
)


def _campaign() -> dict[str, object]:
    return {
        "campaign_id": "wave-c-provider",
        "reference": {
            "kind": "baseline_registry",
            "registry_key": "provider/reference",
        },
    }


@pytest.mark.parametrize(
    ("mode", "extra"),
    [
        ("exact", {}),
        (
            "bounded",
            {
                "governor_fidelity_evidence_name": "wave-c",
                "governor_fidelity_evidence_version": "1",
                "governor_fidelity_budget": {
                    "allowed_sensitive_axes": ["feature_microbatch_size"],
                    "metrics": {
                        "overall_mean_bucket_feature_feature_weighted_jaccard": {
                            "minimum": 0.99,
                            "confidence": 0.95,
                        }
                    },
                },
            },
        ),
        (
            "best_effort",
            {
                "governor_fidelity_budget": {
                    "allowed_sensitive_axes": ["feature_microbatch_size"],
                    "penalty": 2.5,
                },
            },
        ),
        (
            "research",
            {"governor_fidelity_override_fields": ["feature_batch_size"]},
        ),
    ],
)
def test_parse_fidelity_policy_supports_four_modes(
    mode: str, extra: dict[str, object]
) -> None:
    policy = parse_fidelity_policy({"governor_fidelity_mode": mode, **extra})

    assert policy.mode == mode


def test_parse_fidelity_policy_rejects_unstructured_bounded_budget() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        parse_fidelity_policy(
            {
                "governor_fidelity_mode": "bounded",
                "governor_fidelity_budget": {
                    "allowed_sensitive_axes": ["feature_microbatch_size"],
                    "metrics": {"weighted_edge_jaccard": 0.99},
                },
            }
        )


def test_campaign_reference_configures_existing_baseline_comparison() -> None:
    scenario = apply_campaign_reference_to_baseline_check(
        {"calibration_campaign": _campaign()}
    )

    assert scenario["baseline_check"] == {
        "enabled": True,
        "mode": "metrics",
        "registry_key": "provider/reference",
        "baseline_required": True,
    }


def test_build_observation_joins_structured_artifacts(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    artifacts = scenario_root / "artifacts"
    completion = artifacts / "prompt_000" / "completion_000"
    completion.mkdir(parents=True)
    (completion / "completion.json").write_text(
        json.dumps(
            {
                "prompt_id": "prompt_000",
                "completion_id": "completion_000",
                "duration_seconds": 12.0,
                "semantic_fingerprint": "semantic",
                "execution_fingerprint": "execution",
                "resource_snapshot": {"cuda_peak_reserved_gib": 4.0},
                "timing_summary": {
                    "completion_end_to_end_seconds": 12.0,
                    "totals": {"attribution_seconds": 10.0},
                },
                "steps": [
                    {
                        "resource_snapshot": {
                            "rss_gib": 2.0,
                            "cuda_peak_reserved_gib": 4.0,
                        }
                    }
                ],
            }
        )
    )
    (completion / "telemetry.live.jsonl").write_text(
        json.dumps(
            {
                "name": "phase4.complete",
                "attrs": {"duration_seconds": 8.0, "cuda_peak_bytes": 1024},
            }
        )
        + "\n"
        + json.dumps(
            {
                "name": "planning.post_phase0",
                "attrs": {
                    "session_capacity": 256,
                    "feature_microbatch_size": 128,
                    "candidate_count": 7,
                    "admissible_candidate_count": 3,
                    "selected_objective": [["predicted_walltime_high_seconds", 10.5]],
                    "execution_fingerprint": "plan-execution",
                },
            }
        )
        + "\n"
    )
    comparison_path = scenario_root / "baseline_compare.json"
    comparison_path.write_text(
        json.dumps(
            {
                "comparison_complete": True,
                "aligned_completion_count": 1,
                "aligned_step_count": 1,
                "overall_mean_feature_jaccard": 1.0,
                "overall_mean_bucket_feature_feature_support_jaccard": 0.99,
                "overall_mean_bucket_feature_feature_weighted_jaccard": 0.995,
                "overall_mean_bucket_feature_feature_top256_jaccard": 1.0,
            }
        )
    )
    sidecar = tmp_path / "123.gpulog"
    sidecar.write_text(
        "Timestamp GPU_utilization(%) GPU_VRAM(%) GPU_VRAM\n"
        "2026-07-21T12-00-00 75 0.5 71680\n"
    )
    scenario = {
        "name": "case",
        "method": "exact",
        "calibration_case": "execution_b256",
        "calibration_factor": "physical_envelope",
        "calibration_campaign": _campaign(),
        "governor_fidelity_mode": "exact",
        "feature_batch_size": 128,
        "phase4_execution_batch_max_rows": 256,
    }
    result = {
        "status": "success",
        "returncode": 0,
        "duration_seconds": 13.0,
        "output_dir": str(artifacts),
        "run_metadata": {"run_id": "run"},
        "baseline_check": {
            "status": "compared",
            "passed": None,
            "registry_key": "provider/reference",
            "comparison_json": str(comparison_path),
        },
    }

    observation = build_calibration_observation(
        scenario_root=scenario_root,
        scenario=scenario,
        result=result,
        baseline_entry={
            "registry_key": "provider/reference",
            "artifacts_dir": "/reference/artifacts",
        },
        resource_sidecar_paths=[sidecar],
        environ={"SLURM_JOB_ID": "123"},
    )

    assert observation is not None
    assert observation["observation_id"].startswith("calobs-")
    assert observation["campaign"]["case"] == "execution_b256"
    assert observation["decision_axes"] == [
        {
            "scenario_field": "feature_batch_size",
            "axis_id": "feature_batch_size",
            "sensitivity": "semantic",
            "requested_value": 128,
        },
        {
            "scenario_field": "phase4_execution_batch_max_rows",
            "axis_id": "feature_microbatch_size",
            "sensitivity": "numerically_sensitive",
            "requested_value": 256,
        },
    ]
    assert observation["runtime"]["phase_telemetry"]["event_count"] == 2
    assert observation["runtime"]["planning"] == [
        {
            "epoch": "post_phase0",
            "selected_vector": {
                "session_capacity": 256,
                "feature_microbatch_size": 128,
            },
            "candidate_count": 7,
            "admissible_candidate_count": 3,
            "selected_objective": [["predicted_walltime_high_seconds", 10.5]],
            "execution_fingerprint": "plan-execution",
            "semantic_fingerprint": None,
        }
    ]
    assert observation["resources"]["step_peaks"]["cuda_peak_reserved_gib"] == 4.0
    assert observation["resources"]["gpu_sidecar"]["sample_count"] == 1
    assert (
        observation["fidelity"]["comparison"][
            "overall_mean_bucket_feature_feature_weighted_jaccard"
        ]
        == 0.995
    )
    assert observation["uncertainty"]["missing_measurements"] == []
    assert observation["provenance"]["slurm"]["SLURM_JOB_ID"] == "123"
    assert len(observation["observation_fingerprint"]) == 64


def test_timeout_is_runtime_lower_bound_observation(tmp_path: Path) -> None:
    scenario = {
        "calibration_campaign": _campaign(),
        "governor_fidelity_mode": "exact",
    }
    result = {"status": "timeout", "duration_seconds": 30.0}

    observation = build_calibration_observation(
        scenario_root=tmp_path,
        scenario=scenario,
        result=result,
        baseline_entry=None,
    )

    assert observation is not None
    assert observation["outcome"]["scientific_value"] is True
    assert observation["uncertainty"]["censoring"] == "runtime_lower_bound"


def test_diagnostic_probe_is_excluded_from_calibration_observations(
    tmp_path: Path,
) -> None:
    observation = build_calibration_observation(
        scenario_root=tmp_path,
        scenario={"calibration_campaign": _campaign()},
        result={"status": "probe_completed", "duration_seconds": 3.0},
        baseline_entry=None,
    )

    assert observation is None


def test_run_scenario_emits_observation_before_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_artifacts = tmp_path / "reference" / "artifacts"
    reference_artifacts.mkdir(parents=True)
    reference_result = tmp_path / "reference" / "result.json"
    reference_result.write_text(json.dumps({"status": "success"}))
    scenario = {
        "name": "failed-calibration-row",
        "method": "exact",
        "calibration_campaign": _campaign(),
        "governor_fidelity_mode": "exact",
    }
    baseline_entry = {
        "registry_key": "provider/reference",
        "artifacts_dir": str(reference_artifacts),
        "result_json": str(reference_result),
        "expected_status": "success",
        "compact_graph_reference": {"format": "historical_mixed_compact_v1"},
    }

    def fail_to_start(*args: object, **kwargs: object) -> None:
        raise OSError("child process unavailable")

    monkeypatch.setattr(experiment_runner.subprocess, "run", fail_to_start)

    with pytest.raises(OSError, match="child process unavailable"):
        experiment_runner.run_scenario(
            tmp_path / "output",
            scenario,
            env={"PYTHONUNBUFFERED": "1"},
            run_metadata={
                "run_id": "run",
                "run_name": None,
                "run_description": None,
                "run_goal": None,
            },
            baseline_registry={"provider/reference": baseline_entry},
            baseline_registry_path=tmp_path / "registry.json",
        )

    scenario_root = tmp_path / "output" / "failed-calibration-row"
    result = json.loads((scenario_root / "result.json").read_text())
    observation = json.loads(
        (scenario_root / "calibration_observation.json").read_text()
    )
    assert result["status"] == "failed"
    assert result["execution_error"] == "OSError: child process unavailable"
    assert observation["outcome"]["status"] == "failed"
    assert observation["uncertainty"]["censoring"] == "infrastructure_or_unknown"
