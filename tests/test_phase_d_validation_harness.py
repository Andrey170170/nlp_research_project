from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_sparsification_experiment as experiment  # noqa: E402
from experiments.run_sparsification_experiment import build_command, load_scenarios  # noqa: E402
from nlp_research_project.exact_trace_bench.baselines import (  # noqa: E402
    SCENARIO_IDENTITY_KEYS,
    SCENARIO_KNOB_KEYS,
)
from nlp_research_project.exact_trace_bench.config import (  # noqa: E402
    REPO_ROOT,
    base_trace_defaults,
)
from nlp_research_project.exact_trace_bench.extract import (  # noqa: E402
    build_benchmark_index_row,
)
from nlp_research_project.exact_trace_bench.scenarios import (  # noqa: E402
    ADVANCED_PUBLIC_TUNING_KEYS,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (  # noqa: E402
    trace_policy_from_scenario,
)


PHASE_D_KEYS = (
    "nnsight_session_capacity",
    "phase3_compute_microbatch_max_rows",
    "phase4_execution_batch_max_rows",
    "full_retention_backend",
    "feature_row_column_tile_size",
    "influence_row_tile_size",
    "influence_column_tile_size",
    "feature_row_retention",
    "replay_tile_cache_bytes",
)
CONFIGS = tuple(
    REPO_ROOT / "experiments/generated/exact_trace_bench" / name
    for name in (
        "exact_trace_phase_d_validation_gemma3_1b_clt_granite_scenarios.json",
        "exact_trace_phase_d_validation_gemma3_1b_plt_granite_scenarios.json",
    )
)


def test_phase_d_knobs_have_stable_defaults_and_schema_classification() -> None:
    defaults = base_trace_defaults()
    assert set(PHASE_D_KEYS).issubset(defaults)
    assert set(PHASE_D_KEYS).issubset(ADVANCED_PUBLIC_TUNING_KEYS)
    assert set(PHASE_D_KEYS).issubset(SCENARIO_KNOB_KEYS)
    assert {"validation_baseline_key", "validation_mechanism"}.issubset(
        SCENARIO_IDENTITY_KEYS
    )
    assert defaults["nnsight_session_capacity"] is None
    assert defaults["full_retention_backend"] == "full_file"
    assert defaults["feature_row_retention"] == "full_file"


@pytest.mark.parametrize("config_path", CONFIGS)
def test_phase_d_validation_configs_and_forwarding(config_path: Path) -> None:
    scenarios, metadata = load_scenarios(config_path)
    assert metadata["immutable_validation_config"] is True
    assert metadata["slurm"]["gres"] == "gpu:h200:1"
    assert metadata["slurm"]["mem"] == "100G"
    expected_time = "07:00:00" if "plt" in metadata["baseline_variant"] else "02:00:00"
    assert metadata["slurm"]["time"] == expected_time
    assert len(scenarios) == 5
    assert len({scenario["name"] for scenario in scenarios}) == 5
    assert len({scenario["recommended_output_root"] for scenario in scenarios}) == 5

    legacy, explicit, reduced, tiled, recompute = scenarios
    assert not set(PHASE_D_KEYS).intersection(legacy["_explicit_scenario_keys"])
    assert explicit["nnsight_session_capacity"] == explicit["attribution_batch_size"]
    expected_reduced = 128 if "clt" in metadata["baseline_variant"] else 64
    assert reduced["nnsight_session_capacity"] == expected_reduced
    assert tiled["full_retention_backend"] == "column_tiled_v1"
    assert tiled["row_store_preallocate"] is False
    assert recompute["feature_row_retention"] == "none_recompute"
    if "plt" in metadata["baseline_variant"]:
        assert tiled["feature_row_column_tile_size"] == 16384
        assert recompute["feature_row_column_tile_size"] == 16384
        assert recompute["replay_tile_cache_bytes"] == 4 * 1024**3
        assert tiled["timeout_minutes"] == 300
        assert recompute["timeout_minutes"] == 420
    for scenario in (tiled, recompute):
        assert scenario["phase3_gradient_replay_mode"] == "disabled"
        assert scenario["phase3_row_replay_mode"] == "disabled"
        assert scenario["capture_phase3_gradient_bundle"] is False
        assert scenario["capture_phase3_row_bundle"] is False

    command = build_command(Path("/tmp/phase-d-validation"), recompute)
    assert command[1:3] == ["-m", "nlp_research_project.exact_trace_bench.trace_runtime"]
    assert command[-4:] == [
        "--scenario-file",
        "/tmp/scenario.json",
        "--output-dir",
        "/tmp/phase-d-validation",
    ]
    policy = trace_policy_from_scenario(recompute)
    assert policy.execution.session.capacity == recompute["nnsight_session_capacity"]
    assert policy.execution.storage.retention == recompute["feature_row_retention"]
    assert policy.execution.storage.replay_tile_cache_bytes == recompute["replay_tile_cache_bytes"]


@pytest.mark.parametrize("config_path", CONFIGS)
def test_phase_d_column_tiled_scenario_builds_typed_storage_policy(
    config_path: Path,
) -> None:
    scenarios, _metadata = load_scenarios(config_path)
    tiled = scenarios[3]
    policy = trace_policy_from_scenario(tiled)
    assert policy.execution.storage.preallocate is False
    assert policy.execution.storage.full_retention_backend == "column_tiled_v1"


def test_phase_d_fields_are_extracted_for_comparison(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenario"
    artifact_root = scenario_root / "artifacts"
    artifact_root.mkdir(parents=True)
    scenario = {
        "name": "phase_d_extract",
        "cluster": "granite",
        "validation_baseline_key": "phase_d/reference/A",
        "validation_mechanism": "D_reduced_session_column_tiled_v1",
    }
    (scenario_root / "scenario.json").write_text(json.dumps(scenario))
    (scenario_root / "result.json").write_text(
        json.dumps({"status": "success", "output_dir": str(artifact_root)})
    )
    run_config = {key: index + 1 for index, key in enumerate(PHASE_D_KEYS)}
    run_config["full_retention_backend"] = "column_tiled_v1"
    run_config["feature_row_retention"] = "full_file"
    (artifact_root / "run_config.json").write_text(json.dumps(run_config))

    row = build_benchmark_index_row(scenario_root / "result.json")
    for key, value in run_config.items():
        assert row[key] == value
    assert row["validation_baseline_key"] == "phase_d/reference/A"
    assert row["validation_mechanism"] == "D_reduced_session_column_tiled_v1"


def test_extractor_uses_legacy_phase4_rows_when_canonical_default_is_none(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    artifact_root = scenario_root / "artifacts"
    artifact_root.mkdir(parents=True)
    (scenario_root / "scenario.json").write_text(
        json.dumps(
            {
                "name": "legacy_phase4_extract",
                "phase4_execution_batch_max_rows": None,
                "phase4_compute_microbatch_max_rows": 17,
            }
        )
    )
    (scenario_root / "result.json").write_text(
        json.dumps({"status": "success", "output_dir": str(artifact_root)})
    )
    (artifact_root / "run_config.json").write_text(
        json.dumps(
            {
                "phase4_execution_batch_max_rows": None,
                "phase4_compute_microbatch_max_rows": 23,
            }
        )
    )

    row = build_benchmark_index_row(scenario_root / "result.json")

    assert row["phase4_execution_batch_max_rows"] == 23


def test_phase_d_controls_flow_into_owned_canonical_policies() -> None:
    controls = {
        "nnsight_session_capacity": 13,
        "phase3_compute_microbatch_max_rows": 12,
        "phase4_execution_batch_max_rows": 13,
        "full_retention_backend": "column_tiled_v1",
        "feature_row_column_tile_size": 14,
        "influence_row_tile_size": 15,
        "influence_column_tile_size": 16,
        "feature_row_retention": "none_recompute",
        "replay_tile_cache_bytes": 17,
    }
    policy = trace_policy_from_scenario({"method": "exact", **controls})
    assert policy.execution.session.capacity == 13
    assert policy.execution.session.phase3_microbatch_max_rows == 12
    assert policy.execution.session.phase4_execution_batch_max_rows == 13
    assert policy.execution.storage.full_retention_backend == "column_tiled_v1"
    assert policy.execution.storage.feature_column_tile_size == 14
    assert policy.execution.storage.influence_row_tile_size == 15
    assert policy.execution.storage.influence_column_tile_size == 16
    assert policy.execution.storage.retention == "none_recompute"
    assert policy.execution.storage.replay_tile_cache_bytes == 17


@pytest.mark.parametrize("status", ["failed", "timeout", "baseline_invalid"])
def test_experiment_main_writes_summary_then_exits_nonzero_for_failed_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps({"scenarios": [{"name": "phase_d", "method": "exact_chunked"}]})
    )
    output_root = tmp_path / "results"

    def fake_run_scenario(
        selected_output_root: Path, scenario: dict[str, object], **_kwargs: object
    ) -> dict[str, object]:
        artifact_dir = selected_output_root / "artifacts"
        artifact_dir.mkdir(parents=True)
        return {
            "name": scenario["name"],
            "status": status,
            "duration_seconds": 0.01,
            "output_dir": str(artifact_dir),
        }

    monkeypatch.setattr(experiment, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sparsification_experiment.py",
            "--scenarios-file",
            str(scenarios_path),
            "--output-root",
            str(output_root),
        ],
    )
    with pytest.raises(SystemExit, match=rf"phase_d \({status}\)"):
        experiment.main()

    summary = json.loads((output_root / "phase_d" / "summary.json").read_text())
    assert summary["results"][0]["status"] == status
