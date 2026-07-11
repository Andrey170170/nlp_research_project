from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


PHASE_D_KEYS = (
    "nnsight_session_capacity",
    "phase3_compute_microbatch_max_rows",
    "phase4_compute_microbatch_max_rows",
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


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


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
    assert metadata["slurm"]["time"] == "02:00:00"
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
    for scenario in (tiled, recompute):
        assert scenario["phase3_gradient_replay_mode"] == "disabled"
        assert scenario["phase3_row_replay_mode"] == "disabled"
        assert scenario["capture_phase3_gradient_bundle"] is False
        assert scenario["capture_phase3_row_bundle"] is False

    command = build_command(Path("/tmp/phase-d-validation"), recompute)
    for key in PHASE_D_KEYS:
        if recompute[key] is not None:
            assert _flag(key) in command


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
