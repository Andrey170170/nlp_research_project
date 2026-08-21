from __future__ import annotations

import json
from pathlib import Path

from circuit_tracer import AdmissionMode

from nlp_research_project.exact_trace_bench.scenarios.phase4_static_coalescing import (
    PHASE4_STATIC_COALESCING_BASELINE_KEY,
    build_phase4_static_coalescing_config,
)
from nlp_research_project.exact_trace_bench import phase4_static_validation
from nlp_research_project.exact_trace_bench.phase4_static_validation import (
    EXPECTED_CASES,
    validate_phase4_static_coalescing_run,
)
from nlp_research_project.exact_trace_bench.io_utils import write_json
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    trace_policy_from_scenario,
)


def test_static_coalescing_matrix_preserves_semantics_and_scales_execution() -> None:
    payload = build_phase4_static_coalescing_config()
    rows = payload["scenarios"]

    assert [row["validation_case"] for row in rows] == [
        "execution_b128",
        "execution_b256",
        "execution_b512",
    ]
    assert [
        (row["nnsight_session_capacity"], row["phase4_execution_batch_max_rows"])
        for row in rows
    ] == [(128, 128), (256, 256), (512, 512)]
    for row in rows:
        merged = {**payload["defaults"], **row}
        assert merged["feature_batch_size"] == 128
        assert merged["attribution_update_interval"] == 4
        assert merged["phase3_compute_microbatch_max_rows"] == 128
        assert "governor_fidelity_override_fields" not in row
        policy = trace_policy_from_scenario(merged)
        assert policy.governor_admission_mode is AdmissionMode.ADVISORY
        assert policy.execution.session.capacity == row["nnsight_session_capacity"]
        assert (
            policy.execution.session.phase4_execution_batch_max_rows
            == row["phase4_execution_batch_max_rows"]
        )
        assert policy.physical_requirements is not None
        assert (
            policy.physical_requirements.feature_microbatch_size
            == row["phase4_execution_batch_max_rows"]
        )


def test_static_coalescing_metadata_is_launch_ready() -> None:
    scratch_root = Path("/vast/exact-trace")
    payload = build_phase4_static_coalescing_config(scratch_root=scratch_root)
    metadata = payload["metadata"]

    assert metadata["matrix_case_count"] == 3
    assert metadata["fixed_semantics"] == {
        "feature_batch_size": 128,
        "frontier_refresh_stride": 4,
        "nominal_frontier_window_rows": 512,
    }
    assert metadata["execution_batch_ladder"] == [128, 256, 512]
    assert metadata["immutable_validation_config"] is True
    assert metadata["recommended_output_root"] == str(
        scratch_root
        / "granite"
        / "sweep"
        / "governor_calibration"
        / "phase4_static_coalescing"
        / "gemma3_1b_plt"
    )
    assert metadata["slurm"]["gres"] == "gpu:h200:1"
    assert metadata["slurm"]["time"] == "02:00:00"
    assert all(
        row["baseline_check"]["registry_key"] == PHASE4_STATIC_COALESCING_BASELINE_KEY
        for row in payload["scenarios"]
    )
    assert all(row["baseline_check"]["mode"] == "gate" for row in payload["scenarios"])


def _write_validation_case(
    run_root: Path,
    *,
    case: str,
    execution_groups: tuple[tuple[int, ...], ...],
    pending_hash: str = "pending-a",
    ranker_order_hash: str = "order-a",
) -> None:
    scenario_root = run_root / case
    telemetry_path = (
        scenario_root
        / "artifacts"
        / "prompt_000"
        / "completion_000"
        / "telemetry.live.jsonl"
    )
    telemetry_path.parent.mkdir(parents=True)
    write_json(scenario_root / "scenario.json", {"validation_case": case})
    write_json(
        scenario_root / "result.json",
        {
            "status": "success",
            "baseline_check": {"status": "gate_pass", "passed": True},
        },
    )
    records = [
        {
            "name": "phase4.refresh",
            "attrs": {
                "refresh_index": 0,
                "pending_hash": pending_hash,
                "ranker_frontier_selected_order_hash": ranker_order_hash,
                "ranker_frontier_selected_membership_hash": "members-a",
                "ranker_frontier_selected_count": 4,
            },
        }
    ]
    semantic_index = 0
    for execution_index, rows in enumerate(execution_groups, start=1):
        records.append(
            {
                "name": "phase4.feature_batch",
                "attrs": {
                    "phase4_semantic_batch_index_start": semantic_index,
                    "phase4_semantic_batch_rows": json.dumps(list(rows)),
                    "phase4_execution_batch_count": execution_index,
                    "scheduler_refresh_index": 0,
                },
            }
        )
        semantic_index += len(rows)
    telemetry_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _exact_graph_comparison(*args: object, **kwargs: object) -> dict[str, object]:
    return {
        "comparison_complete": True,
        "overall_mean_feature_jaccard": 1.0,
        "overall_mean_edge_jaccard": 1.0,
        "overall_mean_weighted_edge_jaccard": 0.9999999,
        "overall_mean_top256_edge_jaccard": 1.0,
    }


def test_static_coalescing_post_matrix_validator_accepts_physical_regrouping(
    monkeypatch, tmp_path: Path
) -> None:
    groups = {
        "execution_b128": ((128,), (128,), (128,), (128,)),
        "execution_b256": ((128, 128), (128, 128)),
        "execution_b512": ((128, 128, 128, 128),),
    }
    for case in EXPECTED_CASES:
        _write_validation_case(
            tmp_path,
            case=case,
            execution_groups=groups[case],
            ranker_order_hash=f"ranked-{case}",
        )
    monkeypatch.setattr(
        phase4_static_validation, "compare_artifact_dirs", _exact_graph_comparison
    )

    report = validate_phase4_static_coalescing_run(tmp_path)

    assert report["passed"] is True
    assert report["status"] == "pass"
    assert not report["failures"]
    assert all(
        comparison["ranker_pre_locality_order_hash_equal"] is False
        for comparison in report["comparisons"]
    )


def test_static_coalescing_post_matrix_validator_rejects_refresh_drift(
    monkeypatch, tmp_path: Path
) -> None:
    _write_validation_case(
        tmp_path, case="execution_b128", execution_groups=((128,), (128,))
    )
    _write_validation_case(
        tmp_path,
        case="execution_b256",
        execution_groups=((128, 128),),
        pending_hash="changed",
    )
    _write_validation_case(
        tmp_path, case="execution_b512", execution_groups=((128, 128),)
    )
    monkeypatch.setattr(
        phase4_static_validation, "compare_artifact_dirs", _exact_graph_comparison
    )

    report = validate_phase4_static_coalescing_run(tmp_path)

    assert report["passed"] is False
    assert any("semantic schedule or refresh identity" in row for row in report["failures"])
