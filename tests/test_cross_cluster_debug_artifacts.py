"""Lightweight checks for cross-cluster debug artifact helpers.

Usage::

    uv run python tests/test_cross_cluster_debug_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import inspect
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def run_checks() -> None:
    from trace_pipeline import extract_graph, trace_completion
    from trace_pipeline_chunked import (
        build_cross_cluster_debug_records,
        extract_compact_chunked_attribution,
        normalize_cross_cluster_debug_records,
        parse_phase4_scheduler_mode,
        parse_phase4_scheduler_telemetry_detail,
        trace_completion_compact_chunked,
    )

    payload = [
        {"checkpoint_name": "phase0_sparse_setup", "active_feature_count": 123},
        "skip-me",
        {"checkpoint_name": "phase1_target_logits", "target_count": 5},
    ]

    normalized = normalize_cross_cluster_debug_records(payload)
    assert len(normalized) == 2

    records = build_cross_cluster_debug_records(
        prompt_id="prompt_000",
        completion_id="completion_000",
        step_index=0,
        stream_name="cross_cluster_debug_checkpoints",
        records=normalized,
    )
    assert len(records) == 2
    assert records[0]["prompt_id"] == "prompt_000"
    assert records[0]["stream_name"] == "cross_cluster_debug_checkpoints"
    assert records[0]["record_index"] == 0
    assert records[1]["record_index"] == 1
    assert records[1]["checkpoint_name"] == "phase1_target_logits"

    extract_signature = inspect.signature(extract_compact_chunked_attribution)
    trace_signature = inspect.signature(trace_completion_compact_chunked)
    full_graph_extract_signature = inspect.signature(extract_graph)
    full_graph_trace_signature = inspect.signature(trace_completion)
    assert extract_signature.parameters["exact_trace_internal_dtype"].default == "fp32"
    assert extract_signature.parameters["phase4_scheduler_mode"].default == "locality"
    assert extract_signature.parameters["phase4_scheduler_debug"].default is False
    assert (
        extract_signature.parameters["phase4_scheduler_telemetry_detail"].default
        == "normal"
    )
    assert trace_signature.parameters["exact_trace_internal_dtype"].default == "fp32"
    assert trace_signature.parameters["phase4_scheduler_mode"].default == "locality"
    assert trace_signature.parameters["phase4_scheduler_debug"].default is False
    assert (
        trace_signature.parameters["phase4_scheduler_telemetry_detail"].default
        == "normal"
    )
    assert (
        full_graph_extract_signature.parameters["exact_trace_internal_dtype"].default
        == "fp32"
    )
    assert (
        full_graph_trace_signature.parameters["exact_trace_internal_dtype"].default
        == "fp32"
    )

    assert parse_phase4_scheduler_mode("locality") == "locality"
    assert parse_phase4_scheduler_mode("planner_v1") == "planner_v1"
    assert parse_phase4_scheduler_mode("planner_v2") == "planner_v2"
    assert parse_phase4_scheduler_mode("legacy") == "locality"

    assert parse_phase4_scheduler_telemetry_detail("summary") == "summary"
    assert parse_phase4_scheduler_telemetry_detail("normal") == "normal"
    assert parse_phase4_scheduler_telemetry_detail("debug") == "debug"
    assert parse_phase4_scheduler_telemetry_detail("compact") == "summary"
    assert parse_phase4_scheduler_telemetry_detail("full") == "debug"

    try:
        parse_phase4_scheduler_mode("unknown")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("Expected scheduler mode parser to reject unknown value")

    try:
        parse_phase4_scheduler_telemetry_detail("verbose")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError(
            "Expected scheduler telemetry detail parser to reject unknown value"
        )


def run_launcher_and_extractor_roundtrip_checks() -> None:
    from experiments.exact_trace_bench.extract import build_benchmark_index_row
    from experiments.extract_benchmark_index import build_row
    from experiments.run_sparsification_experiment import build_command

    scenario = {
        "method": "exact",
        "completions": 1,
        "temperature": 0.0,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_steps": 1,
        "attribution_batch_size": 128,
        "feature_batch_size": None,
        "logit_batch_size": None,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "attribution_update_interval": 4,
        "gsm8k_indices": [94],
        "decoder_chunk_size": 2048,
        "exact_trace_internal_dtype": "fp32",
        "cross_cluster_debug": True,
        "telemetry_max_events": 17,
        "phase4_anomaly_debug": False,
        "phase4_scheduler_mode": "planner_v2",
        "phase4_scheduler_debug": True,
        "phase4_scheduler_telemetry_detail": "debug",
    }

    command = build_command(Path("/tmp/out"), scenario)
    assert "--exact-trace-internal-dtype" in command
    assert "fp32" in command
    assert "--cross-cluster-debug" in command
    assert "--telemetry-max-events" in command
    assert "17" in command
    assert "--phase4-scheduler-mode" in command
    assert "planner_v2" in command
    assert "--phase4-scheduler-debug" in command
    assert "--phase4-scheduler-telemetry-detail" in command
    assert "debug" in command

    with tempfile.TemporaryDirectory() as tmp_dir:
        scenario_root = Path(tmp_dir) / "scenario"
        artifact_dir = scenario_root / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        scenario_payload = {
            **scenario,
            "name": "roundtrip",
            "stage": "fast",
            "save_raw": True,
        }
        run_config_payload = {
            "save_raw": True,
            "exact_trace_internal_dtype": "fp32",
            "exact_trace_internal_dtype_requested": "fp32",
            "exact_trace_internal_dtype_contract_supported": False,
            "cross_cluster_debug": False,
            "telemetry_max_events": 11,
            "phase4_scheduler_mode": "planner_v2",
            "phase4_scheduler_requested_mode": "planner_v2",
            "phase4_scheduler_effective_mode": "planner_v2",
            "phase4_scheduler_version": "planner_v2",
            "phase4_scheduler_effective_version": "planner_v2",
            "phase4_scheduler_policy": "bounded_membership_selection",
            "phase4_scheduler_effective_policy": "bounded_membership_selection",
            "phase4_scheduler_debug": True,
            "phase4_scheduler_telemetry_detail": "debug",
            "attribution_batch_size": 128,
            "decoder_chunk_size": 2048,
            "max_feature_nodes": 8192,
            "max_edges": 20000,
            "max_steps": 1,
        }
        result_payload = {
            "name": "roundtrip",
            "method": "exact",
            "status": "success",
            "returncode": 0,
            "duration_seconds": 1.0,
            "output_dir": str(artifact_dir),
        }

        (scenario_root / "scenario.json").write_text(
            json.dumps(scenario_payload, indent=2)
        )
        (scenario_root / "result.json").write_text(json.dumps(result_payload, indent=2))
        (artifact_dir / "run_config.json").write_text(
            json.dumps(run_config_payload, indent=2)
        )
        completion_dir = artifact_dir / "prompt_000" / "completion_000"
        completion_dir.mkdir(parents=True, exist_ok=True)
        completion_payload = {
            "prompt_id": "prompt_000",
            "completion_id": "completion_000",
            "phase4_scheduler_mode": "planner_v2",
            "phase4_scheduler_requested_mode": "planner_v2",
            "phase4_scheduler_mode_requested": "planner_v2",
            "phase4_scheduler_effective_mode": "planner_v2",
            "phase4_scheduler_mode_effective": "planner_v2",
            "phase4_scheduler_version": "planner_v2",
            "phase4_scheduler_effective_version": "planner_v2",
            "phase4_scheduler_version_effective": "planner_v2",
            "phase4_scheduler_policy": "bounded_membership_selection",
            "phase4_scheduler_effective_policy": "bounded_membership_selection",
            "phase4_scheduler_policy_effective": "bounded_membership_selection",
            "phase4_scheduler_debug_effective": True,
            "phase4_scheduler_telemetry_detail_effective": "debug",
            "steps": [],
        }
        (completion_dir / "completion.json").write_text(
            json.dumps(completion_payload, indent=2)
        )

        benchmark_row = build_benchmark_index_row(scenario_root / "result.json")
        legacy_row = build_row(scenario_root / "result.json")

        assert benchmark_row["exact_trace_internal_dtype"] == "fp32"
        assert benchmark_row["exact_trace_internal_dtype_contract_supported"] is False
        assert benchmark_row["telemetry_max_events"] == 11
        assert benchmark_row["phase4_scheduler_mode"] == "planner_v2"
        assert benchmark_row["phase4_scheduler_mode_requested"] == "planner_v2"
        assert benchmark_row["phase4_scheduler_mode_effective"] == "planner_v2"
        assert benchmark_row["phase4_scheduler_version"] == "planner_v2"
        assert benchmark_row["phase4_scheduler_version_requested"] == "planner_v2"
        assert benchmark_row["phase4_scheduler_version_effective"] == "planner_v2"
        assert (
            benchmark_row["phase4_scheduler_policy"] == "bounded_membership_selection"
        )
        assert (
            benchmark_row["phase4_scheduler_policy_requested"]
            == "bounded_membership_selection"
        )
        assert (
            benchmark_row["phase4_scheduler_policy_effective"]
            == "bounded_membership_selection"
        )
        assert (
            benchmark_row["phase4_scheduler_effective_policy"]
            == "bounded_membership_selection"
        )
        assert benchmark_row["phase4_scheduler_debug"] is True
        assert benchmark_row["phase4_scheduler_telemetry_detail"] == "debug"

        assert legacy_row["exact_trace_internal_dtype"] == "fp32"
        assert legacy_row["exact_trace_internal_dtype_contract_supported"] is False
        assert legacy_row["telemetry_max_events"] == 11
        assert legacy_row["phase4_scheduler_mode"] == "planner_v2"
        assert legacy_row["phase4_scheduler_mode_requested"] == "planner_v2"
        assert legacy_row["phase4_scheduler_mode_effective"] == "planner_v2"
        assert legacy_row["phase4_scheduler_version"] == "planner_v2"
        assert legacy_row["phase4_scheduler_version_requested"] == "planner_v2"
        assert legacy_row["phase4_scheduler_version_effective"] == "planner_v2"
        assert legacy_row["phase4_scheduler_policy"] == "bounded_membership_selection"
        assert (
            legacy_row["phase4_scheduler_policy_requested"]
            == "bounded_membership_selection"
        )
        assert (
            legacy_row["phase4_scheduler_policy_effective"]
            == "bounded_membership_selection"
        )
        assert (
            legacy_row["phase4_scheduler_effective_policy"]
            == "bounded_membership_selection"
        )
        assert legacy_row["phase4_scheduler_debug"] is True
        assert legacy_row["phase4_scheduler_telemetry_detail"] == "debug"


def main() -> None:
    run_checks()
    run_launcher_and_extractor_roundtrip_checks()
    print("OK: cross-cluster debug helper and round-trip checks passed")


if __name__ == "__main__":
    main()
