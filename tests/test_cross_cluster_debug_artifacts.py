"""Lightweight checks for cross-cluster debug artifact helpers.

Usage::

    uv run python tests/test_cross_cluster_debug_artifacts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_checks() -> None:
    from trace_pipeline_chunked import (
        build_cross_cluster_debug_records,
        normalize_cross_cluster_debug_records,
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
    }

    command = build_command(Path("/tmp/out"), scenario)
    assert "--exact-trace-internal-dtype" in command
    assert "fp32" in command
    assert "--cross-cluster-debug" in command
    assert "--telemetry-max-events" in command
    assert "17" in command

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
            "exact_trace_internal_dtype": "fp64",
            "exact_trace_internal_dtype_requested": "fp64",
            "exact_trace_internal_dtype_contract_supported": False,
            "cross_cluster_debug": False,
            "telemetry_max_events": 11,
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

        benchmark_row = build_benchmark_index_row(scenario_root / "result.json")
        legacy_row = build_row(scenario_root / "result.json")

        assert benchmark_row["exact_trace_internal_dtype"] == "fp64"
        assert benchmark_row["exact_trace_internal_dtype_contract_supported"] is False
        assert benchmark_row["telemetry_max_events"] == 11

        assert legacy_row["exact_trace_internal_dtype"] == "fp64"
        assert legacy_row["exact_trace_internal_dtype_contract_supported"] is False
        assert legacy_row["telemetry_max_events"] == 17


def main() -> None:
    run_checks()
    run_launcher_and_extractor_roundtrip_checks()
    print("OK: cross-cluster debug helper and round-trip checks passed")


if __name__ == "__main__":
    main()
