"""Lightweight checks for cross-cluster debug artifact helpers.

Usage::

    uv run python tests/test_cross_cluster_debug_artifacts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def run_checks() -> None:
    from trace_pipeline_chunked import (
        build_cross_cluster_debug_records,
        normalize_cross_cluster_debug_records,
    )

    payload = [
        {
            "checkpoint_name": "phase0_sparse_setup",
            "active_feature_count": 123,
            "phase0_pre_clt_input_global_hash": "abcd1234",
            "phase0_boundary_fingerprints": {
                "global_hashes": {
                    "pre_activation_hash_global": "beef5678",
                }
            },
        },
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
    assert records[0]["phase0_pre_clt_input_global_hash"] == "abcd1234"
    assert (
        records[0]["phase0_boundary_fingerprints"]["global_hashes"][
            "pre_activation_hash_global"
        ]
        == "beef5678"
    )
    assert records[1]["record_index"] == 1
    assert records[1]["checkpoint_name"] == "phase1_target_logits"


def run_launcher_and_extractor_roundtrip_checks() -> None:
    from experiments.exact_trace_bench.scenarios import _select_exact_mode_knobs
    from experiments.exact_trace_bench.extract import build_benchmark_index_row
    from experiments.extract_benchmark_index import build_row
    from experiments.run_sparsification_experiment import build_command

    selected_knobs = _select_exact_mode_knobs(
        {
            "capture_phase0_donor_bundle": True,
            "phase0_donor_bundle": "/tmp/donor_bundle.npz",
            "phase0_replay_mode": "donor_phase0",
            "phase0_donor_context_policy": "warn",
            "capture_phase3_seed_bundle": True,
        }
    )
    assert selected_knobs["capture_phase0_donor_bundle"] is True
    assert selected_knobs["phase0_donor_bundle"] == "/tmp/donor_bundle.npz"
    assert selected_knobs["phase0_replay_mode"] == "donor_phase0"
    assert selected_knobs["phase0_donor_context_policy"] == "warn"

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
        "phase0_activation_threshold_compare_mode": "fp32",
        "cross_cluster_debug": True,
        "capture_phase0_donor_bundle": True,
        "phase0_donor_bundle": "/tmp/donor_bundle.npz",
        "phase0_replay_mode": "donor_phase0",
        "phase0_donor_context_policy": "warn",
        "capture_phase3_seed_bundle": True,
        "telemetry_max_events": 17,
        "phase4_anomaly_debug": False,
    }

    command = build_command(Path("/tmp/out"), scenario)
    assert "--exact-trace-internal-dtype" in command
    assert "fp32" in command
    assert "--cross-cluster-debug" in command
    assert "--telemetry-max-events" in command
    assert "17" in command
    assert "--phase0-donor-bundle" in command
    assert "/tmp/donor_bundle.npz" in command
    assert "--phase0-replay-mode" in command
    assert "donor_phase0" in command
    assert "--phase0-donor-context-policy" in command
    assert "warn" in command
    assert "--phase0-activation-threshold-compare-mode" in command
    assert "fp32" in command
    assert "--capture-phase0-donor-bundle" in command
    assert "--capture-phase3-seed-bundle" in command

    old_patch_command = build_command(
        Path("/tmp/out"),
        {
            **scenario,
            "method": "old_patch",
        },
    )
    assert "trace_pipeline.py" in old_patch_command[1]
    assert "--phase0-activation-threshold-compare-mode" not in old_patch_command
    assert "--capture-phase0-donor-bundle" not in old_patch_command
    assert "--phase0-donor-bundle" not in old_patch_command
    assert "--phase0-replay-mode" not in old_patch_command
    assert "--phase0-donor-context-policy" not in old_patch_command
    assert "--capture-phase3-seed-bundle" not in old_patch_command

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
            "phase0_activation_threshold_compare_mode": "bf16",
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

        prompt_dir = artifact_dir / "prompt_000"
        completion_dir = prompt_dir / "completion_000"
        completion_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt_meta.json").write_text(
            json.dumps(
                {
                    "prompt_source": "gsm8k",
                    "prompt_token_count": 12,
                    "initial_input_token_count": 12,
                },
                indent=2,
            )
        )
        seed_bundle_name = "step_000_phase3_seed_bundle.npz"
        donor_bundle_name = "step_000_phase0_donor_bundle.npz"
        (completion_dir / seed_bundle_name).write_bytes(b"placeholder")
        (completion_dir / donor_bundle_name).write_bytes(b"placeholder")
        (completion_dir / "completion.json").write_text(
            json.dumps(
                {
                    "prompt_source": "gsm8k",
                    "prompt_token_count": 12,
                    "initial_input_token_count": 12,
                    "generated_token_count": 1,
                    "n_steps_traced": 1,
                    "phase0_donor_bundle_capture_enabled": True,
                    "phase0_donor_bundle_status": "captured",
                    "phase0_donor_bundle_statuses_observed": ["captured"],
                    "phase0_replay_mode": "donor_phase0",
                    "phase0_replay_status": "replayed",
                    "phase0_replay_statuses_observed": ["replayed"],
                    "phase0_donor_context_policy": "warn",
                    "phase0_donor_bundle": "/tmp/donor_bundle.npz",
                    "phase0_replay_validation_warning_count": 2,
                    "phase0_replay_validation_warning_count_max": 2,
                    "phase0_replay_dtype_roundtrip_loss": True,
                    "phase0_replay_any_dtype_roundtrip_loss": True,
                    "phase3_seed_bundle_capture_enabled": True,
                    "phase3_seed_bundle_status": "captured",
                    "phase3_seed_bundle_statuses_observed": ["captured"],
                    "steps": [
                        {
                            "phase0_donor_bundle_capture_enabled": True,
                            "phase0_donor_bundle_path": donor_bundle_name,
                            "phase0_donor_bundle_status": "captured",
                            "phase0_replay_mode": "donor_phase0",
                            "phase0_replay_status": "replayed",
                            "phase0_replay_donor_context_policy": "warn",
                            "phase0_replay_donor_bundle_path": "/tmp/donor_bundle.npz",
                            "phase0_replay_validation_warning_count": 2,
                            "phase0_replay_dtype_roundtrip_loss": True,
                            "phase3_seed_bundle_capture_enabled": True,
                            "phase3_seed_bundle_path": seed_bundle_name,
                            "phase3_seed_bundle_status": "captured",
                        }
                    ],
                },
                indent=2,
            )
        )

        benchmark_row = build_benchmark_index_row(scenario_root / "result.json")
        legacy_row = build_row(scenario_root / "result.json")

        assert benchmark_row["exact_trace_internal_dtype"] == "fp64"
        assert benchmark_row["exact_trace_internal_dtype_contract_supported"] is False
        assert benchmark_row["telemetry_max_events"] == 11
        assert benchmark_row["phase0_activation_threshold_compare_mode"] == "bf16"

        assert legacy_row["exact_trace_internal_dtype"] == "fp64"
        assert legacy_row["exact_trace_internal_dtype_contract_supported"] is False
        assert legacy_row["telemetry_max_events"] == 11
        assert legacy_row["phase0_activation_threshold_compare_mode"] == "bf16"

        assert benchmark_row["phase3_seed_bundle_present"] is True
        assert benchmark_row["phase3_seed_bundle_file_count"] == 1
        assert benchmark_row["phase3_seed_bundle_status"] == "captured"
        assert benchmark_row["phase0_donor_bundle_present"] is True
        assert benchmark_row["phase0_donor_bundle_file_count"] == 1
        assert benchmark_row["phase0_donor_bundle_status"] == "captured"
        assert benchmark_row["phase0_replay_mode"] == "donor_phase0"
        assert benchmark_row["phase0_replay_status"] == "replayed"
        assert benchmark_row["phase0_replay_donor_context_policy"] == "warn"
        assert (
            benchmark_row["phase0_replay_donor_bundle_path"] == "/tmp/donor_bundle.npz"
        )
        assert benchmark_row["phase0_replay_validation_warning_count"] == 2
        assert benchmark_row["phase0_replay_validation_warning_count_max"] == 2
        assert benchmark_row["phase0_replay_dtype_roundtrip_loss"] is True
        assert benchmark_row["phase0_replay_any_dtype_roundtrip_loss"] is True

        assert legacy_row["phase3_seed_bundle_present"] is True
        assert legacy_row["phase3_seed_bundle_file_count"] == 1
        assert legacy_row["phase3_seed_bundle_status"] == "captured"
        assert legacy_row["phase0_donor_bundle_present"] is True
        assert legacy_row["phase0_donor_bundle_file_count"] == 1
        assert legacy_row["phase0_donor_bundle_status"] == "captured"
        assert legacy_row["phase0_replay_mode"] == "donor_phase0"
        assert legacy_row["phase0_replay_status"] == "replayed"
        assert legacy_row["phase0_replay_donor_context_policy"] == "warn"
        assert legacy_row["phase0_replay_donor_bundle_path"] == "/tmp/donor_bundle.npz"
        assert legacy_row["phase0_replay_validation_warning_count"] == 2
        assert legacy_row["phase0_replay_validation_warning_count_max"] == 2
        assert legacy_row["phase0_replay_dtype_roundtrip_loss"] is True
        assert legacy_row["phase0_replay_any_dtype_roundtrip_loss"] is True


def run_feature_semantic_descriptor_save_checks() -> None:
    import torch

    from trace_pipeline_chunked import save_feature_semantic_descriptors

    payload = {
        "status": "captured",
        "descriptor_version": "v1",
        "descriptor_kind": "fallback_identity_metadata_v1",
        "descriptor_dim": 4,
        "semantic_descriptor_top_k": 16,
        "candidate_count": 2,
        "total_active_features": 5,
        "candidate_features": torch.tensor([[0, 0, 7], [1, 2, 3]], dtype=torch.int64),
        "candidate_row_indices": torch.tensor([1, 4], dtype=torch.int64),
        "activation_value": torch.tensor([0.2, -0.7], dtype=torch.float32),
        "seed_influence": torch.tensor([0.9, 0.1], dtype=torch.float64),
        "seed_rank": torch.tensor([0, 4], dtype=torch.int64),
        "is_top_seed": torch.tensor([True, False], dtype=torch.bool),
        "is_frontier_pre": torch.tensor([True, True], dtype=torch.bool),
        "frontier_pre_rank": torch.tensor([1, 3], dtype=torch.int64),
        "is_frontier_post": torch.tensor([False, True], dtype=torch.bool),
        "frontier_post_rank": torch.tensor([-1, 2], dtype=torch.int64),
        "is_selected_phase4": torch.tensor([True, False], dtype=torch.bool),
        "phase4_selected_rank": torch.tensor([0, -1], dtype=torch.int64),
        "phase4_selection_available": True,
        "seed_influence_available": True,
        "semantic_sketch": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, -0.2, 0.7, -0.8]],
            dtype=torch.float32,
        ),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "step_000_feature_semantic_descriptors.npz"
        save_feature_semantic_descriptors(payload, path)
        loaded = np.load(path, allow_pickle=False)

        assert loaded["candidate_features"].shape == (2, 3)
        assert loaded["candidate_row_indices"].shape == (2,)
        assert loaded["semantic_sketch"].shape == (2, 4)
        assert loaded["status"].item() == "captured"
        assert loaded["descriptor_version"].item() == "v1"
        assert loaded["descriptor_kind"].item() == "fallback_identity_metadata_v1"
        assert int(loaded["descriptor_dim"].item()) == 4
        assert int(loaded["candidate_count"].item()) == 2
        assert bool(loaded["phase4_selection_available"].item()) is True
        assert bool(loaded["seed_influence_available"].item()) is True


def run_phase3_seed_bundle_save_checks() -> None:
    import torch

    from trace_pipeline_chunked import save_phase3_seed_bundle

    payload = {
        "status": "captured",
        "active_features": torch.tensor([[0, 0, 7], [1, 2, 3]], dtype=torch.int64),
        # Runtime activation_matrix.values() can be bfloat16, which NumPy cannot
        # convert directly from a torch tensor.
        "activation_values": torch.tensor([0.25, 1.5], dtype=torch.bfloat16),
        "seed_feature_influences": torch.tensor([0.9, 0.1], dtype=torch.float64),
        "frontier_pre_locality": torch.tensor([1, 0], dtype=torch.int64),
        "frontier_post_locality": torch.tensor([0, 1], dtype=torch.int64),
        "queue_size": 2,
        "actual_max_feature_nodes": 4,
        "total_active_features": 2,
        "planner_compute_dtype": "float64",
        "influence_compute_dtype": "float64",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "step_000_phase3_seed_bundle.npz"
        save_phase3_seed_bundle(payload, path)
        loaded = np.load(path, allow_pickle=False)

        assert loaded["active_features"].shape == (2, 3)
        assert loaded["activation_values"].dtype == np.float32
        assert loaded["seed_feature_influences"].dtype == np.float64
        assert loaded["frontier_pre_locality"].tolist() == [1, 0]
        assert loaded["frontier_post_locality"].tolist() == [0, 1]
        assert loaded["status"].item() == "captured"
        assert loaded["planner_compute_dtype"].item() == "float64"


def run_phase0_donor_bundle_save_checks() -> None:
    import torch

    from trace_pipeline_chunked import save_phase0_donor_bundle

    activation_values = torch.tensor([0.25, -1.5], dtype=torch.bfloat16)
    payload = {
        "status": "captured",
        "schema_version": 1,
        "replay_kind": "phase0_active_features_v1",
        "replayed_effective_state": True,
        "phase0_replay_mode": "donor_phase0",
        "active_features": torch.tensor([[0, 0, 7], [1, 2, 3]], dtype=torch.int64),
        "activation_values": activation_values,
        "activation_values_dtype": "bfloat16",
        "activation_matrix_shape": [2, 3, 16],
        "active_feature_count": 2,
        "active_feature_membership_hash_raw_order": "rawhash",
        "active_feature_membership_hash_canonical": "canonicalhash",
        "active_feature_values_hash": "valuehash",
        "active_feature_layer_counts": torch.tensor([1, 1], dtype=torch.int64),
        "input_tokens": torch.tensor([11, 22, 33], dtype=torch.int64),
        "input_token_count": 3,
        "input_tokens_hash": "inputhash",
        "target_token_ids": torch.tensor([9, 10], dtype=torch.int64),
        "target_count": 2,
        "target_token_ids_hash": "targethash",
        "target_probabilities": torch.tensor([0.7, 0.2], dtype=torch.float32),
        "target_probability_hash": "probhash",
        "target_logits": torch.tensor([3.5, -1.2], dtype=torch.float32),
        "target_logit_hash": "logithash",
        "clt_constants_hash": "clthash",
        "provenance": {"source": "unit-test"},
        "prompt_metadata": {"prompt_source": "fixture"},
        "target_metadata": {"target_selection": "salient"},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "step_000_phase0_donor_bundle.npz"
        save_phase0_donor_bundle(payload, path)
        loaded = np.load(path, allow_pickle=False)

        expected_raw = activation_values.view(torch.uint16).cpu().numpy()

        assert loaded["active_features"].shape == (2, 3)
        assert loaded["activation_values"].dtype == np.float32
        assert loaded["activation_values_dtype"].item() == "bfloat16"
        assert loaded["activation_values_raw_uint16"].dtype == np.uint16
        assert loaded["activation_values_raw_uint16"].tolist() == expected_raw.tolist()
        assert loaded["active_feature_membership_hash_raw_order"].item() == "rawhash"
        assert (
            loaded["active_feature_membership_hash_canonical"].item() == "canonicalhash"
        )
        assert loaded["active_feature_values_hash"].item() == "valuehash"
        assert loaded["target_token_ids"].tolist() == [9, 10]
        assert loaded["schema_version"].item() == 1
        assert loaded["replay_kind"].item() == "phase0_active_features_v1"
        assert bool(loaded["replayed_effective_state"].item()) is True
        assert loaded["phase0_replay_mode"].item() == "donor_phase0"
        assert loaded["status"].item() == "captured"


def main() -> None:
    run_checks()
    run_launcher_and_extractor_roundtrip_checks()
    run_phase0_donor_bundle_save_checks()
    run_phase3_seed_bundle_save_checks()
    run_feature_semantic_descriptor_save_checks()
    print("OK: cross-cluster debug helper and round-trip checks passed")


if __name__ == "__main__":
    main()
