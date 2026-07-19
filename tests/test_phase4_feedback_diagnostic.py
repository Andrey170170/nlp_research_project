from __future__ import annotations

from pathlib import Path

from circuit_tracer import AdmissionMode

from nlp_research_project.exact_trace_bench.scenarios.phase4_feedback_diagnostic import (
    PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_KEY,
    PHASE4_FEEDBACK_DIAGNOSTIC_RESOURCE_PROFILE,
    build_phase4_feedback_diagnostic_config,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    trace_policy_from_scenario,
)


EXPECTED_CASES = [
    "canonical_reference",
    "constant_window_logical_b256_session128_physical_b128",
    "constant_window_logical_b256_physical_b128",
    "constant_window_logical_b256_physical_b256",
    "stale_window_logical_b256_physical_b128",
]


def test_phase4_feedback_diagnostic_rows_are_bounded_and_ordered() -> None:
    payload = build_phase4_feedback_diagnostic_config()
    rows = payload["scenarios"]

    assert [row["diagnostic_case"] for row in rows] == EXPECTED_CASES
    assert [
        (
            row["feature_batch_size"],
            row["attribution_update_interval"],
            row["nnsight_session_capacity"],
            row["phase4_compute_microbatch_max_rows"],
        )
        for row in rows
    ] == [
        (128, 4, 128, 128),
        (256, 2, 128, 128),
        (256, 2, 256, 128),
        (256, 2, 256, 256),
        (256, 4, 256, 128),
    ]

    for row in rows:
        assert row["attribution_batch_size"] == 128
        assert row["logit_batch_size"] == 128
        assert row["phase1_trace_batch_policy"] == "cap_effective_batches"
        assert row["phase1_trace_batch_size_max"] == 128
        assert row["phase3_compute_microbatch_max_rows"] == 128
        assert row["decoder_chunk_size"] == 4096
        assert row["cross_batch_decoder_cache_bytes"] == 0
        assert row["timeout_minutes"] == 120
        assert row["governor_resource_envelope"]["walltime_seconds"] == 120 * 60
        assert row["nnsight_session_capacity"] <= row["feature_batch_size"]

    assert [
        row["feature_batch_size"] * row["attribution_update_interval"]
        for row in rows
    ] == [512, 512, 512, 512, 1024]


def test_phase4_feedback_diagnostic_declares_fidelity_and_baselines() -> None:
    payload = build_phase4_feedback_diagnostic_config()
    rows = payload["scenarios"]

    assert [row["governor_fidelity_mode"] for row in rows] == [
        "strict",
        "research",
        "research",
        "research",
        "research",
    ]
    assert "governor_fidelity_override_fields" not in rows[0]
    assert rows[1]["governor_fidelity_override_fields"] == [
        "feature_batch_size",
        "frontier_refresh_stride",
    ]
    assert rows[2]["governor_fidelity_override_fields"] == [
        "feature_batch_size",
        "frontier_refresh_stride",
    ]
    assert rows[3]["governor_fidelity_override_fields"] == [
        "feature_batch_size",
        "frontier_refresh_stride",
    ]
    assert rows[4]["governor_fidelity_override_fields"] == ["feature_batch_size"]

    expected_baseline_check = {
        "enabled": True,
        "mode": "metrics",
        "registry_key": PHASE4_FEEDBACK_DIAGNOSTIC_BASELINE_KEY,
        "baseline_required": True,
    }
    assert all(row["baseline_check"] == expected_baseline_check for row in rows)
    for row in rows:
        policy = trace_policy_from_scenario({**payload["defaults"], **row})
        assert policy.governor_admission_mode is AdmissionMode.ADVISORY
        assert policy.governor_fidelity.mode.value == row["governor_fidelity_mode"]


def test_phase4_feedback_diagnostic_metadata_and_resources() -> None:
    scratch_root = Path("/vast/exact-trace")
    payload = build_phase4_feedback_diagnostic_config(scratch_root=scratch_root)
    defaults = payload["defaults"]
    metadata = payload["metadata"]

    assert defaults["governor_profile_name"] == (
        "granite_h200_1b_plt_b128_c4096_cache0"
    )
    assert defaults["governor_admission_mode"] == "advisory"
    assert defaults["exact_trace_internal_dtype"] == "fp32"
    assert defaults["attribution_batch_size"] == 128
    assert defaults["feature_batch_size"] == 128
    assert defaults["logit_batch_size"] == 128
    assert defaults["decoder_chunk_size"] == 4096
    assert defaults["cross_batch_decoder_cache_bytes"] == 0

    assert metadata["resource_profile"] == PHASE4_FEEDBACK_DIAGNOSTIC_RESOURCE_PROFILE
    assert metadata["array_concurrency"] == 1
    assert metadata["baseline_registry"] == (
        "experiments/baselines/governor_calibration_granite_20260719.json"
    )
    assert metadata["baseline_registry_pinned"] is True
    assert metadata["fail_on_baseline_missing"] is True
    assert metadata["fail_on_validation_fail"] is True
    assert metadata["immutable_validation_config"] is True
    assert metadata["matrix_case_count"] == 5
    assert metadata["recommended_output_root"] == str(
        scratch_root
        / "granite"
        / "sweep"
        / "governor_calibration"
        / "phase4_feedback_diagnostic"
        / "gemma3_1b_plt"
    )
    assert metadata["governor_admission_policy"]["mode"] == "advisory"
    assert metadata["fixed_protocol"] == {
        "hardware": "single_h200",
        "gpu_count": 1,
        "dtype": "fp32",
        "fixture": "361_base",
        "decoder_chunk_size": 4096,
        "decoder_cache_bytes": 0,
        "source_batch_size": 128,
        "logit_batch_size": 128,
        "phase1_physical_cap": 128,
        "phase3_physical_cap": 128,
    }
    assert set(metadata["contrasts"]) == {
        "session_capacity",
        "logical_grouping_constant_window",
        "physical_microbatch_constant_window",
        "stale_window",
    }
    assert {
        name: (contrast["reference"], contrast["comparison"])
        for name, contrast in metadata["contrasts"].items()
    } == {
        "session_capacity": (
            "constant_window_logical_b256_session128_physical_b128",
            "constant_window_logical_b256_physical_b128",
        ),
        "logical_grouping_constant_window": (
            "canonical_reference",
            "constant_window_logical_b256_session128_physical_b128",
        ),
        "physical_microbatch_constant_window": (
            "constant_window_logical_b256_physical_b128",
            "constant_window_logical_b256_physical_b256",
        ),
        "stale_window": (
            "constant_window_logical_b256_physical_b128",
            "stale_window_logical_b256_physical_b128",
        ),
    }
    assert "512-feature" in metadata["contrasts"]["logical_grouping_constant_window"][
        "change"
    ]
    assert "1024-feature" in metadata["contrasts"]["stale_window"]["change"]
    assert metadata["slurm"] == {
        "account": "rai",
        "partition": "rai-gpu-grn",
        "qos": "rai-gpu-grn-short",
        "gres": "gpu:h200:1",
        "cpus_per_task": 12,
        "mem": "200G",
        "time": "02:00:00",
    }
