from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
from nlp_research_project.exact_trace_bench.full_answer.prepared_sequence import (
    build_prepared_sequence,
)
from nlp_research_project.exact_trace_bench.performance_campaign import (
    resolve_frozen_campaign_workload,
    validate_performance_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_batched_vjp_holdout_512_v1.json"
)
LADDER_12B = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls5_12b_batched_vjp_length_ladder_v1.json"
)
FEASIBILITY_12B_1024 = (
    REPO_ROOT / "experiments/performance_campaigns/ls5_12b_batched_vjp_1024_v1.json"
)
FEASIBILITY_12B_1024_DYNAMIC = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls5_12b_batched_vjp_1024_dynamic_rows_v2.json"
)
PROFILE_12B = "ls5-12b-cuda-windowed-width1-batched-vjp-v1"
PROFILE_12B_DYNAMIC = "ls5-12b-cuda-windowed-width1-batched-vjp-dynamic-rows-v2"
WORKLOADS_12B = (
    "ls5-batched-vjp-12b-361-dev-129",
    "ls5-batched-vjp-12b-361-dev-256",
    "ls5-batched-vjp-12b-361-dev-512",
)


def _prepare(
    *,
    manifest: Path,
    workload_id: str,
    output_dir: Path,
    launch_output_root: Path,
    memory: str,
    walltime: str,
    preheat_path: Path | None = None,
) -> None:
    command = [
        "prepare-campaign-workload",
        str(manifest),
        workload_id,
        "--profile-role",
        "run",
        "--execution-mode",
        "full",
        "--output-dir",
        str(output_dir),
        "--launch-output-root",
        str(launch_output_root),
        "--run-id",
        f"test-{workload_id}",
        "--allocation-cluster",
        "granite",
        "--allocation-profile",
        "rai-gpu-grn",
        "--allocation-account",
        "rai",
        "--allocation-partition",
        "rai-gpu-grn",
        "--allocation-qos",
        "rai-gpu-grn",
        "--allocation-gpus-per-task",
        "1",
        "--allocation-cpus-per-task",
        "32",
        "--allocation-mem",
        memory,
        "--allocation-walltime",
        walltime,
    ]
    if preheat_path is not None:
        command.extend(
            ["--preheat-policy", "file_cache", "--preheat-path", str(preheat_path)]
        )
    assert perf_cli.main(command) == 0


def test_12b_batched_vjp_profile_preserves_source_and_freezes_shared_cache_scope() -> (
    None
):
    source = perf_cli.CANDIDATE_PROFILES[
        "plt-selective-mapped-rows-cuda-windowed-12b-v1"
    ]
    profile = perf_cli.CANDIDATE_PROFILES[PROFILE_12B]

    assert len(source.variants) == len(profile.variants) == 1
    source_variant = source.variants[0]
    variant = profile.variants[0]
    assert variant.requires == source_variant.requires
    assert variant.baseline_pins == source_variant.baseline_pins
    assert variant.baseline_pins.semantic.exact_trace_internal_dtype == "fp32"
    assert profile.evidence_scope == source.evidence_scope
    assert variant.physical.as_overrides() == {
        **source_variant.physical.as_overrides(),
        "checkpoint_asset_scope": "shared",
        "backward_engine_mode": "single_forward_batched_vjp",
        "feature_row_influence_requirement": "required",
    }
    physical = variant.physical.as_overrides()
    assert physical["feature_row_influence_mode"] == "cuda_windowed"
    assert physical["checkpoint_asset_scope"] == "shared"
    assert physical["nnsight_session_capacity"] == 64
    assert physical["phase1_trace_batch_size_max"] == 64
    assert physical["phase3_compute_microbatch_max_rows"] == 64
    assert physical["phase4_execution_batch_max_rows"] == 64
    assert physical["decoder_active_row_max_bytes"] == 8 * 1024**3
    assert "decoder_active_row_safety_margin_bytes" not in physical
    assert "decoder_active_row_residency_requirement" not in physical


def test_12b_dynamic_row_profile_is_versioned_and_fail_closed() -> None:
    legacy = perf_cli.CANDIDATE_PROFILES[PROFILE_12B].variants[0]
    dynamic = perf_cli.CANDIDATE_PROFILES[PROFILE_12B_DYNAMIC].variants[0]

    assert dynamic.requires == legacy.requires
    assert dynamic.baseline_pins == legacy.baseline_pins
    physical = dynamic.physical.as_overrides()
    assert physical["decoder_active_row_residency"] is True
    assert physical["decoder_active_row_residency_requirement"] == "required"
    assert physical["decoder_active_row_max_bytes"] == 0
    assert physical["decoder_active_row_safety_margin_bytes"] == 16 * 1024**3


def test_holdout_and_12b_ladder_are_frozen_full_single_run_arms() -> None:
    holdout_payload = json.loads(HOLDOUT.read_text())
    holdout = validate_performance_campaign(holdout_payload, require_frozen=True)
    assert len(holdout) == 1
    assert holdout[0].model_variant == "gemma3_4b"
    assert holdout[0].prompt_role == "holdout"
    assert holdout[0].requested_prefix_tokens == 512
    assert holdout_payload["execution_policy"]["ordered_profile_roles"] == ["run"]
    assert holdout_payload["execution_policy"]["required_scheduler_request"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "cpus_per_task": 32,
        "memory": "400G",
        "walltime": "01:00:00",
    }

    ladder_payload = json.loads(LADDER_12B.read_text())
    ladder = validate_performance_campaign(ladder_payload, require_frozen=True)
    assert [workload.workload_id for workload in ladder] == list(WORKLOADS_12B)
    assert [workload.requested_prefix_tokens for workload in ladder] == [129, 256, 512]
    assert {workload.model_variant for workload in ladder} == {"gemma3_12b"}
    execution = ladder_payload["execution_policy"]
    assert execution["ordered_workload_ids"] == list(WORKLOADS_12B)
    assert execution["ordered_profile_roles"] == ["run"]
    assert execution["same_scheduler_allocation_required"] is True
    assert execution["preheat_policy"] == "file_cache_once_before_sequence"
    assert execution["failure_policy"] == "stop_on_first_failure"
    assert execution["classification_policy"] == ("no_advance_past_failed_predecessor")
    assert execution["token_1024_rung_blocked"] is True
    assert execution["required_scheduler_request"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "cpus_per_task": 32,
        "memory": "600G",
        "walltime": "02:00:00",
    }
    for raw in ladder_payload["workloads"]:
        assert raw["profiles"] == {"run": PROFILE_12B}
        policy = raw["comparison_policy"]
        assert policy["required_execution_mode"] == "full"
        assert policy["required_exact_trace_internal_dtype"] == "fp32"
        assert policy["required_checkpoint_asset_scope"] == "shared"
        assert policy["required_actual_phase1_forward_trace_width"] == 1
        assert policy["runtime_resource_policy"] == "measure_only"
        arm = policy["arms"]["run"]
        assert arm["forward_lane_count"] == 1
        assert arm["config_pins"] == {
            "exact_trace_internal_dtype": "fp32",
            "checkpoint_asset_scope": "shared",
        }
        assert arm["physical_caps"] == {
            "nnsight_session_capacity": 64,
            "phase1_trace_batch_size_max": 64,
            "phase3_compute_microbatch_max_rows": 64,
            "phase4_execution_batch_max_rows": 64,
        }
        assert arm["derived_execution"] == {
            "actual_phase1_forward_trace_width": 1,
            "backward_batch_capacity": 64,
        }
        resolved = resolve_frozen_campaign_workload(LADDER_12B, raw["workload_id"])
        assert len(resolved.prefix_token_ids) == raw["prefix"]["actual_tokens"]


def test_12b_1024_is_a_frozen_single_run_after_the_512_gate() -> None:
    payload = json.loads(FEASIBILITY_12B_1024.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.workload_id == "ls5-batched-vjp-12b-361-dev-1024"
    assert workload.model_variant == "gemma3_12b"
    assert workload.requested_prefix_tokens == 1024
    assert workload.actual_prefix_tokens == 1024

    execution = payload["execution_policy"]
    assert execution["ordered_profile_roles"] == ["run"]
    assert execution["full_execution_required"] is True
    assert execution["single_scheduler_job_required"] is True
    assert execution["preheat_policy"] == "file_cache_once_before_run"
    assert execution["legacy_control_forbidden"] is True
    assert execution["required_scheduler_request"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "cpus_per_task": 32,
        "memory": "600G",
        "walltime": "02:00:00",
    }
    assert execution["qualification_only_no_default_or_promotion"] is True
    assert execution["qualification_only_no_repeatability_claim"] is True
    assert execution["qualification_only_no_cross_regime_claim"] is True
    assert execution["qualification_only_no_native_12b_trajectory_claim"] is True
    assert execution["qualification_only_no_further_scaling_authorization"] is True

    raw = payload["workloads"][0]
    assert raw["profiles"] == {"run": PROFILE_12B}
    assert raw["prefix"] == {
        "requested_tokens": 1024,
        "actual_tokens": 1024,
        "token_ids_sha256": (
            "75346f1791031a997292f1b625eda54d86cdc28244f811658f343f1ef9fa9625"
        ),
    }
    assert raw["target"] == {
        "absolute_position": 1024,
        "token_id": 9338,
        "token_text": " journey",
    }
    policy = raw["comparison_policy"]
    assert policy["predecessor_gate"] == "ls5-batched-vjp-12b-361-dev-512"
    assert policy["advance_only_if_predecessor_passes"] is True
    assert policy["required_execution_mode"] == "full"
    assert policy["required_exact_trace_internal_dtype"] == "fp32"
    assert policy["required_checkpoint_asset_scope"] == "shared"
    assert policy["required_feature_row_influence_mode"] == "cuda_windowed"
    assert policy["feature_row_influence_requirement"] == "required"
    assert policy["required_backward_engine_mode"] == "single_forward_batched_vjp"
    assert policy["required_forward_graph_mode"] == "single_lane"
    assert policy["required_vjp_kernel_mode"] == "autograd_batched"
    assert policy["required_forward_lane_count"] == 1
    assert policy["required_actual_phase1_forward_trace_width"] == 1
    assert policy["runtime_resource_policy"] == "measure_only"
    arm = policy["arms"]["run"]
    assert arm["config_pins"] == {
        "exact_trace_internal_dtype": "fp32",
        "checkpoint_asset_scope": "shared",
    }
    assert arm["physical_caps"] == {
        "nnsight_session_capacity": 64,
        "phase1_trace_batch_size_max": 64,
        "phase3_compute_microbatch_max_rows": 64,
        "phase4_execution_batch_max_rows": 64,
    }
    assert arm["derived_execution"] == {
        "actual_phase1_forward_trace_width": 1,
        "backward_batch_capacity": 64,
    }

    resolved = resolve_frozen_campaign_workload(
        FEASIBILITY_12B_1024,
        "ls5-batched-vjp-12b-361-dev-1024",
    )
    assert len(resolved.prefix_token_ids) == 1024


def test_12b_1024_dynamic_row_rerun_supersedes_failed_v1_without_mutating_it() -> None:
    legacy = json.loads(FEASIBILITY_12B_1024.read_text())
    payload = json.loads(FEASIBILITY_12B_1024_DYNAMIC.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert legacy["campaign_id"] == "ls5-gemma3-12b-batched-vjp-1024-feasibility-v1"
    assert legacy["workloads"][0]["profiles"] == {"run": PROFILE_12B}
    assert payload["execution_policy"]["supersedes_failed_job_id"] == "1819287"
    assert len(workloads) == 1
    raw = payload["workloads"][0]
    assert raw["profiles"] == {"run": PROFILE_12B_DYNAMIC}
    pins = raw["comparison_policy"]["arms"]["run"]["config_pins"]
    assert pins["decoder_active_row_residency"] is True
    assert pins["decoder_active_row_residency_requirement"] == "required"
    assert pins["decoder_active_row_max_bytes"] == 0
    assert pins["decoder_active_row_safety_margin_bytes"] == 16 * 1024**3


@pytest.mark.parametrize(
    ("manifest", "workload_id", "prefix", "model_profile", "memory", "walltime"),
    [
        (
            HOLDOUT,
            "ls4-batched-vjp-4b-828-holdout-512",
            512,
            "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1",
            "400G",
            "01:00:00",
        ),
        *[
            (LADDER_12B, workload_id, prefix, PROFILE_12B, "600G", "02:00:00")
            for workload_id, prefix in zip(WORKLOADS_12B, (129, 256, 512), strict=True)
        ],
        (
            FEASIBILITY_12B_1024,
            "ls5-batched-vjp-12b-361-dev-1024",
            1024,
            PROFILE_12B,
            "600G",
            "02:00:00",
        ),
    ],
)
def test_prepare_scaling_arm_freezes_mechanism_capacity_and_allocation(
    tmp_path: Path,
    manifest: Path,
    workload_id: str,
    prefix: int,
    model_profile: str,
    memory: str,
    walltime: str,
) -> None:
    output_dir = tmp_path / "prepared"
    _prepare(
        manifest=manifest,
        workload_id=workload_id,
        output_dir=output_dir,
        launch_output_root=tmp_path / "run",
        memory=memory,
        walltime=walltime,
    )
    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())

    assert prepared["profile_name"] == model_profile
    assert prepared["execution_mode"] == "full"
    assert trace_spec["prefix_token_count"] == prefix
    knobs = trace_spec["graph_knobs"]
    assert knobs["exact_trace_internal_dtype"] == "fp32"
    assert knobs["feature_row_influence_mode"] == "cuda_windowed"
    assert knobs["feature_row_influence_requirement"] == "required"
    assert knobs["backward_engine_mode"] == "single_forward_batched_vjp"
    expected_capacity = 128 if manifest == HOLDOUT else 64
    for key in (
        "nnsight_session_capacity",
        "phase1_trace_batch_size_max",
        "phase3_compute_microbatch_max_rows",
        "phase4_execution_batch_max_rows",
    ):
        assert knobs[key] == expected_capacity
    if manifest in {LADDER_12B, FEASIBILITY_12B_1024}:
        assert knobs["checkpoint_asset_scope"] == "shared"
    mechanism = launch_spec["mechanism_selections"][0]
    assert mechanism["forward_graph_mode"] == "single_lane"
    assert mechanism["vjp_kernel_mode"] == "autograd_batched"
    assert mechanism["forward_lane_count"] == 1
    assert launch_spec["runtime_resource_policy"] == "measure_only"
    assert launch_spec["scheduler_request"]["memory"] == memory
    assert launch_spec["scheduler_request"]["walltime"] == walltime


def test_prepare_12b_sequence_is_one_preheated_stop_on_failure_allocation(
    tmp_path: Path,
) -> None:
    entries: list[tuple[str, Path]] = []
    for workload_id in WORKLOADS_12B:
        output_dir = tmp_path / "prepared" / workload_id
        _prepare(
            manifest=LADDER_12B,
            workload_id=workload_id,
            output_dir=output_dir,
            launch_output_root=tmp_path / "runs" / workload_id,
            memory="600G",
            walltime="02:00:00",
            preheat_path=LADDER_12B,
        )
        entries.append((workload_id.rsplit("-", 1)[-1], output_dir))

    sequence = build_prepared_sequence(entries)
    assert sequence["execution_policy"] == {
        "allocation_reuse": "single_allocation",
        "preheat_count": 1,
        "failure_policy": "stop_on_first_failure",
        "output_policy": "refuse_existing",
    }
    assert [entry["entry_id"] for entry in sequence["entries"]] == [
        "129",
        "256",
        "512",
    ]
    assert sequence["preheat"] == {
        "policy": "file_cache",
        "paths": [str(LADDER_12B.resolve())],
    }


def test_prepare_12b_fails_closed_if_shared_cache_pin_drifts(tmp_path: Path) -> None:
    payload = json.loads(LADDER_12B.read_text())
    payload["workloads"][0]["comparison_policy"]["arms"]["run"]["config_pins"][
        "checkpoint_asset_scope"
    ] = "job_private"
    manifest = tmp_path / "drifted.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="config_pins.checkpoint_asset_scope"):
        _prepare(
            manifest=manifest,
            workload_id=WORKLOADS_12B[0],
            output_dir=tmp_path / "prepared",
            launch_output_root=tmp_path / "run",
            memory="600G",
            walltime="02:00:00",
        )
