from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
from nlp_research_project.exact_trace_bench.full_answer.launch_spec import (
    build_full_answer_launch_spec,
    render_local_full_answer_command,
)
from nlp_research_project.exact_trace_bench.performance_campaign import (
    freeze_performance_campaign,
    resolve_frozen_campaign_workload,
    validate_mechanism_claims,
    validate_performance_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "experiments/performance_campaigns/sp0_mechanism_claims.json"
CAMPAIGN = REPO_ROOT / "experiments/performance_campaigns/ls0_prefix_scaling.json"
LS4_CAMPAIGN = REPO_ROOT / "experiments/performance_campaigns/ls4_4b_transfer.json"
LS4_REMAINING_GATES_V2 = (
    REPO_ROOT / "experiments/performance_campaigns/ls4_4b_remaining_gates_v2.json"
)
LS4_BATCHED_VJP_V1 = (
    REPO_ROOT / "experiments/performance_campaigns/ls4_4b_batched_vjp_v1.json"
)
LS4_BATCHED_VJP_SELF_CONSISTENCY_V1 = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_batched_vjp_self_consistency_v1.json"
)
LS4_BATCHED_VJP_SELF_CONSISTENCY_512_V1 = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_batched_vjp_self_consistency_512_v1.json"
)
LS4_BATCHED_VJP_1024_V1 = (
    REPO_ROOT / "experiments/performance_campaigns/ls4_4b_batched_vjp_1024_v1.json"
)
LS4_VJP_DECOMPOSITION_DIAGNOSTIC_V1 = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_vjp_decomposition_diagnostic_v1.json"
)
LS4_VJP_EARLY_LOCALIZATION_V2 = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_vjp_early_localization_v2.json"
)
LS4_VJP_INJECTED_WIDTH_SWEEP_V1 = (
    REPO_ROOT
    / "experiments/performance_campaigns/ls4_4b_vjp_injected_width_sweep_v1.json"
)
LS5_DIRECT_1024_V1 = (
    REPO_ROOT / "experiments/performance_campaigns/ls5_12b_1024_transfer.json"
)
LS5_LENGTH_LADDER_V2 = (
    REPO_ROOT / "experiments/performance_campaigns/ls5_12b_length_ladder_v2.json"
)


def test_committed_sp0_claims_validate() -> None:
    payload = json.loads(CLAIMS.read_text())
    validate_mechanism_claims(payload)
    statuses = {claim["disposition"]["status"] for claim in payload["claims"]}
    assert statuses == {
        "exact_promotion_candidate",
        "exact_opt_in",
        "bounded_research",
        "rejected",
    }


def test_committed_ls0_scaffold_lists_without_model_loading() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    workloads = validate_performance_campaign(payload)
    assert [workload.requested_prefix_tokens for workload in workloads] == [
        129,
        256,
        512,
        1024,
        256,
        256,
        512,
        512,
    ]
    assert [workload.prompt_role for workload in workloads] == [
        "development",
        "development",
        "development",
        "development",
        "holdout",
        "holdout",
        "holdout",
        "holdout",
    ]


def test_ls0_development_and_holdout_workloads_are_frozen() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert {workload.prompt_role for workload in workloads} == {
        "development",
        "holdout",
    }


def test_ls4_transfer_manifest_is_frozen_and_4b_only() -> None:
    payload = json.loads(LS4_CAMPAIGN.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert [workload.requested_prefix_tokens for workload in workloads] == [
        129,
        256,
        512,
        1024,
        512,
    ]
    assert {workload.model_variant for workload in workloads} == {"gemma3_4b"}
    assert [workload.prompt_role for workload in workloads] == [
        "development",
        "development",
        "development",
        "development",
        "holdout",
    ]


def test_ls5_direct_1024_v1_manifest_preserves_superseded_launch_provenance() -> None:
    payload = json.loads(LS5_DIRECT_1024_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.workload_id == "ls5-12b-361-dev-1024"
    assert workload.model_variant == "gemma3_12b"
    assert workload.requested_prefix_tokens == 1024
    assert workload.actual_prefix_tokens == 1024
    assert workload.candidate_profile == (
        "plt-selective-mapped-rows-cuda-windowed-12b-v1"
    )
    assert (
        payload["workloads"][0]["comparison_policy"][
            "full_trace_runtime_ceiling_seconds"
        ]
        == 28800
    )


def test_ls4_remaining_gates_v2_are_frozen_ordered_and_two_hour_bounded() -> None:
    payload = json.loads(LS4_REMAINING_GATES_V2.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert [workload.workload_id for workload in workloads] == [
        "ls4-v2-4b-361-dev-1024",
        "ls4-v2-4b-828-holdout-512",
    ]
    assert [workload.requested_prefix_tokens for workload in workloads] == [1024, 512]
    assert [workload.prompt_role for workload in workloads] == [
        "development",
        "holdout",
    ]
    assert {workload.model_variant for workload in workloads} == {"gemma3_4b"}
    assert payload["execution_policy"]["ordered_workload_ids"] == [
        workload.workload_id for workload in workloads
    ]
    assert payload["execution_policy"]["advance_only_after_classification"] is True
    for raw_workload in payload["workloads"]:
        assert raw_workload["resource_envelope"]["walltime_seconds"] <= 7200
        policy = raw_workload["comparison_policy"]
        assert policy["probe_before_full"] is True
        assert policy["required_candidate_mode"] == "cuda_windowed"
        assert policy["runtime_resource_policy"] == "measure_only"
        assert policy["full_trace_runtime_ceiling_seconds"] <= 7200
        resolved = resolve_frozen_campaign_workload(
            LS4_REMAINING_GATES_V2, raw_workload["workload_id"]
        )
        assert len(resolved.prefix_token_ids) == raw_workload["prefix"]["actual_tokens"]


def test_ls4_batched_vjp_campaign_is_frozen_matched_and_staged_256_to_512() -> None:
    payload = json.loads(LS4_BATCHED_VJP_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert [workload.requested_prefix_tokens for workload in workloads] == [256, 512]
    assert {workload.model_variant for workload in workloads} == {"gemma3_4b"}
    assert payload["execution_policy"]["ordered_workload_ids"] == [
        workload.workload_id for workload in workloads
    ]
    for raw_workload in payload["workloads"]:
        assert raw_workload["resource_envelope"]["walltime_seconds"] <= 7200
        assert raw_workload["profiles"]["control"] == (
            "ls4-4b-transfer-cuda-windowed-512mib-b128-v1"
        )
        assert raw_workload["profiles"]["candidate"] == (
            "ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"
        )
        policy = raw_workload["comparison_policy"]
        assert policy["isolated_axis"] == "backward_engine_mode"
        assert policy["required_candidate_mode"] == "single_forward_batched_vjp"
        assert policy["required_candidate_forward_lane_count"] == 1
        resolved = resolve_frozen_campaign_workload(
            LS4_BATCHED_VJP_V1, raw_workload["workload_id"]
        )
        assert len(resolved.prefix_token_ids) == raw_workload["prefix"]["actual_tokens"]


def test_ls4_batched_vjp_self_consistency_is_two_frozen_full_repeats() -> None:
    payload = json.loads(LS4_BATCHED_VJP_SELF_CONSISTENCY_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    assert workloads[0].requested_prefix_tokens == 256
    assert workloads[0].model_variant == "gemma3_4b"
    execution = payload["execution_policy"]
    assert execution["ordered_profile_roles"] == ["repeat_a", "repeat_b"]
    assert execution["full_execution_required"] is True
    assert execution["separate_scheduler_jobs_required"] is True
    assert execution["same_profile_required"] is True
    assert execution["legacy_control_forbidden"] is True
    assert execution["qualification_only_no_default_or_promotion"] is True

    raw_workload = payload["workloads"][0]
    assert len(set(raw_workload["profiles"].values())) == 1
    assert raw_workload["profiles"]["repeat_a"] == (
        "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
    )
    policy = raw_workload["comparison_policy"]
    assert policy["required_execution_mode"] == "full"
    assert policy["required_feature_row_influence_mode"] == "cuda_windowed"
    assert policy["feature_row_influence_requirement"] == "required"
    assert policy["required_backward_engine_mode"] == ("single_forward_batched_vjp")
    assert policy["required_forward_lane_count"] == 1
    assert policy["required_actual_phase1_forward_trace_width"] == 1
    assert policy["runtime_resource_policy"] == "measure_only"
    assert "does not establish equivalence" in policy["claim_boundary"]
    assert "actual Phase-1 forward trace width=1" in policy["capacity_contract"]
    assert "all frozen at 128" in policy["capacity_contract"]
    assert "run_id" in policy["allowed_repeat_variation"]
    assert "launch_output_root" in policy["allowed_repeat_variation"]
    expected_physical_caps = {
        "nnsight_session_capacity": 128,
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
    }
    for arm in policy["arms"].values():
        assert arm["forward_lane_count"] == 1
        assert arm["physical_caps"] == expected_physical_caps
        assert arm["derived_execution"] == {
            "actual_phase1_forward_trace_width": 1,
            "backward_batch_capacity": 128,
        }

    resolved = resolve_frozen_campaign_workload(
        LS4_BATCHED_VJP_SELF_CONSISTENCY_V1,
        "ls4-batched-vjp-self-consistency-4b-361-dev-256",
    )
    assert len(resolved.prefix_token_ids) == 256


def test_ls4_batched_vjp_512_self_consistency_is_an_isolated_frozen_rung() -> None:
    payload = json.loads(LS4_BATCHED_VJP_SELF_CONSISTENCY_512_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    assert workloads[0].requested_prefix_tokens == 512
    assert workloads[0].model_variant == "gemma3_4b"
    execution = payload["execution_policy"]
    assert execution["ordered_profile_roles"] == ["repeat_a", "repeat_b"]
    assert execution["full_execution_required"] is True
    assert execution["separate_scheduler_jobs_required"] is True
    assert execution["same_profile_required"] is True
    assert execution["legacy_control_forbidden"] is True
    assert execution["qualification_only_no_default_or_promotion"] is True
    assert execution["qualification_only_no_scaling_authorization"] is True
    assert execution["required_scheduler_request"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "cpus_per_task": 32,
        "memory": "400G",
        "walltime": "01:00:00",
    }

    raw_workload = payload["workloads"][0]
    assert len(set(raw_workload["profiles"].values())) == 1
    assert raw_workload["profiles"]["repeat_a"] == (
        "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
    )
    assert raw_workload["resource_envelope"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "host_memory_stop_gib": 400,
        "host_rss_stop_gib": 300,
        "hbm_peak_fraction_max": 0.9,
        "walltime_seconds": 3600,
    }
    policy = raw_workload["comparison_policy"]
    assert policy["required_execution_mode"] == "full"
    assert policy["required_feature_row_influence_mode"] == "cuda_windowed"
    assert policy["feature_row_influence_requirement"] == "required"
    assert policy["required_backward_engine_mode"] == ("single_forward_batched_vjp")
    assert policy["required_forward_lane_count"] == 1
    assert policy["required_actual_phase1_forward_trace_width"] == 1
    assert policy["runtime_resource_policy"] == "measure_only"
    assert "does not establish equivalence" in policy["claim_boundary"]
    assert "authorize the next scaling rung" in policy["claim_boundary"]
    assert policy["comparator_scope"] == {
        "canonical_all_edge_scope": "legacy_compact_edges_only",
        "canonical_all_edge_includes_typed_buckets": False,
        "typed_bucket_comparison_required": True,
        "whole_npz_strict_exact_label_forbidden": True,
    }
    acceptance = policy["acceptance_contract"]
    assert acceptance["classification"] == ("two_tier_same_regime_self_consistency")
    assert acceptance["canonical_selected_feature_circuit"] == {
        "required_classification": "strict_exact",
        "required_target_token_match": 1.0,
        "required_support_jaccard": 1.0,
        "required_weighted_jaccard": 1.0,
        "required_shared_sign_agreement": 1.0,
        "required_normalized_l1_deviation": 0.0,
        "required_signed_normalized_l1_deviation": 0.0,
        "required_topk_exact": [64, 128, 256, 512, 1024],
    }
    assert acceptance["non_error_typed_buckets"] == {
        "buckets": [
            "feature<-feature",
            "feature<-token",
            "logit<-feature",
            "logit<-token",
        ],
        "required_classification": "strict_exact",
    }
    assert acceptance["error_typed_buckets"] == {
        "buckets": ["feature<-error", "logit<-error"],
        "minimum_support_jaccard": 0.99,
        "minimum_weighted_jaccard": 0.99,
        "minimum_shared_sign_agreement": 0.99,
        "maximum_normalized_l1_deviation": 0.01,
        "maximum_signed_normalized_l1_deviation": 0.01,
        "topk_exactness_required": False,
    }
    assert acceptance["failure_action"] == {
        "if_core_not_strict_exact_or_error_bucket_out_of_bounds": [
            "stop_before_1024",
            "run_third_independent_repeat",
            "localize_by_typed_bucket",
        ],
        "default_promotion_forbidden": True,
        "automatic_scaling_authorization_forbidden": True,
    }
    assert policy["fidelity"] == ("two_tier_same_regime_self_consistency_only")
    expected_physical_caps = {
        "nnsight_session_capacity": 128,
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
    }
    for arm in policy["arms"].values():
        assert arm["forward_lane_count"] == 1
        assert arm["physical_caps"] == expected_physical_caps
        assert arm["derived_execution"] == {
            "actual_phase1_forward_trace_width": 1,
            "backward_batch_capacity": 128,
        }

    resolved = resolve_frozen_campaign_workload(
        LS4_BATCHED_VJP_SELF_CONSISTENCY_512_V1,
        "ls4-batched-vjp-self-consistency-4b-361-dev-512",
    )
    assert len(resolved.prefix_token_ids) == 512
    assert resolved.workload["prefix"]["token_ids_sha256"] == (
        "8a01ec6a0f7fe0f91a1ebf8c489f258af140086aac7db35eccb2e0016091b78b"
    )
    assert resolved.workload["target"] == {
        "absolute_position": 512,
        "token_id": 33036,
        "token_text": " travels",
    }


def test_ls4_batched_vjp_1024_is_a_single_frozen_feasibility_run() -> None:
    payload = json.loads(LS4_BATCHED_VJP_1024_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    assert workloads[0].requested_prefix_tokens == 1024
    assert workloads[0].model_variant == "gemma3_4b"
    execution = payload["execution_policy"]
    assert execution["ordered_profile_roles"] == ["run"]
    assert execution["full_execution_required"] is True
    assert execution["single_scheduler_job_required"] is True
    assert execution["legacy_control_forbidden"] is True
    assert execution["qualification_only_no_default_or_promotion"] is True
    assert execution["qualification_only_no_self_consistency_claim"] is True
    assert execution["qualification_only_no_cross_regime_claim"] is True
    assert execution["qualification_only_no_12b_authorization"] is True
    assert execution["qualification_only_no_further_scaling_authorization"] is True
    assert execution["required_scheduler_request"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "cpus_per_task": 32,
        "memory": "400G",
        "walltime": "01:00:00",
    }

    raw_workload = payload["workloads"][0]
    assert raw_workload["profiles"] == {
        "run": "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
    }
    assert raw_workload["resource_envelope"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "host_memory_stop_gib": 400,
        "host_rss_stop_gib": 300,
        "hbm_peak_fraction_max": 0.9,
        "walltime_seconds": 3600,
    }
    policy = raw_workload["comparison_policy"]
    assert policy["comparison_domain"] == "none_single_run"
    assert policy["required_execution_mode"] == "full"
    assert policy["required_feature_row_influence_mode"] == "cuda_windowed"
    assert policy["feature_row_influence_requirement"] == "required"
    assert policy["required_backward_engine_mode"] == ("single_forward_batched_vjp")
    assert policy["required_forward_graph_mode"] == "single_lane"
    assert policy["required_vjp_kernel_mode"] == "autograd_batched"
    assert policy["required_forward_lane_count"] == 1
    assert policy["required_actual_phase1_forward_trace_width"] == 1
    assert policy["runtime_resource_policy"] == "measure_only"
    assert policy["artifact_acceptance_contract"] == {
        "classification": "single_run_feasibility_and_within_regime_artifact_only",
        "required": [
            "full_trace_completion",
            "strict_artifact_validation",
            "graph_reopen_success",
            "frozen_mechanism_telemetry_match",
            "resource_telemetry_persisted",
        ],
        "forbidden_claims": [
            "self_consistency",
            "cross_regime_equivalence",
            "default_promotion",
            "further_scaling_authorization",
            "12b_authorization",
        ],
    }
    assert policy["arms"] == {
        "run": {
            "backward_engine_mode": "single_forward_batched_vjp",
            "forward_graph_mode": "single_lane",
            "vjp_kernel_mode": "autograd_batched",
            "forward_lane_count": 1,
            "physical_caps": {
                "nnsight_session_capacity": 128,
                "phase1_trace_batch_size_max": 128,
                "phase3_compute_microbatch_max_rows": 128,
                "phase4_execution_batch_max_rows": 128,
            },
            "derived_execution": {
                "actual_phase1_forward_trace_width": 1,
                "backward_batch_capacity": 128,
            },
        }
    }
    assert policy["fidelity"] == (
        "single_run_same_regime_feasibility_and_artifact_only"
    )
    claim_boundary = policy["claim_boundary"]
    for forbidden_claim in (
        "self-consistency",
        "equivalence",
        "default",
        "12B",
        "further scaling",
    ):
        assert forbidden_claim in claim_boundary

    resolved = resolve_frozen_campaign_workload(
        LS4_BATCHED_VJP_1024_V1,
        "ls4-batched-vjp-4b-361-dev-1024",
    )
    assert len(resolved.prefix_token_ids) == 1024
    assert resolved.workload["prefix"]["token_ids_sha256"] == (
        "75346f1791031a997292f1b625eda54d86cdc28244f811658f343f1ef9fa9625"
    )
    assert resolved.workload["target"] == {
        "absolute_position": 1024,
        "token_id": 9338,
        "token_text": " journey",
    }


def test_prepare_batched_vjp_self_consistency_rejects_nonfrozen_run_controls(
    tmp_path: Path,
) -> None:
    base_args = [
        "prepare-campaign-workload",
        str(LS4_BATCHED_VJP_SELF_CONSISTENCY_V1),
        "ls4-batched-vjp-self-consistency-4b-361-dev-256",
        "--profile-role",
        "repeat_a",
    ]
    with pytest.raises(ValueError, match="requires execution mode 'full'"):
        perf_cli.main(
            [
                *base_args,
                "--execution-mode",
                "phase3-probe",
                "--output-dir",
                str(tmp_path / "wrong-mode"),
            ]
        )
    with pytest.raises(ValueError, match="requires runtime resource policy"):
        perf_cli.main(
            [
                *base_args,
                "--execution-mode",
                "full",
                "--runtime-resource-policy",
                "enforce",
                "--output-dir",
                str(tmp_path / "wrong-resource-policy"),
            ]
        )


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "expected_error"),
    [
        (
            "physical_caps",
            "phase3_compute_microbatch_max_rows",
            127,
            "physical_caps.phase3_compute_microbatch_max_rows",
        ),
        (
            "derived_execution",
            "backward_batch_capacity",
            127,
            "derived_execution.backward_batch_capacity",
        ),
        (
            "derived_execution",
            "actual_phase1_forward_trace_width",
            2,
            "derived_execution.actual_phase1_forward_trace_width",
        ),
    ],
)
def test_prepare_batched_vjp_self_consistency_fails_closed_on_capacity_mismatch(
    tmp_path: Path,
    section: str,
    field: str,
    bad_value: int,
    expected_error: str,
) -> None:
    payload = json.loads(LS4_BATCHED_VJP_SELF_CONSISTENCY_V1.read_text())
    payload["workloads"][0]["comparison_policy"]["arms"]["repeat_a"][section][field] = (
        bad_value
    )
    manifest = tmp_path / f"bad-{section}-{field}.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=expected_error):
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(manifest),
                "ls4-batched-vjp-self-consistency-4b-361-dev-256",
                "--profile-role",
                "repeat_a",
                "--execution-mode",
                "full",
                "--output-dir",
                str(tmp_path / f"prepared-{section}-{field}"),
            ]
        )


def test_ls4_vjp_decomposition_campaign_is_frozen_three_arm_diagnostic() -> None:
    payload = json.loads(LS4_VJP_DECOMPOSITION_DIAGNOSTIC_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    workload = payload["workloads"][0]
    assert workload["prefix"]["actual_tokens"] == 256
    assert payload["execution_policy"]["ordered_profile_roles"] == [
        "control",
        "serial",
        "batched",
    ]
    assert payload["execution_policy"]["phase4_execution_batch_limit"] == 1
    assert (
        workload["comparison_policy"]["required_feature_row_influence_mode"]
        == "cuda_windowed"
    )
    arms = workload["comparison_policy"]["arms"]
    assert arms["control"] == {
        "backward_engine_mode": "duplicated_lanes",
        "forward_graph_mode": "logical_capacity",
        "vjp_kernel_mode": "nnsight_injected",
        "forward_lane_count": 128,
    }
    assert arms["serial"]["vjp_kernel_mode"] == "autograd_serial"
    assert arms["batched"]["vjp_kernel_mode"] == "autograd_batched"
    resolved = resolve_frozen_campaign_workload(
        LS4_VJP_DECOMPOSITION_DIAGNOSTIC_V1,
        "ls4-vjp-decomposition-4b-361-dev-256",
    )
    assert len(resolved.prefix_token_ids) == 256


@pytest.mark.parametrize(
    (
        "profile_role",
        "backward_engine_mode",
        "forward_graph_mode",
        "vjp_kernel_mode",
        "lane_count",
    ),
    [
        ("control", "duplicated_lanes", "logical_capacity", "nnsight_injected", 128),
        ("serial", "single_forward_serial_vjp", "single_lane", "autograd_serial", 1),
        ("batched", "single_forward_batched_vjp", "single_lane", "autograd_batched", 1),
    ],
)
def test_prepare_vjp_decomposition_arm_records_diagnostics_and_identity(
    tmp_path: Path,
    profile_role: str,
    backward_engine_mode: str,
    forward_graph_mode: str,
    vjp_kernel_mode: str,
    lane_count: int,
) -> None:
    output_dir = tmp_path / profile_role
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(LS4_VJP_DECOMPOSITION_DIAGNOSTIC_V1),
                "ls4-vjp-decomposition-4b-361-dev-256",
                "--profile-role",
                profile_role,
                "--execution-mode",
                "transition-probe",
                "--probe-batches",
                "1",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())
    knobs = trace_spec["graph_knobs"]
    assert knobs["feature_row_influence_mode"] == "cuda_windowed"
    assert knobs["feature_row_influence_requirement"] == "required"
    assert knobs["backward_engine_mode"] == backward_engine_mode
    assert knobs["diagnostic_stop_mode"] == "transition_probe"
    assert knobs["diagnostic_stop_phase4_batches"] == 1
    for capture in (
        "cross_cluster_debug",
        "capture_phase0_donor_bundle",
        "capture_phase3_gradient_bundle",
        "capture_phase3_row_bundle",
        "capture_phase3_seed_bundle",
    ):
        assert knobs[capture] is True
    mechanism = launch_spec["mechanism_selections"][0]
    assert mechanism["backward_engine_mode"] == backward_engine_mode
    assert mechanism["forward_graph_mode"] == forward_graph_mode
    assert mechanism["vjp_kernel_mode"] == vjp_kernel_mode
    assert mechanism["forward_lane_count"] == lane_count


def test_ls4_vjp_early_localization_is_frozen_four_arm_phase3_probe() -> None:
    payload = json.loads(LS4_VJP_EARLY_LOCALIZATION_V2.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.control_profile is None
    assert workload.candidate_profile is None
    assert dict(workload.profiles) == payload["workloads"][0]["profiles"]
    assert payload["execution_policy"]["ordered_profile_roles"] == [
        "wide_injected",
        "narrow_injected",
        "narrow_serial",
        "narrow_batched",
    ]
    assert payload["execution_policy"]["phase3_probe_required"] is True
    assert payload["execution_policy"]["phase4_execution_forbidden"] is True
    policy = payload["workloads"][0]["comparison_policy"]
    assert "joint effect" in policy["claim_boundary"]
    assert "remains composite" in policy["claim_boundary"]
    assert "singleton width" in policy["claim_boundary"]
    assert (
        "do not support Phase-3 gradient replay"
        in policy["followup_replay_policy"]["phase3_gradient_replay"]
    )


@pytest.mark.parametrize(
    (
        "profile_role",
        "backward_engine_mode",
        "forward_graph_mode",
        "vjp_kernel_mode",
        "lane_count",
        "physical_cap",
    ),
    [
        (
            "wide_injected",
            "duplicated_lanes",
            "logical_capacity",
            "nnsight_injected",
            128,
            128,
        ),
        (
            "narrow_injected",
            "duplicated_lanes",
            "logical_capacity",
            "nnsight_injected",
            1,
            1,
        ),
        (
            "narrow_serial",
            "single_forward_serial_vjp",
            "single_lane",
            "autograd_serial",
            1,
            1,
        ),
        (
            "narrow_batched",
            "single_forward_batched_vjp",
            "single_lane",
            "autograd_batched",
            1,
            1,
        ),
    ],
)
def test_prepare_vjp_early_localization_arm_pins_identity_and_widths(
    tmp_path: Path,
    profile_role: str,
    backward_engine_mode: str,
    forward_graph_mode: str,
    vjp_kernel_mode: str,
    lane_count: int,
    physical_cap: int,
) -> None:
    output_dir = tmp_path / profile_role
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(LS4_VJP_EARLY_LOCALIZATION_V2),
                "ls4-vjp-early-localization-4b-361-dev-256",
                "--profile-role",
                profile_role,
                "--execution-mode",
                "phase3-probe",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())
    knobs = trace_spec["graph_knobs"]
    assert knobs["diagnostic_stop_mode"] == "phase3_probe"
    assert knobs["diagnostic_stop_phase4_batches"] is None
    assert knobs["nnsight_session_capacity"] == physical_cap
    assert knobs["phase1_trace_batch_policy"] == "cap_effective_batches"
    assert knobs["phase1_trace_batch_size_max"] == physical_cap
    assert knobs["phase3_compute_microbatch_max_rows"] == physical_cap
    assert knobs["phase4_execution_batch_max_rows"] == physical_cap
    for capture in (
        "capture_phase0_donor_bundle",
        "capture_phase3_gradient_bundle",
        "capture_phase3_row_bundle",
        "capture_phase3_seed_bundle",
    ):
        assert knobs[capture] is True
    mechanism = launch_spec["mechanism_selections"][0]
    assert mechanism["backward_engine_mode"] == backward_engine_mode
    assert mechanism["forward_graph_mode"] == forward_graph_mode
    assert mechanism["vjp_kernel_mode"] == vjp_kernel_mode
    assert mechanism["forward_lane_count"] == lane_count


def test_campaign_ordered_profile_roles_must_exist_in_each_workload() -> None:
    payload = json.loads(LS4_VJP_EARLY_LOCALIZATION_V2.read_text())
    del payload["workloads"][0]["profiles"]["narrow_injected"]

    with pytest.raises(ValueError, match="missing ordered roles: narrow_injected"):
        validate_performance_campaign(payload)


def test_ls4_vjp_injected_width_sweep_is_frozen_diagnostic_only() -> None:
    payload = json.loads(LS4_VJP_INJECTED_WIDTH_SWEEP_V1.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)

    assert len(workloads) == 1
    expected_roles = [
        "width128_caps1",
        "width64_caps1",
        "width32_caps1",
        "width16_caps1",
        "width8_caps1",
        "width4_caps1",
        "width2_caps1",
        "width1_caps1",
    ]
    assert payload["execution_policy"]["ordered_profile_roles"] == expected_roles
    assert payload["execution_policy"]["phase3_probe_required"] is True
    assert payload["execution_policy"]["phase4_execution_forbidden"] is True
    assert payload["execution_policy"]["diagnostic_only_no_promotion"] is True
    raw_workload = payload["workloads"][0]
    assert list(raw_workload["profiles"]) == expected_roles
    assert len(set(raw_workload["profiles"].values())) == len(expected_roles)
    assert (
        "ls4-4b-vjp-diagnostic-injected-width-128-caps-128-v1"
        not in perf_cli.CANDIDATE_PROFILES
    )
    policy = raw_workload["comparison_policy"]
    assert "one structural width factor" in policy["claim_boundary"]
    assert "does not independently isolate" in policy["claim_boundary"]
    assert "cannot promote fidelity, performance" in policy["promotion_boundary"]
    protocol = payload["execution_policy"]["experimental_protocol"]
    assert protocol["single_allocation_required"] is True
    assert protocol["endpoint_repeat_sequence"] == [
        "width128_caps1",
        "width1_caps1",
        "width1_caps1",
        "width128_caps1",
    ]
    assert "within-width repeat spread is much smaller" in protocol["endpoint_gate"]
    assert "all 34 layers" in protocol["endpoint_gate"]
    assert protocol["middle_width_sequence_after_gate"] == [
        "width2_caps1",
        "width64_caps1",
        "width4_caps1",
        "width32_caps1",
        "width8_caps1",
        "width16_caps1",
    ]
    assert protocol["stop_if_endpoint_gate_fails"] is True
    resolved = resolve_frozen_campaign_workload(
        LS4_VJP_INJECTED_WIDTH_SWEEP_V1,
        "ls4-vjp-injected-width-sweep-4b-361-dev-256",
    )
    assert len(resolved.prefix_token_ids) == 256


@pytest.mark.parametrize(
    ("profile_role", "session_width", "physical_cap"),
    [
        ("width128_caps1", 128, 1),
        ("width64_caps1", 64, 1),
        ("width32_caps1", 32, 1),
        ("width16_caps1", 16, 1),
        ("width8_caps1", 8, 1),
        ("width4_caps1", 4, 1),
        ("width2_caps1", 2, 1),
        ("width1_caps1", 1, 1),
    ],
)
def test_prepare_vjp_injected_width_sweep_pins_only_intended_axes(
    tmp_path: Path,
    profile_role: str,
    session_width: int,
    physical_cap: int,
) -> None:
    output_dir = tmp_path / profile_role
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(LS4_VJP_INJECTED_WIDTH_SWEEP_V1),
                "ls4-vjp-injected-width-sweep-4b-361-dev-256",
                "--profile-role",
                profile_role,
                "--execution-mode",
                "phase3-probe",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())
    knobs = trace_spec["graph_knobs"]
    assert knobs["diagnostic_stop_mode"] == "phase3_probe"
    assert knobs["diagnostic_stop_phase4_batches"] is None
    assert knobs["feature_row_influence_mode"] == "cuda_windowed"
    assert knobs["feature_row_influence_requirement"] == "required"
    assert knobs["backward_engine_mode"] == "duplicated_lanes"
    assert knobs["nnsight_session_capacity"] == session_width
    assert knobs["phase1_trace_batch_policy"] == "cap_effective_batches"
    assert knobs["phase1_trace_batch_size_max"] == physical_cap
    assert knobs["phase3_compute_microbatch_max_rows"] == physical_cap
    assert knobs["phase4_execution_batch_max_rows"] == physical_cap
    for capture in (
        "capture_phase0_donor_bundle",
        "capture_phase3_gradient_bundle",
        "capture_phase3_row_bundle",
        "capture_phase3_seed_bundle",
    ):
        assert knobs[capture] is True
    mechanism = launch_spec["mechanism_selections"][0]
    assert mechanism["backward_engine_mode"] == "duplicated_lanes"
    assert mechanism["forward_graph_mode"] == "logical_capacity"
    assert mechanism["vjp_kernel_mode"] == "nnsight_injected"
    assert mechanism["forward_lane_count"] == session_width


def test_ls5_12b_v2_length_ladder_is_frozen_sequential_and_two_hour_bounded() -> None:
    payload = json.loads(LS5_LENGTH_LADDER_V2.read_text())
    workloads = validate_performance_campaign(payload, require_frozen=True)
    assert [workload.requested_prefix_tokens for workload in workloads] == [
        129,
        256,
        512,
        1024,
    ]
    assert {workload.model_variant for workload in workloads} == {"gemma3_12b"}
    assert payload["execution_policy"]["ordered_workload_ids"] == [
        workload.workload_id for workload in workloads
    ]
    assert payload["execution_policy"]["advance_only_after_classification"] is True
    assert payload["execution_policy"]["scheduler_walltime_ceiling_seconds"] == 7200
    for index, raw_workload in enumerate(payload["workloads"]):
        assert raw_workload["resource_envelope"]["walltime_seconds"] <= 7200
        policy = raw_workload["comparison_policy"]
        assert policy["probe_before_full"] is True
        assert policy["required_candidate_mode"] == "cuda_windowed"
        assert policy["runtime_resource_policy"] == "measure_only"
        assert policy["advance_only_after_classification"] is True
        assert policy["full_trace_runtime_ceiling_seconds"] <= 7200
        if index:
            assert policy["predecessor_workload_id"] == workloads[index - 1].workload_id
        resolved = resolve_frozen_campaign_workload(
            LS5_LENGTH_LADDER_V2, raw_workload["workload_id"]
        )
        assert len(resolved.prefix_token_ids) == raw_workload["prefix"]["actual_tokens"]


def test_current_scaling_ladders_have_no_ceiling_over_two_hours() -> None:
    for manifest in (LS4_REMAINING_GATES_V2, LS5_LENGTH_LADDER_V2):
        payload = json.loads(manifest.read_text())
        assert payload["execution_policy"]["scheduler_walltime_ceiling_seconds"] <= 7200
        assert all(
            workload["resource_envelope"]["walltime_seconds"] <= 7200
            and workload["comparison_policy"]["full_trace_runtime_ceiling_seconds"]
            <= 7200
            for workload in payload["workloads"]
        )


def test_resolve_frozen_campaign_workload_rechecks_prefix_and_target() -> None:
    resolved = resolve_frozen_campaign_workload(CAMPAIGN, "ls0-361-dev-512")
    assert len(resolved.prefix_token_ids) == 512
    assert resolved.generated_index == 383
    assert resolved.workload["target"]["token_id"] == 33036
    assert resolved.trajectory_path.name == "361_trajectory.json"


def test_prepare_campaign_workload_uses_existing_full_answer_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "prepared"
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(CAMPAIGN),
                "ls0-361-dev-512",
                "--profile-role",
                "candidate",
                "--execution-mode",
                "transition-probe",
                "--probe-batches",
                "4",
                "--feature-row-influence-requirement",
                "required",
                "--correctness-probe-mode",
                "smoke",
                "--preheat-policy",
                "file_cache",
                "--preheat-path",
                str(CAMPAIGN),
                "--allocation-profile",
                "full-h200",
                "--allocation-partition",
                "rai-gpu-grn",
                "--allocation-gpus-per-task",
                "1",
                "--allocation-cpus-per-task",
                "32",
                "--allocation-mem",
                "600G",
                "--allocation-walltime",
                "08:00:00",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())
    shards = json.loads((output_dir / "shards.json").read_text())
    assert prepared["launcher"] == "existing_full_answer_shard_runner"
    assert prepared["profile_name"] == "sp5-selective-phase0-finalist-plt-v1"
    assert prepared["prefix_token_count"] == 512
    assert prepared["launch_command"][:4] == [
        "uv",
        "run",
        "exact-trace-bench",
        "run-full-answer-shard",
    ]
    assert trace_spec["prefix_token_count"] == 512
    assert trace_spec["target_token_id"] == 33036
    assert trace_spec["graph_knobs"]["phase0_decoder_row_ranges"] is True
    assert trace_spec["graph_knobs"]["diagnostic_stop_mode"] == "transition_probe"
    assert trace_spec["graph_knobs"]["diagnostic_stop_phase4_batches"] == 4
    assert trace_spec["graph_knobs"]["feature_row_influence_requirement"] == (
        "required"
    )
    assert trace_spec["graph_knobs"]["runtime_resource_policy"] == "measure_only"
    assert trace_spec["graph_knobs"]["correctness_probe_mode"] == "smoke"
    assert (
        trace_spec["graph_knobs"]["correctness_policy_id"]
        == "behavioral_closure_v1"
    )
    assert launch_spec["runtime_resource_policy"] == "measure_only"
    assert (
        launch_spec["mechanism_selections"][0]["feature_row_influence_requirement"]
        == "required"
    )
    assert launch_spec["mechanism_selections"][0]["correctness_probe_mode"] == "smoke"
    assert launch_spec["scheduler_request"] == {
        "account": None,
        "cluster": "granite",
        "cpus_per_task": 32,
        "gpus_per_task": 1,
        "memory": "600G",
        "partition": "rai-gpu-grn",
        "qos": None,
        "resource_profile": "full-h200",
        "walltime": "08:00:00",
    }
    assert launch_spec["preheat"]["policy"] == "file_cache"
    assert (
        launch_spec["selection_fingerprint"] == prepared["launch_selection_fingerprint"]
    )
    assert shards["shards"][0]["spec_indices"] == [0]
    assert "Launch: uv run exact-trace-bench run-full-answer-shard" in (
        capsys.readouterr().out
    )


def test_prepare_campaign_workload_accepts_registered_profile_override(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    profile_name = "ls2-1b-long-prefix-active-rows-1536mib-v1"
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(CAMPAIGN),
                "ls0-361-dev-1024",
                "--profile-role",
                "candidate",
                "--profile-name",
                profile_name,
                "--execution-mode",
                "transition-probe",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    assert prepared["profile_role"] == "candidate"
    assert prepared["profile_name"] == profile_name
    assert prepared["profile_selection_source"] == "explicit_override"
    assert (
        prepared["profile_contract"]["physical"]["decoder_active_row_max_bytes"]
        == 1536 * 1024**2
    )
    assert trace_spec["graph_knobs"]["decoder_active_row_max_bytes"] == 1536 * 1024**2


def test_prepare_batched_vjp_candidate_records_required_mode_and_single_forward_lane(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared-batched-vjp"
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(LS4_BATCHED_VJP_V1),
                "ls4-batched-vjp-4b-361-dev-256",
                "--profile-role",
                "candidate",
                "--execution-mode",
                "transition-probe",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    assert prepared["profile_name"] == (
        "ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"
    )
    assert trace_spec["graph_knobs"]["backward_engine_mode"] == (
        "single_forward_batched_vjp"
    )
    assert trace_spec["graph_knobs"]["nnsight_session_capacity"] == 128
    assert trace_spec["graph_knobs"]["diagnostic_stop_mode"] == "transition_probe"


def test_prepare_batched_vjp_self_consistency_repeats_differ_only_in_provenance(
    tmp_path: Path,
) -> None:
    prepared: dict[str, dict[str, object]] = {}
    trace_specs: dict[str, dict[str, object]] = {}
    launch_specs: dict[str, dict[str, object]] = {}
    for role in ("repeat_a", "repeat_b"):
        output_dir = tmp_path / role
        launch_output_root = tmp_path / "runs" / role
        assert (
            perf_cli.main(
                [
                    "prepare-campaign-workload",
                    str(LS4_BATCHED_VJP_SELF_CONSISTENCY_V1),
                    "ls4-batched-vjp-self-consistency-4b-361-dev-256",
                    "--profile-role",
                    role,
                    "--execution-mode",
                    "full",
                    "--output-dir",
                    str(output_dir),
                    "--launch-output-root",
                    str(launch_output_root),
                    "--run-id",
                    f"self-consistency-{role}",
                ]
            )
            == 0
        )
        prepared[role] = json.loads((output_dir / "prepared_workload.json").read_text())
        trace_specs[role] = json.loads((output_dir / "trace_specs.jsonl").read_text())
        launch_specs[role] = json.loads((output_dir / "launch_spec.json").read_text())

    assert trace_specs["repeat_a"] == trace_specs["repeat_b"]
    assert prepared["repeat_a"]["profile_name"] == prepared["repeat_b"]["profile_name"]
    assert (
        prepared["repeat_a"]["profile_contract"]
        == prepared["repeat_b"]["profile_contract"]
    )
    assert prepared["repeat_a"]["execution_mode"] == "full"
    assert prepared["repeat_b"]["execution_mode"] == "full"
    assert prepared["repeat_a"]["diagnostic_stop_mode"] == "none"
    assert prepared["repeat_b"]["diagnostic_stop_mode"] == "none"
    assert (
        prepared["repeat_a"]["launch_output_root"]
        != prepared["repeat_b"]["launch_output_root"]
    )

    for role in ("repeat_a", "repeat_b"):
        knobs = trace_specs[role]["graph_knobs"]
        assert knobs["feature_row_influence_mode"] == "cuda_windowed"
        assert knobs["feature_row_influence_requirement"] == "required"
        assert knobs["backward_engine_mode"] == "single_forward_batched_vjp"
        assert knobs["nnsight_session_capacity"] == 128
        assert knobs["phase1_trace_batch_size_max"] == 128
        assert knobs["phase3_compute_microbatch_max_rows"] == 128
        assert knobs["phase4_execution_batch_max_rows"] == 128
        assert knobs["runtime_resource_policy"] == "measure_only"
        mechanism = launch_specs[role]["mechanism_selections"][0]
        assert mechanism["backward_engine_mode"] == "single_forward_batched_vjp"
        assert mechanism["forward_graph_mode"] == "single_lane"
        assert mechanism["vjp_kernel_mode"] == "autograd_batched"
        assert mechanism["forward_lane_count"] == 1
        assert mechanism["backward_selection_source"] == "preset"
        assert mechanism["feature_row_influence_mode"] == "cuda_windowed"
        assert mechanism["feature_row_influence_requirement"] == "required"
        assert launch_specs[role]["runtime_resource_policy"] == "measure_only"

    assert launch_specs["repeat_a"]["run"]["run_id"] == "self-consistency-repeat_a"
    assert launch_specs["repeat_b"]["run"]["run_id"] == "self-consistency-repeat_b"


def test_prepare_batched_vjp_512_repeats_freezes_science_and_scheduler_contract(
    tmp_path: Path,
) -> None:
    prepared: dict[str, dict[str, object]] = {}
    trace_specs: dict[str, dict[str, object]] = {}
    launch_specs: dict[str, dict[str, object]] = {}
    for role in ("repeat_a", "repeat_b"):
        output_dir = tmp_path / role
        launch_output_root = tmp_path / "runs" / role
        assert (
            perf_cli.main(
                [
                    "prepare-campaign-workload",
                    str(LS4_BATCHED_VJP_SELF_CONSISTENCY_512_V1),
                    "ls4-batched-vjp-self-consistency-4b-361-dev-512",
                    "--profile-role",
                    role,
                    "--execution-mode",
                    "full",
                    "--output-dir",
                    str(output_dir),
                    "--launch-output-root",
                    str(launch_output_root),
                    "--run-id",
                    f"self-consistency-512-{role}",
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
                    "400G",
                    "--allocation-walltime",
                    "01:00:00",
                ]
            )
            == 0
        )
        prepared[role] = json.loads((output_dir / "prepared_workload.json").read_text())
        trace_specs[role] = json.loads((output_dir / "trace_specs.jsonl").read_text())
        launch_specs[role] = json.loads((output_dir / "launch_spec.json").read_text())

    assert trace_specs["repeat_a"] == trace_specs["repeat_b"]
    assert trace_specs["repeat_a"]["prefix_token_count"] == 512
    assert trace_specs["repeat_a"]["target_position"] == 512
    assert trace_specs["repeat_a"]["target_token_id"] == 33036

    prepared_difference_keys = {
        key
        for key in prepared["repeat_a"]
        if prepared["repeat_a"][key] != prepared["repeat_b"][key]
    }
    assert prepared_difference_keys == {
        "profile_role",
        "trace_specs_path",
        "shards_path",
        "launch_output_root",
        "launch_spec_path",
        "launch_selection_fingerprint",
        "launch_command",
    }

    launch_difference_keys = {
        key
        for key in launch_specs["repeat_a"]
        if launch_specs["repeat_a"][key] != launch_specs["repeat_b"][key]
    }
    assert launch_difference_keys == {
        "trace_specs_path",
        "shards_path",
        "output_root",
        "run",
        "selection_fingerprint",
    }
    assert {
        key
        for key in launch_specs["repeat_a"]["run"]
        if launch_specs["repeat_a"]["run"][key] != launch_specs["repeat_b"]["run"][key]
    } == {"run_id", "run_description"}

    expected_scheduler_request = {
        "account": "rai",
        "cluster": "granite",
        "cpus_per_task": 32,
        "gpus_per_task": 1,
        "memory": "400G",
        "partition": "rai-gpu-grn",
        "qos": "rai-gpu-grn",
        "resource_profile": "rai-gpu-grn",
        "walltime": "01:00:00",
    }
    for role in ("repeat_a", "repeat_b"):
        assert prepared[role]["profile_name"] == (
            "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
        )
        assert prepared[role]["execution_mode"] == "full"
        assert prepared[role]["diagnostic_stop_mode"] == "none"
        knobs = trace_specs[role]["graph_knobs"]
        assert knobs["feature_row_influence_mode"] == "cuda_windowed"
        assert knobs["feature_row_influence_requirement"] == "required"
        assert knobs["backward_engine_mode"] == "single_forward_batched_vjp"
        assert knobs["nnsight_session_capacity"] == 128
        assert knobs["phase1_trace_batch_size_max"] == 128
        assert knobs["phase3_compute_microbatch_max_rows"] == 128
        assert knobs["phase4_execution_batch_max_rows"] == 128
        assert knobs["runtime_resource_policy"] == "measure_only"
        mechanism = launch_specs[role]["mechanism_selections"][0]
        assert mechanism["forward_graph_mode"] == "single_lane"
        assert mechanism["vjp_kernel_mode"] == "autograd_batched"
        assert mechanism["forward_lane_count"] == 1
        assert launch_specs[role]["scheduler_request"] == (expected_scheduler_request)
        assert launch_specs[role]["runtime_resource_policy"] == "measure_only"


def test_prepare_batched_vjp_1024_freezes_mechanism_and_scheduler_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prepared"
    launch_output_root = tmp_path / "run"
    assert (
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(LS4_BATCHED_VJP_1024_V1),
                "ls4-batched-vjp-4b-361-dev-1024",
                "--profile-role",
                "run",
                "--execution-mode",
                "full",
                "--output-dir",
                str(output_dir),
                "--launch-output-root",
                str(launch_output_root),
                "--run-id",
                "batched-vjp-1024-feasibility",
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
                "400G",
                "--allocation-walltime",
                "01:00:00",
            ]
        )
        == 0
    )

    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
    launch_spec = json.loads((output_dir / "launch_spec.json").read_text())

    assert prepared["profile_role"] == "run"
    assert prepared["profile_name"] == (
        "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
    )
    assert prepared["execution_mode"] == "full"
    assert prepared["diagnostic_stop_mode"] == "none"
    assert trace_spec["prefix_token_count"] == 1024
    assert trace_spec["target_position"] == 1024
    assert trace_spec["target_token_id"] == 9338
    knobs = trace_spec["graph_knobs"]
    assert knobs["feature_row_influence_mode"] == "cuda_windowed"
    assert knobs["feature_row_influence_requirement"] == "required"
    assert knobs["backward_engine_mode"] == "single_forward_batched_vjp"
    assert knobs["nnsight_session_capacity"] == 128
    assert knobs["phase1_trace_batch_size_max"] == 128
    assert knobs["phase3_compute_microbatch_max_rows"] == 128
    assert knobs["phase4_execution_batch_max_rows"] == 128
    assert knobs["runtime_resource_policy"] == "measure_only"
    assert launch_spec["runtime_resource_policy"] == "measure_only"
    assert launch_spec["scheduler_request"] == {
        "account": "rai",
        "cluster": "granite",
        "cpus_per_task": 32,
        "gpus_per_task": 1,
        "memory": "400G",
        "partition": "rai-gpu-grn",
        "qos": "rai-gpu-grn",
        "resource_profile": "rai-gpu-grn",
        "walltime": "01:00:00",
    }
    assert launch_spec["planning_envelope"] == {
        "gpu": "NVIDIA H200",
        "gpu_count": 1,
        "host_memory_stop_gib": 400,
        "host_rss_stop_gib": 300,
        "hbm_peak_fraction_max": 0.9,
        "walltime_seconds": 3600,
    }
    assert len(launch_spec["mechanism_selections"]) == 1
    mechanism = launch_spec["mechanism_selections"][0]
    assert mechanism["backward_engine_mode"] == "single_forward_batched_vjp"
    assert mechanism["backward_selection_source"] == "preset"
    assert mechanism["feature_row_influence_mode"] == "cuda_windowed"
    assert mechanism["feature_row_influence_requirement"] == "required"
    assert mechanism["forward_graph_mode"] == "single_lane"
    assert mechanism["forward_lane_count"] == 1
    assert mechanism["vjp_kernel_mode"] == "autograd_batched"


def test_prepare_batched_vjp_1024_fails_closed_on_declared_arm_mismatch(
    tmp_path: Path,
) -> None:
    payload = json.loads(LS4_BATCHED_VJP_1024_V1.read_text())
    payload["workloads"][0]["comparison_policy"]["arms"]["run"]["physical_caps"][
        "phase3_compute_microbatch_max_rows"
    ] = 127
    manifest = tmp_path / "bad-1024-arm.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="physical_caps.phase3_compute_microbatch_max_rows",
    ):
        perf_cli.main(
            [
                "prepare-campaign-workload",
                str(manifest),
                "ls4-batched-vjp-4b-361-dev-1024",
                "--profile-role",
                "run",
                "--execution-mode",
                "full",
                "--output-dir",
                str(tmp_path / "prepared-bad-1024-arm"),
            ]
        )


def _write_canonical_prepared_bundle(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    trajectory = bundle / "trajectory.json"
    trace_specs = bundle / "trace_specs.jsonl"
    shards = bundle / "shards.json"
    output_root = tmp_path / "run"
    trajectory.write_text("{}", encoding="utf-8")
    trace_specs.write_text("{}\n", encoding="utf-8")
    shards.write_text("{}", encoding="utf-8")
    config = {
        "runtime_resource_policy": "enforce",
        "resource_planning_envelope": {
            "host_memory_stop_gib": 200,
            "host_rss_stop_gib": 64,
        },
    }
    launch = build_full_answer_launch_spec(
        trajectory_path=trajectory,
        trace_specs_path=trace_specs,
        shards_path=shards,
        output_root=output_root,
        shard_selection="0",
        run={"run_id": "prepared-test"},
        specs=[{"trace_id": "trace", "graph_knobs": config}],  # type: ignore[list-item]
        planning_envelope=config["resource_planning_envelope"],
        scheduler_request={},
        runtime_resource_policy="enforce",
        runtime_resource_override_rationale=None,
        preheat={"policy": "none", "paths": []},
        workspace={"policy": "test"},
        monitoring={},
    )
    launch_record = launch.to_record()
    launch_path = bundle / "launch_spec.json"
    launch_path.write_text(json.dumps(launch_record), encoding="utf-8")
    command = render_local_full_answer_command(launch)
    (bundle / "prepared_workload.json").write_text(
        json.dumps(
            {
                "launcher": "existing_full_answer_shard_runner",
                "launch_command": command,
                "launch_output_root": str(output_root),
                "launch_spec_path": str(launch_path),
                "launch_selection_fingerprint": launch_record["selection_fingerprint"],
                "execution_mode": "full",
            }
        ),
        encoding="utf-8",
    )
    return bundle, output_root, command


def test_run_prepared_campaign_workload_reuses_recorded_command_and_resource_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, output_root, command = _write_canonical_prepared_bundle(tmp_path)
    calls: list[tuple[object, ...]] = []

    def fake_stream_runner(recorded_command, **kwargs):
        calls.append((recorded_command, kwargs))
        return 0

    monkeypatch.setattr(perf_cli, "_stream_runner", fake_stream_runner)
    assert perf_cli.main(["run-prepared-campaign-workload", str(bundle)]) == 0
    assert calls == [
        (
            command,
            {
                "output_root": output_root,
                "host_memory_stop_gib": 200.0,
                "host_rss_stop_gib": 64.0,
            },
        )
    ]


def test_run_prepared_campaign_workload_accepts_bundle_and_validate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _output_root, _command = _write_canonical_prepared_bundle(tmp_path)

    def fail_stream_runner(*args, **kwargs):
        raise AssertionError("validation-only preflight must not launch")

    monkeypatch.setattr(perf_cli, "_stream_runner", fail_stream_runner)
    assert (
        perf_cli.main(
            ["run-prepared-campaign-workload", str(bundle), "--validate-only"]
        )
        == 0
    )


def test_run_prepared_campaign_workload_rejects_bundle_without_record(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="readable regular file"):
        perf_cli.main(["run-prepared-campaign-workload", str(bundle)])


def test_long_prefix_b128_profile_splits_only_physical_phase4_execution() -> None:
    profile = perf_cli.CANDIDATE_PROFILES[
        "ls2-1b-long-prefix-active-rows-1536mib-b128-v1"
    ]
    assert len(profile.variants) == 1
    physical = profile.variants[0].physical.as_overrides()
    assert physical["decoder_active_row_max_bytes"] == 1536 * 1024**2
    assert physical["phase4_execution_batch_max_rows"] == 128
    assert physical["nnsight_session_capacity"] == 256


def test_long_prefix_windowed_profile_composes_b128_safety_envelope() -> None:
    profile = perf_cli.CANDIDATE_PROFILES[
        "ls2-1b-long-prefix-cuda-windowed-512mib-b128-v1"
    ]
    assert len(profile.variants) == 1
    physical = profile.variants[0].physical.as_overrides()
    assert physical["decoder_active_row_max_bytes"] == 1536 * 1024**2
    assert physical["phase4_execution_batch_max_rows"] == 128
    assert physical["nnsight_session_capacity"] == 256
    assert physical["feature_row_influence_mode"] == "cuda_windowed"
    assert physical["feature_row_gpu_window_max_bytes"] == 512 * 1024**2
    assert physical["feature_row_gpu_resident_safety_margin_bytes"] == 16 * 1024**3
    assert profile.evidence_scope.bounded_baseline is perf_cli.BaselineScope.MECHANISM


def test_long_prefix_control_matches_physical_stack_with_canonical_phase0() -> None:
    profile = perf_cli.CANDIDATE_PROFILES[
        "ls2-1b-long-prefix-canonical-active-rows-1536mib-b128-v1"
    ]
    assert len(profile.variants) == 1
    physical = profile.variants[0].physical.as_overrides()
    assert physical["phase0_decoder_row_ranges"] is False
    assert physical["decoder_active_row_max_bytes"] == 1536 * 1024**2
    assert physical["phase4_execution_batch_max_rows"] == 128


def test_ls4_transfer_profiles_are_matched_and_4b_only() -> None:
    control = perf_cli.CANDIDATE_PROFILES["ls4-4b-transfer-cpu-exact-b128-v1"]
    candidate = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-transfer-cuda-windowed-512mib-b128-v1"
    ]
    batched_vjp = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"
    ]
    assert (
        len(control.variants)
        == len(candidate.variants)
        == len(batched_vjp.variants)
        == 1
    )
    assert control.variants[0].requires.minimum_layer_count == 27
    assert control.variants[0].requires.maximum_layer_count == 34
    control_physical = control.variants[0].physical.as_overrides()
    candidate_physical = candidate.variants[0].physical.as_overrides()
    assert control_physical["phase0_decoder_row_ranges"] is True
    assert control_physical["decoder_active_row_max_bytes"] == 4 * 1024**3
    assert control_physical["phase4_execution_batch_max_rows"] == 128
    assert control_physical["feature_row_influence_mode"] == "cpu_exact"
    assert candidate_physical == {
        **control_physical,
        "feature_row_influence_mode": "cuda_windowed",
        "feature_row_gpu_window_max_bytes": 512 * 1024**2,
        "feature_row_gpu_resident_safety_margin_bytes": 16 * 1024**3,
    }
    assert batched_vjp.variants[0].physical.as_overrides() == {
        **candidate_physical,
        "backward_engine_mode": "single_forward_batched_vjp",
    }
    assert candidate.evidence_scope.bounded_baseline is perf_cli.BaselineScope.MECHANISM
    assert (
        batched_vjp.evidence_scope.bounded_baseline is perf_cli.BaselineScope.MECHANISM
    )


def test_ls4_self_consistency_profile_freezes_required_width1_regime() -> None:
    source = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-transfer-cuda-windowed-512mib-b128-batched-vjp-v1"
    ]
    profile = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-self-consistency-cuda-windowed-width1-batched-vjp-v1"
    ]

    assert len(profile.variants) == 1
    source_physical = source.variants[0].physical.as_overrides()
    physical = profile.variants[0].physical.as_overrides()
    assert physical == {
        **source_physical,
        "feature_row_influence_requirement": "required",
    }
    assert physical["backward_engine_mode"] == "single_forward_batched_vjp"
    assert physical["nnsight_session_capacity"] == 128
    assert physical["phase1_trace_batch_size_max"] == 128
    assert physical["phase3_compute_microbatch_max_rows"] == 128
    assert physical["phase4_execution_batch_max_rows"] == 128
    assert profile.variants[0].baseline_pins == source.variants[0].baseline_pins
    assert profile.evidence_scope == source.evidence_scope


def test_frozen_workload_requires_immutable_hashes() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    payload["workloads"][0]["fixture"]["catalog_sha256"] = None
    with pytest.raises(ValueError, match="catalog_sha256"):
        validate_performance_campaign(payload)


def test_perf_cli_campaign_and_claim_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert perf_cli.main(["validate-claims", str(CLAIMS)]) == 0
    assert "validated mechanism claims" in capsys.readouterr().out
    assert perf_cli.main(["list-campaign", str(CAMPAIGN)]) == 0
    output = capsys.readouterr().out
    assert "ls0-361-length-scaling-v1: 8 workloads" in output
    assert "ls0-361-dev-1024" in output
    assert "ls0-828-holdout-512" in output
    assert "ls0-94-holdout-512" in output


@pytest.mark.parametrize("fixture", ["828", "94"])
def test_long_holdout_trajectory_extends_frozen_256_prefix(fixture: str) -> None:
    root = REPO_ROOT / "experiments/generated/performance_campaigns/ls0"
    short = json.loads((root / f"{fixture}_trajectory.json").read_text())
    long = json.loads((root / f"{fixture}_long_trajectory.json").read_text())
    assert long["prompt_token_ids"] == short["prompt_token_ids"]
    assert (
        long["generated_tokens"][: len(short["generated_tokens"])]
        == short["generated_tokens"]
    )
    assert len(long["prompt_token_ids"]) + len(long["generated_tokens"]) > 512


def test_freeze_campaign_derives_prefix_and_target_fingerprints(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    catalog_path = tmp_path / "catalog.json"
    trajectory_path = tmp_path / "trajectory.json"
    manifest_path = tmp_path / "campaign.json"
    prompt_path.write_text("prompt", encoding="utf-8")
    catalog_path.write_text('{"schema_version": 1}', encoding="utf-8")
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectory_id": "trajectory-v1",
                "prompt_token_count": 2,
                "prompt_token_ids": [10, 11],
                "prompt_text": "prompt",
                "generated_tokens": [
                    {
                        "generated_index": index,
                        "absolute_token_position": index + 2,
                        "token_id": 20 + index,
                        "token_text": f"t{index}",
                        "is_stop": False,
                    }
                    for index in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    workload = json.loads(CAMPAIGN.read_text())["workloads"][0]
    workload["workload_id"] = "freeze-test"
    workload["preparation_status"] = "planned"
    workload["fixture"] = {
        "catalog": "catalog.json",
        "catalog_sha256": None,
        "fixture_name": "test",
        "prompt_file": "prompt.txt",
        "prompt_sha256": None,
    }
    workload["trajectory"] = {
        "trajectory_id": "trajectory-v1",
        "path": "trajectory.json",
        "sha256": None,
    }
    workload["prefix"] = {
        "requested_tokens": 4,
        "actual_tokens": None,
        "token_ids_sha256": None,
    }
    workload["target"] = {
        "absolute_position": None,
        "token_id": None,
        "token_text": None,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "freeze-test-v1",
                "workloads": [workload],
            }
        ),
        encoding="utf-8",
    )

    frozen = freeze_performance_campaign(manifest_path, repo_root=tmp_path)

    frozen_workload = frozen["workloads"][0]
    assert frozen_workload["preparation_status"] == "frozen"
    assert frozen_workload["prefix"]["actual_tokens"] == 4
    assert len(frozen_workload["prefix"]["token_ids_sha256"]) == 64
    assert frozen_workload["target"] == {
        "absolute_position": 4,
        "token_id": 22,
        "token_text": "t2",
    }
    assert validate_performance_campaign(frozen, require_frozen=True)
