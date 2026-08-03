from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench import perf_cli
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
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    prepared = json.loads((output_dir / "prepared_workload.json").read_text())
    trace_spec = json.loads((output_dir / "trace_specs.jsonl").read_text())
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
    assert prepared["profile_contract"]["physical"][
        "decoder_active_row_max_bytes"
    ] == 1536 * 1024**2
    assert trace_spec["graph_knobs"]["decoder_active_row_max_bytes"] == 1536 * 1024**2


def test_run_prepared_campaign_workload_reuses_recorded_command_and_resource_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    prepared_path = tmp_path / "prepared_workload.json"
    command = [
        "uv",
        "run",
        "exact-trace-bench",
        "run-full-answer-shard",
        "--output-root",
        str(output_root),
    ]
    prepared_path.write_text(
        json.dumps(
            {
                "launcher": "existing_full_answer_shard_runner",
                "launch_command": command,
                "launch_output_root": str(output_root),
                "resource_envelope": {
                    "host_memory_stop_gib": 200,
                    "host_rss_stop_gib": 64,
                },
            }
        )
    )
    calls: list[tuple[object, ...]] = []

    def fake_stream_runner(recorded_command, **kwargs):
        calls.append((recorded_command, kwargs))
        return 0

    monkeypatch.setattr(perf_cli, "_stream_runner", fake_stream_runner)
    assert (
        perf_cli.main(
            ["run-prepared-campaign-workload", str(prepared_path)]
        )
        == 0
    )
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
    control = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-transfer-cpu-exact-b128-v1"
    ]
    candidate = perf_cli.CANDIDATE_PROFILES[
        "ls4-4b-transfer-cuda-windowed-512mib-b128-v1"
    ]
    assert len(control.variants) == len(candidate.variants) == 1
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
    assert (
        candidate.evidence_scope.bounded_baseline
        is perf_cli.BaselineScope.MECHANISM
    )


def test_frozen_workload_requires_immutable_hashes() -> None:
    payload = json.loads(CAMPAIGN.read_text())
    payload["workloads"][0]["fixture"]["catalog_sha256"] = None
    with pytest.raises(ValueError, match="catalog_sha256"):
        validate_performance_campaign(payload)


def test_perf_cli_campaign_and_claim_commands(capsys: pytest.CaptureFixture[str]) -> None:
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
    assert long["generated_tokens"][: len(short["generated_tokens"])] == short[
        "generated_tokens"
    ]
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
