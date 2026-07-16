from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from circuit_tracer import FidelityMode

from nlp_research_project.exact_trace_bench.config import base_trace_defaults
from nlp_research_project.exact_trace_bench.trace_runtime import tracing
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    _cap_walltime_to_slurm,
    trace_policy_from_scenario,
)


GIB = 1024**3
PROFILE = "granite_h200_1b_clt_b1000_c4096_cache8"
ROOT = Path(__file__).resolve().parents[1]


def _governed_scenario() -> dict[str, object]:
    return {
        "name": "phase-e-test",
        "method": "exact",
        "attribution_batch_size": 1000,
        "governor_profile_name": PROFILE,
        "governor_resource_envelope": {
            "total_vram_bytes": 141 * GIB,
            "vram_fraction": 0.9,
            "host_budget_bytes": 200 * GIB,
            "file_cache_allowance_bytes": 64 * GIB,
            "local_disk_bytes": 4 * GIB,
            "scratch_disk_bytes": 4 * GIB,
            "walltime_seconds": 7200,
        },
    }


def test_governed_policy_resolves_named_profile_and_envelope() -> None:
    policy = trace_policy_from_scenario(_governed_scenario())

    assert policy.provider_profile is not None
    assert policy.provider_profile.profile_name == PROFILE
    assert policy.resources is not None
    assert policy.resources.effective_vram_budget_bytes == int(141 * GIB * 0.9)
    assert policy.resources.host_budget_bytes == 200 * GIB
    assert policy.evidence.metadata["governor_profile_name"] == PROFILE
    assert policy.governor_fidelity.mode is FidelityMode.STRICT


def test_governor_walltime_is_capped_to_live_slurm_remainder() -> None:
    envelope = {"walltime_seconds": 7200}

    capped = _cap_walltime_to_slurm(
        envelope,
        environ={"SLURM_JOB_END_TIME": "10000"},
        now_seconds=6000,
        reserve_seconds=120,
    )

    assert capped["walltime_seconds"] == 3880
    assert envelope["walltime_seconds"] == 3880


def test_governor_walltime_is_unchanged_outside_slurm() -> None:
    envelope = {"walltime_seconds": 7200}

    assert _cap_walltime_to_slurm(envelope, environ={}) == envelope


def test_governed_policy_forwards_explicit_research_authorization() -> None:
    scenario = _governed_scenario()
    scenario.update(
        governor_fidelity_mode="research",
        governor_fidelity_override_fields=["source_batch_size", "feature_batch_size"],
    )

    policy = trace_policy_from_scenario(scenario)
    request = policy.request(model=SimpleNamespace(), prompt=[1, 2])

    assert policy.governor_fidelity.mode is FidelityMode.RESEARCH
    assert policy.governor_fidelity.override_fields == (
        "feature_batch_size",
        "source_batch_size",
    )
    assert request.governor_fidelity is policy.governor_fidelity


@pytest.mark.parametrize(
    "fidelity",
    [
        {"governor_fidelity_mode": "research"},
        {
            "governor_fidelity_mode": "strict",
            "governor_fidelity_override_fields": ["source_batch_size"],
        },
        {
            "governor_fidelity_mode": "validated_relaxed",
            "governor_fidelity_override_fields": ["source_batch_size"],
        },
    ],
)
def test_governed_policy_rejects_incomplete_fidelity_authorization(
    fidelity: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        trace_policy_from_scenario({**_governed_scenario(), **fidelity})


@pytest.mark.parametrize(
    "scenario",
    [
        {"governor_profile_name": PROFILE},
        {"governor_resource_envelope": {}},
        {
            "governor_profile_name": "not-recorded",
            "governor_resource_envelope": {},
        },
    ],
)
def test_governed_policy_fails_closed_on_incomplete_or_unknown_inputs(
    scenario: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        trace_policy_from_scenario(scenario)


def test_trace_adapter_forwards_governor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = trace_policy_from_scenario(_governed_scenario())
    captured: dict[str, object] = {}

    def fake_trace_one(request, *, resources, provider_profile):
        captured.update(
            request=request,
            resources=resources,
            provider_profile=provider_profile,
        )
        return SimpleNamespace(
            output={},
            status=SimpleNamespace(value="succeeded"),
            semantic_fingerprint="semantic",
            execution_fingerprint="execution",
            telemetry_summary={},
            telemetry_events=(),
            admission_report=None,
        )

    monkeypatch.setattr(tracing, "trace_one", fake_trace_one)
    result = tracing.extract_compact_chunked_attribution(
        SimpleNamespace(),
        [1, 2],
        policy=policy,
    )

    assert captured["resources"] is policy.resources
    assert captured["provider_profile"] is policy.provider_profile
    assert result["semantic_fingerprint"] == "semantic"


def test_trace_adapter_surfaces_governor_refusal_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = trace_policy_from_scenario(_governed_scenario())

    def fake_trace_one(request, *, resources, provider_profile):
        del request, resources, provider_profile
        return SimpleNamespace(
            output=None,
            status=SimpleNamespace(value="refused"),
            admission_report=SimpleNamespace(refusals=("host budget exceeded",)),
        )

    monkeypatch.setattr(tracing, "trace_one", fake_trace_one)

    with pytest.raises(RuntimeError, match="host budget exceeded"):
        tracing.extract_compact_chunked_attribution(
            SimpleNamespace(),
            [1, 2],
            policy=policy,
        )


def test_explicit_policy_keeps_governor_disabled() -> None:
    policy = trace_policy_from_scenario({"name": "explicit", "method": "exact"})

    assert policy.resources is None
    assert policy.provider_profile is None
    assert policy.physical_requirements is None


def test_governed_policy_preserves_explicit_zero_and_full_requirements() -> None:
    scenario = _governed_scenario()
    scenario.update(
        decoder_chunk_size=4096,
        cross_batch_decoder_cache_bytes=0,
        governor_required_row_store_policy="file_backed_full",
    )

    policy = trace_policy_from_scenario(scenario)

    assert policy.physical_requirements is not None
    assert policy.physical_requirements.decoder_fetch_chunk_size == 4096
    assert policy.physical_requirements.decoder_cache_bytes == 0
    assert policy.physical_requirements.row_store_policy.value == "file_backed_full"


def test_governed_policy_accepts_a_row_store_constraint_without_legacy_knobs() -> None:
    scenario = _governed_scenario()
    scenario["governor_required_row_store_policy"] = "recompute"
    scenario["governor_required_replay_tile_cache_bytes"] = 4 * GIB

    policy = trace_policy_from_scenario(scenario)

    assert policy.physical_requirements is not None
    assert policy.physical_requirements.row_store_policy.value == "recompute"
    assert policy.physical_requirements.replay_tile_cache_bytes == 4 * GIB
    assert policy.physical_requirements.session_capacity is None
    assert policy.physical_requirements.feature_microbatch_size is None
    assert policy.physical_requirements.logit_microbatch_size is None


def test_governed_policy_ignores_inherited_optional_and_storage_defaults() -> None:
    scenario = {**base_trace_defaults(), **_governed_scenario()}

    policy = trace_policy_from_scenario(scenario)

    assert policy.physical_requirements is not None
    assert policy.physical_requirements.session_capacity is None
    assert policy.physical_requirements.feature_microbatch_size is None
    assert policy.physical_requirements.logit_microbatch_size is None
    assert policy.physical_requirements.replay_window is None
    assert policy.physical_requirements.prefetch_depth is None
    assert policy.physical_requirements.row_store_policy is None
    assert policy.physical_requirements.encoder_residency is None


@pytest.mark.parametrize("architecture", ["clt", "plt"])
def test_phase_e_gate_scenarios_leave_governed_mechanisms_unpinned(
    architecture: str,
) -> None:
    path = (
        ROOT
        / "experiments"
        / "generated"
        / "exact_trace_bench"
        / f"exact_trace_phase_e_governor_gemma3_1b_{architecture}_granite_scenarios.json"
    )
    payload = json.loads(path.read_text())
    assert payload["metadata"]["immutable_validation_config"] is True
    assert len(payload["scenarios"]) == 1
    auto = payload["scenarios"][0]
    assert auto["governor_expected_row_store_policy"] == "file_backed_full"
    assert auto["run_name"].endswith("auto")
    assert "governor_required_row_store_policy" not in auto

    governed_keys = {
        "cross_batch_decoder_cache_bytes",
        "nnsight_session_capacity",
        "phase1_trace_batch_policy",
        "phase1_trace_batch_size_max",
        "phase3_compute_microbatch_max_rows",
        "phase4_compute_microbatch_max_rows",
        "full_retention_backend",
        "feature_row_column_tile_size",
        "feature_row_retention",
        "replay_tile_cache_bytes",
        "exact_encoder_residency",
    }
    for scope in [payload["defaults"], *payload["scenarios"]]:
        assert governed_keys.isdisjoint(scope)
