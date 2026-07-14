from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nlp_research_project.exact_trace_bench.trace_runtime import tracing
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
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
        feature_row_retention="full_file",
        full_retention_backend="full_file",
    )

    policy = trace_policy_from_scenario(scenario)

    assert policy.physical_requirements is not None
    assert policy.physical_requirements.decoder_fetch_chunk_size == 4096
    assert policy.physical_requirements.decoder_cache_bytes == 0
    assert policy.physical_requirements.row_store_policy.value == "file_backed_full"


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
    assert len(payload["scenarios"]) == 3
    assert {
        scenario["governor_expected_row_store_policy"]
        for scenario in payload["scenarios"]
    } == {"file_backed_full", "tiled", "recompute"}

    governed_keys = {
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
