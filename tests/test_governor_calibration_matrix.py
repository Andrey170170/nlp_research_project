from __future__ import annotations

from nlp_research_project.exact_trace_bench.scenarios.governor_calibration import (
    GIB,
    GOVERNOR_CALIBRATION_PROVIDERS,
    GOVERNOR_CALIBRATION_RESOURCE_PROFILE,
    build_governor_calibration_config,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    _physical_requirements_from_scenario,
    trace_policy_from_scenario,
)


def test_governor_calibration_wave_a_is_upward_and_complete() -> None:
    payloads = {
        variant: build_governor_calibration_config(variant=variant)
        for variant in GOVERNOR_CALIBRATION_PROVIDERS
    }

    assert set(payloads) == {"gemma3_1b_clt", "gemma3_1b_plt"}
    assert sum(len(payload["scenarios"]) for payload in payloads.values()) == 36
    assert len(payloads["gemma3_1b_clt"]["scenarios"]) == 17
    assert len(payloads["gemma3_1b_plt"]["scenarios"]) == 19

    for variant, payload in payloads.items():
        provider = GOVERNOR_CALIBRATION_PROVIDERS[variant]
        metadata = payload["metadata"]
        assert metadata["immutable_validation_config"] is True
        assert metadata["calibration_wave"] == "A"
        assert metadata["array_concurrency"] == 1
        assert metadata["resource_profile"] == GOVERNOR_CALIBRATION_RESOURCE_PROFILE
        assert metadata["fixed_protocol"]["hardware"] == "single_h200"
        assert metadata["slurm"]["qos"] == "rai-gpu-grn-short"
        assert metadata["slurm"]["cpus_per_task"] == 12
        assert payload["defaults"]["governor_profile_name"] == provider.governor_profile

        cases = {row["calibration_case"]: row for row in payload["scenarios"]}
        assert len(cases) == len(payload["scenarios"])
        assert {"reference", "reference_repeat"} <= cases.keys()
        assert all("governor_resource_envelope" in row for row in cases.values())
        assert all(
            row["governor_resource_envelope"]["walltime_seconds"]
            == row["timeout_minutes"] * 60
            for row in cases.values()
        )

        reference = cases["reference"]
        assert reference["governor_fidelity_mode"] == "strict"
        assert "governor_fidelity_override_fields" not in reference
        assert reference["attribution_batch_size"] == provider.reference_logical_batch
        assert max(
            row["attribution_batch_size"] for row in cases.values()
        ) > provider.reference_logical_batch

        for row in cases.values():
            policy = trace_policy_from_scenario({**payload["defaults"], **row})
            assert policy.governor_fidelity.mode.value == row["governor_fidelity_mode"]
            if row["governor_fidelity_mode"] == "strict":
                assert "governor_fidelity_override_fields" not in row
            else:
                assert row["governor_fidelity_mode"] == "research"
                assert row["governor_fidelity_override_fields"]


def test_clt_wave_a_covers_high_batches_chunks_caches_and_semantic_axes() -> None:
    payload = build_governor_calibration_config(variant="gemma3_1b_clt")
    cases = {row["calibration_case"]: row for row in payload["scenarios"]}

    assert {
        "logical_batch_b1500",
        "logical_batch_b2000",
        "logical_batch_b3000",
        "logical_batch_b4096",
        "decoder_chunk_c8192",
        "decoder_chunk_c10080",
        "decoder_cache_g0",
        "decoder_cache_g16",
        "decoder_cache_g32",
        "coupled_b2000_c8192_cacheg16",
        "coupled_b3000_c8192_cacheg32",
        "coupled_b4096_c10080_cacheg32",
        "semantic_source_b1500",
        "semantic_feature_b1500",
        "semantic_logit_b1500",
    } <= cases.keys()
    assert cases["decoder_chunk_c10080"]["governor_fidelity_mode"] == "strict"
    assert (
        _physical_requirements_from_scenario(cases["decoder_chunk_c10080"])
        .decoder_fetch_chunk_size
        == 10080
    )
    assert cases["decoder_cache_g32"]["cross_batch_decoder_cache_bytes"] == 32 * GIB
    assert cases["semantic_source_b1500"]["governor_fidelity_override_fields"] == [
        "source_batch_size"
    ]


def test_plt_wave_a_covers_high_batches_chunks_cache_and_refresh_research() -> None:
    payload = build_governor_calibration_config(variant="gemma3_1b_plt")
    cases = {row["calibration_case"]: row for row in payload["scenarios"]}

    assert {
        "logical_batch_b256",
        "logical_batch_b512",
        "logical_batch_b768",
        "logical_batch_b1024",
        "logical_batch_b1536",
        "decoder_chunk_c8192",
        "decoder_chunk_c16384",
        "decoder_chunk_c32768",
        "decoder_cache_g4",
        "coupled_b512_c8192_cacheg0",
        "coupled_b1024_c16384_cacheg0",
        "coupled_b1536_c32768_cacheg0",
        "semantic_source_b256",
        "semantic_feature_b256",
        "semantic_logit_b256",
        "semantic_refresh_g2",
        "semantic_refresh_g8",
    } <= cases.keys()
    assert cases["decoder_chunk_c32768"]["governor_fidelity_mode"] == "strict"
    assert (
        _physical_requirements_from_scenario(cases["decoder_chunk_c32768"])
        .decoder_fetch_chunk_size
        == 32768
    )
    assert cases["semantic_refresh_g8"]["governor_fidelity_override_fields"] == [
        "frontier_refresh_stride"
    ]
