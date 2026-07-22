from __future__ import annotations

from types import SimpleNamespace

from nlp_research_project.exact_trace_bench.scenarios.governor_calibration import (
    GIB,
    GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
    GOVERNOR_CALIBRATION_PROVIDERS,
    GOVERNOR_CALIBRATION_RESOURCE_PROFILE,
    build_governor_calibration_config,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    _physical_requirements_from_scenario,
    trace_policy_from_scenario,
)
from circuit_tracer import AdmissionMode
from circuit_tracer.tracing.governor_bridge import _workload


def test_governor_calibration_wave_a_is_upward_and_complete() -> None:
    payloads = {
        variant: build_governor_calibration_config(variant=variant)
        for variant in GOVERNOR_CALIBRATION_PROVIDERS
        if variant in {"gemma3_1b_clt", "gemma3_1b_plt"}
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
        assert (
            "does not depend on array task order"
            in metadata["array_concurrency_reason"]
        )
        assert metadata["baseline_registry"] == str(
            GOVERNOR_CALIBRATION_BASELINE_REGISTRY
        )
        assert metadata["fail_on_baseline_missing"] is True
        assert metadata["fail_on_validation_fail"] is True
        assert metadata["resource_profile"] == GOVERNOR_CALIBRATION_RESOURCE_PROFILE
        assert metadata["fixed_protocol"]["hardware"] == "single_h200"
        assert metadata["slurm"]["qos"] == "rai-gpu-grn-short"
        assert metadata["slurm"]["cpus_per_task"] == 12
        assert payload["defaults"]["governor_profile_name"] == provider.governor_profile
        assert payload["defaults"]["governor_admission_mode"] == "advisory"
        assert metadata["governor_admission_policy"] == {
            "mode": "advisory",
            "scope": "wave_a_calibration_only",
            "reason": (
                "Deliberate force override so refused configurations still execute "
                "and produce calibration evidence; ordinary launches remain enforce."
            ),
        }
        assert "governor_admission_refusal" not in metadata["stop_rules"]["hard"]

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
        assert payload["scenarios"][0] is reference
        assert reference["governor_fidelity_mode"] == "exact"
        assert "governor_fidelity_override_fields" not in reference
        assert reference["attribution_batch_size"] == provider.reference_logical_batch
        assert (
            max(row["attribution_batch_size"] for row in cases.values())
            > provider.reference_logical_batch
        )

        for row in payload["scenarios"]:
            assert row["baseline_check"] == {
                "enabled": True,
                "mode": "metrics",
                "registry_key": provider.baseline_registry_key,
                "baseline_required": True,
            }

        for row in cases.values():
            policy = trace_policy_from_scenario({**payload["defaults"], **row})
            assert policy.governor_admission_mode is AdmissionMode.ADVISORY
            assert policy.governor_fidelity.mode.value == row["governor_fidelity_mode"]
            if row["governor_fidelity_mode"] == "exact":
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
    assert cases["decoder_chunk_c10080"]["governor_fidelity_mode"] == "exact"
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
    assert cases["decoder_chunk_c32768"]["governor_fidelity_mode"] == "exact"
    assert (
        _physical_requirements_from_scenario(cases["decoder_chunk_c32768"])
        .decoder_fetch_chunk_size
        == 32768
    )
    assert cases["semantic_refresh_g8"]["governor_fidelity_override_fields"] == [
        "frontier_refresh_stride"
    ]


def test_governor_calibration_wave_b_separates_semantics_and_execution() -> None:
    expected = {
        "gemma3_4b_plt": (128, 256, 512, "400G", "02:00:00"),
        "gemma3_12b_plt": (64, 128, 256, "600G", "08:00:00"),
    }

    for variant, (semantic_batch, midpoint, high, memory, walltime) in expected.items():
        payload = build_governor_calibration_config(variant=variant)
        metadata = payload["metadata"]
        rows = payload["scenarios"]
        cases = {row["calibration_case"]: row for row in rows}

        assert metadata["calibration_wave"] == "B"
        assert metadata["matrix_case_count"] == 7
        assert metadata["baseline_registry_pinned"] is True
        assert metadata["fixed_protocol"]["baseline_reference"] == (
            "pinned_corrected_hook_registry"
        )
        assert metadata["immutable_validation_config"] is True
        assert metadata["slurm"]["gres"] == "gpu:h200:1"
        assert metadata["slurm"]["qos"] == "rai-gpu-grn"
        assert metadata["slurm"]["mem"] == memory
        assert metadata["slurm"]["time"] == walltime
        assert "gpu_utilization_time_series" in metadata["required_measurements"]
        assert payload["defaults"]["attribution_batch_size"] == semantic_batch
        assert payload["defaults"]["feature_batch_size"] == semantic_batch
        assert payload["defaults"]["logit_batch_size"] == semantic_batch
        assert payload["defaults"]["attribution_update_interval"] == 4

        assert [row["calibration_case"] for row in rows] == [
            "reference",
            f"physical_envelope_b{midpoint}",
            f"physical_envelope_b{high}",
            "decoder_chunk_c16384",
            "decoder_chunk_c32768",
            f"coupled_execution_b{midpoint}_c32768",
            f"semantic_feature_b{semantic_batch * 2}_execution_b{midpoint}",
        ]

        for row in rows:
            assert row["attribution_batch_size"] == semantic_batch
            assert row["logit_batch_size"] == semantic_batch
            assert row["phase1_trace_batch_policy"] == "cap_effective_batches"
            assert row["phase1_trace_batch_size_max"] == semantic_batch
            assert row["phase3_compute_microbatch_max_rows"] == semantic_batch
            assert row["cross_batch_decoder_cache_bytes"] == 0
            assert row["timeout_minutes"] * 60 == row["governor_resource_envelope"][
                "walltime_seconds"
            ]
            assert row["baseline_check"] == {
                "enabled": True,
                "mode": "metrics",
                "registry_key": (
                    f"governor_calibration/{variant}/361_base"
                ),
                "baseline_required": True,
            }

        assert cases["reference"]["phase4_execution_batch_max_rows"] == semantic_batch
        for batch in (midpoint, high):
            row = cases[f"physical_envelope_b{batch}"]
            assert row["calibration_factor"] == "physical_envelope"
            assert row["nnsight_session_capacity"] == batch
            assert row["phase4_execution_batch_max_rows"] == batch


def test_governor_calibration_wave_b_research_rows_name_exact_overrides() -> None:
    for variant in ("gemma3_4b_plt", "gemma3_12b_plt"):
        payload = build_governor_calibration_config(variant=variant)
        rows = payload["scenarios"]
        decoder_rows = [row for row in rows if row["decoder_chunk_size"] != 4096]
        semantic_row = rows[-1]

        assert len(decoder_rows) == 3
        for row in decoder_rows:
            assert row["governor_fidelity_mode"] == "research"
            assert row["governor_fidelity_override_fields"] == [
                "decoder_reduction_tile"
            ]
            assert row["row_subchunk_size"] == row["decoder_chunk_size"]

            policy = trace_policy_from_scenario({**payload["defaults"], **row})
            request = policy.request(model=SimpleNamespace(), prompt=[1, 2])
            workload = _workload(request, policy.provider_profile, "wave-b-test")
            assert workload.research_overrides == (
                ("decoder_reduction_tile", str(row["decoder_chunk_size"])),
            )

        assert semantic_row["feature_batch_size"] == (
            payload["defaults"]["feature_batch_size"] * 2
        )
        assert semantic_row["attribution_update_interval"] == 2
        assert semantic_row["governor_fidelity_mode"] == "research"
        assert semantic_row["governor_fidelity_override_fields"] == [
            "feature_batch_size",
            "frontier_refresh_stride",
        ]
        assert (
            semantic_row["feature_batch_size"]
            * semantic_row["attribution_update_interval"]
            == payload["defaults"]["feature_batch_size"]
            * payload["defaults"]["attribution_update_interval"]
        )
        policy = trace_policy_from_scenario({**payload["defaults"], **semantic_row})
        request = policy.request(model=SimpleNamespace(), prompt=[1, 2])
        workload = _workload(request, policy.provider_profile, "wave-b-test")
        assert dict(workload.research_overrides) == {
            "feature_batch_size": str(semantic_row["feature_batch_size"]),
            "frontier_refresh_stride": "2",
        }

        for row in rows[:3]:
            assert row["governor_fidelity_mode"] == "exact"
            assert "governor_fidelity_override_fields" not in row
