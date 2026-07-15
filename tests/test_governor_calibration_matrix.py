from __future__ import annotations

from nlp_research_project.exact_trace_bench.scenarios.governor_calibration import (
    GOVERNOR_CALIBRATION_PROVIDERS,
    build_governor_calibration_config,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    _physical_requirements_from_scenario,
)


def test_governor_calibration_matrix_is_causal_and_complete() -> None:
    payloads = {
        variant: build_governor_calibration_config(variant=variant)
        for variant in GOVERNOR_CALIBRATION_PROVIDERS
    }

    assert sum(len(payload["scenarios"]) for payload in payloads.values()) == 38
    assert len(payloads["gemma3_1b_clt"]["scenarios"]) == 15
    assert len(payloads["gemma3_1b_plt"]["scenarios"]) == 15
    assert len(payloads["gemma3_4b_plt"]["scenarios"]) == 4
    assert len(payloads["gemma3_12b_plt"]["scenarios"]) == 4

    for variant, payload in payloads.items():
        assert payload["metadata"]["immutable_validation_config"] is True
        assert payload["metadata"]["fixed_protocol"]["hardware"] == "single_h200"
        cases = {row["calibration_case"]: row for row in payload["scenarios"]}
        assert {"reference", "session_reference", "lower_session"} <= cases.keys()
        reference = cases["reference"]
        session_reference = cases["session_reference"]
        lower = cases["lower_session"]
        assert lower["nnsight_session_capacity"] < reference["nnsight_session_capacity"]
        assert lower["phase3_compute_microbatch_max_rows"] == session_reference[
            "phase3_compute_microbatch_max_rows"
        ]
        assert lower["phase4_compute_microbatch_max_rows"] == session_reference[
            "phase4_compute_microbatch_max_rows"
        ]
        assert lower["phase1_trace_batch_size_max"] == session_reference[
            "phase1_trace_batch_size_max"
        ]
        assert payload["defaults"]["governor_profile_name"] == (
            GOVERNOR_CALIBRATION_PROVIDERS[variant].governor_profile
        )
        assert all("governor_resource_envelope" in row for row in payload["scenarios"])

        if variant.startswith("gemma3_1b_"):
            assert {
                "lower_phase3_microbatch",
                "lower_phase4_microbatch",
                "lower_phase1_batch",
                "decoder_cache_contrast",
                "replay_window_contrast",
                "prefetch_contrast",
                "recompute_cache_zero",
                "recompute_cache_reference",
                "tiled_policy",
                "eager_encoder_residency",
                "scratch_spill",
                "reference_repeat",
            } <= cases.keys()
            assert cases["lower_phase3_microbatch"][
                "phase4_compute_microbatch_max_rows"
            ] == reference["phase4_compute_microbatch_max_rows"]
            assert cases["lower_phase4_microbatch"][
                "phase3_compute_microbatch_max_rows"
            ] == reference["phase3_compute_microbatch_max_rows"]
            assert cases["recompute_cache_zero"][
                "governor_required_replay_tile_cache_bytes"
            ] == 0
            assert cases["recompute_cache_reference"][
                "governor_required_replay_tile_cache_bytes"
            ] > 0
            assert cases["prefetch_contrast"]["error_vector_prefetch_lookahead"] == 0
            assert cases["scratch_spill"]["governor_required_spill_target"] == "scratch"
            assert (
                _physical_requirements_from_scenario(cases["scratch_spill"])
                .spill_target.value
                == "scratch"
            )
        else:
            assert cases["tiled_policy"]["governor_required_row_store_policy"] == "tiled"
