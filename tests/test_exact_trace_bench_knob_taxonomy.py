from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exact_trace_bench.scenarios import (  # noqa: E402
    ADVANCED_PUBLIC_TUNING_KEYS,
    DEBUG_REPLAY_PUBLIC_KEYS,
    DEPRECATED_COMPAT_KEYS,
    EXACT_MODE_KNOB_KEYS,
    SCENARIO_TIERS,
    STABLE_PUBLIC_SCENARIO_KEYS,
    TELEMETRY_KEYS,
    WAVE0_CANONICAL_ANOMALY_FIXTURES,
    WAVE0_CANONICAL_FAST_FIXTURES,
    WAVE0_CANONICAL_LATE_FIXTURES,
    WAVE0_NEW_BASE_FIXTURES,
    WAVE0_NEW_LATE_FIXTURES,
    WAVE0_REPEAT_COUNT,
    build_tier_config,
    build_wave0_baseline_config,
    build_wave3_interaction_confirmation_config,
)
from experiments.run_sparsification_experiment import build_command  # noqa: E402


CLUSTERS = ("ascend", "cardinal")

DISALLOWED_CANONICAL_COMMAND_FLAGS = {
    "--cross-cluster-debug",
    "--capture-phase0-donor-bundle",
    "--phase0-donor-bundle",
    "--phase0-replay-mode",
    "--phase0-donor-context-policy",
    "--phase3-gradient-donor-bundle",
    "--phase3-gradient-replay-mode",
    "--phase3-row-donor-bundle",
    "--phase3-row-replay-mode",
    "--phase3-replay-validation-policy",
    "--capture-phase3-seed-bundle",
    "--capture-phase3-gradient-bundle",
    "--capture-phase3-row-bundle",
    "--capture-feature-semantic-descriptors",
    "--semantic-descriptor-top-k",
    "--semantic-descriptor-dim",
    "--phase4-anomaly-debug",
    "--phase4-scheduler-debug",
    "--telemetry-max-events",
}


def _canonical_payloads() -> list[tuple[str, str, dict]]:
    return [
        (tier, cluster, build_tier_config(tier=tier, cluster=cluster))
        for tier in SCENARIO_TIERS
        for cluster in CLUSTERS
    ]


def _merged_scenarios(payload: dict) -> list[dict]:
    return [payload["defaults"] | scenario for scenario in payload["scenarios"]]


def _fake_wave0_catalog() -> dict[str, dict]:
    fixture_names = (
        WAVE0_CANONICAL_FAST_FIXTURES
        + WAVE0_CANONICAL_ANOMALY_FIXTURES
        + WAVE0_CANONICAL_LATE_FIXTURES
        + WAVE0_NEW_BASE_FIXTURES
        + WAVE0_NEW_LATE_FIXTURES
    )
    catalog = {}
    for fixture_name in fixture_names:
        index_str, fixture_suffix = fixture_name.split("_", maxsplit=1)
        fixture_kind = "late_prefix" if fixture_suffix == "late" else "base"
        catalog[fixture_name] = {
            "fixture_name": fixture_name,
            "fixture_kind": fixture_kind,
            "gsm8k_index": int(index_str),
            "prepared_prompt_file": f"/tmp/wave0/{fixture_name}/prompt.txt",
            "prepared_prompt_meta_file": f"/tmp/wave0/{fixture_name}/fixture_meta.json",
        }
    return catalog


def test_exact_mode_knobs_are_classified_without_duplicates() -> None:
    classified_keys = (
        STABLE_PUBLIC_SCENARIO_KEYS
        + ADVANCED_PUBLIC_TUNING_KEYS
        + DEBUG_REPLAY_PUBLIC_KEYS
        + TELEMETRY_KEYS
        + DEPRECATED_COMPAT_KEYS
    )

    assert EXACT_MODE_KNOB_KEYS == classified_keys
    assert len(EXACT_MODE_KNOB_KEYS) == len(set(EXACT_MODE_KNOB_KEYS))

    assert "exact_trace_internal_dtype" in STABLE_PUBLIC_SCENARIO_KEYS
    assert "phase4_scheduler_mode" in ADVANCED_PUBLIC_TUNING_KEYS
    assert "phase3_row_replay_mode" in DEBUG_REPLAY_PUBLIC_KEYS
    assert "telemetry_max_events" in TELEMETRY_KEYS
    assert "auto_scale_feature_batch_size" in DEPRECATED_COMPAT_KEYS


def test_canonical_exact_bench_defaults_are_stable() -> None:
    for _tier, _cluster, payload in _canonical_payloads():
        defaults = payload["defaults"]
        assert defaults["exact_trace_internal_dtype"] == "fp32"
        assert defaults["phase0_activation_threshold_compare_mode"] == "baseline"
        assert defaults["phase4_anomaly_debug"] is False
        assert defaults["cross_cluster_debug"] is False
        assert defaults["phase4_scheduler_mode"] == "locality"
        assert defaults["phase4_scheduler_debug"] is False
        assert defaults["phase4_scheduler_telemetry_detail"] == "normal"
        assert defaults["phase4_refresh_optimization"] == "off"
        assert defaults["phase4_row_executor"] == "batched"
        assert defaults["telemetry_max_events"] is None


def test_canonical_scenario_rows_keep_resource_knobs_explicit() -> None:
    for _tier, _cluster, payload in _canonical_payloads():
        for scenario in payload["scenarios"]:
            assert "decoder_chunk_size" in scenario
            assert "cross_batch_decoder_cache_bytes" in scenario
            assert scenario["decoder_chunk_size"] > 0
            assert scenario["cross_batch_decoder_cache_bytes"] >= 0


def test_only_long_eval_cache_probe_scenarios_enable_cross_batch_cache() -> None:
    for tier, cluster, payload in _canonical_payloads():
        for scenario in payload["scenarios"]:
            cache_bytes = scenario["cross_batch_decoder_cache_bytes"]
            if cache_bytes > 0:
                assert tier == "long_eval"
                assert scenario["long_eval_group"] == "cache_probe"
                assert scenario["fixture_name"] == "361_late"
                assert scenario["name"].startswith(f"{cluster}_long_eval_cache_probe")


def test_canonical_commands_do_not_enable_debug_or_replay_knobs() -> None:
    for _tier, _cluster, payload in _canonical_payloads():
        for scenario in _merged_scenarios(payload):
            command = build_command(Path("/tmp/exact-bench-taxonomy"), scenario)
            command_flags = {part for part in command if part.startswith("--")}
            assert command_flags.isdisjoint(DISALLOWED_CANONICAL_COMMAND_FLAGS)

            assert "--exact-trace-internal-dtype" in command
            dtype_index = command.index("--exact-trace-internal-dtype")
            assert command[dtype_index + 1] == "fp32"

            assert "--decoder-chunk-size" in command
            assert "--cross-batch-decoder-cache-bytes" in command


def test_wave0_baseline_scenario_counts_and_tiers() -> None:
    catalog = _fake_wave0_catalog()
    fast = build_wave0_baseline_config(
        tier="fast",
        cluster="ascend",
        catalog_by_name=catalog,
    )
    anomaly = build_wave0_baseline_config(
        tier="anomaly",
        cluster="ascend",
        catalog_by_name=catalog,
    )
    long_eval = build_wave0_baseline_config(
        tier="long_eval",
        cluster="ascend",
        catalog_by_name=catalog,
    )

    assert fast["metadata"]["tier"] == "fast"
    assert anomaly["metadata"]["tier"] == "anomaly"
    assert long_eval["metadata"]["tier"] == "long_eval"
    assert long_eval["metadata"]["resource_profile"] == "long_eval_high_mem"

    assert len(fast["scenarios"]) == (
        len(WAVE0_CANONICAL_FAST_FIXTURES) * WAVE0_REPEAT_COUNT
        + len(WAVE0_NEW_BASE_FIXTURES)
    )
    assert len(anomaly["scenarios"]) == (
        len(WAVE0_CANONICAL_ANOMALY_FIXTURES) * WAVE0_REPEAT_COUNT
    )
    assert len(long_eval["scenarios"]) == (
        len(WAVE0_CANONICAL_LATE_FIXTURES) + len(WAVE0_NEW_LATE_FIXTURES)
    )

    assert {scenario["wave0_role"] for scenario in fast["scenarios"]} == {
        "canonical_repeat",
        "expanded_base",
    }
    assert {scenario["wave0_role"] for scenario in long_eval["scenarios"]} == {
        "canonical_late",
        "expanded_late",
    }


def test_wave0_scenarios_require_explicit_catalog_entries() -> None:
    catalog = _fake_wave0_catalog()
    catalog.pop(WAVE0_NEW_BASE_FIXTURES[0])

    try:
        build_wave0_baseline_config(
            tier="fast",
            cluster="ascend",
            catalog_by_name=catalog,
        )
    except KeyError as exc:
        assert WAVE0_NEW_BASE_FIXTURES[0] in str(exc)
    else:
        raise AssertionError("missing Wave 0 fixture did not fail")


def test_wave0_commands_do_not_enable_debug_or_replay_knobs() -> None:
    catalog = _fake_wave0_catalog()
    for tier in SCENARIO_TIERS:
        payload = build_wave0_baseline_config(
            tier=tier,
            cluster="cardinal",
            catalog_by_name=catalog,
        )
        for scenario in _merged_scenarios(payload):
            command = build_command(Path("/tmp/exact-bench-wave0"), scenario)
            command_flags = {part for part in command if part.startswith("--")}
            assert command_flags.isdisjoint(DISALLOWED_CANONICAL_COMMAND_FLAGS)
            assert "--exact-trace-internal-dtype" in command
            assert "--decoder-chunk-size" in command
            assert "--cross-batch-decoder-cache-bytes" in command


def test_wave3_interaction_scenarios_default_and_optional_variants() -> None:
    catalog = _fake_wave0_catalog()
    default_payload = build_wave3_interaction_confirmation_config(
        tier="fast",
        cluster="ascend",
        catalog_by_name=catalog,
    )
    optional_payload = build_wave3_interaction_confirmation_config(
        tier="fast",
        cluster="ascend",
        catalog_by_name=catalog,
        include_optional_speed_interaction=True,
    )

    assert default_payload["metadata"]["wave"] == "wave3"
    assert default_payload["metadata"]["sweep_family"] == "interaction_confirmation"
    assert len(default_payload["scenarios"]) == len(WAVE0_CANONICAL_FAST_FIXTURES) * 6
    assert len(optional_payload["scenarios"]) == len(WAVE0_CANONICAL_FAST_FIXTURES) * 7

    default_variants = {
        scenario["interaction_variant"] for scenario in default_payload["scenarios"]
    }
    assert default_variants == {
        "baseline",
        "deferred_v1",
        "row_subchunk_512",
        "plan_feature_batch_size",
        "deferred_v1_row_subchunk_512",
        "deferred_v1_plan_feature_batch_size",
    }
    assert "deferred_v1_streaming_v1_row_subchunk_512" not in default_variants

    for scenario in default_payload["scenarios"]:
        assert scenario["baseline_check"]["enabled"] is True
        assert scenario["baseline_check"]["mode"] == "metrics"
        assert scenario["baseline_check"]["baseline_required"] is True
        assert scenario["phase1_trace_batch_policy"] == "legacy"
        assert scenario["phase4_scheduler_mode"] == "locality"
        assert scenario["phase4_ranker"] == "argsort"
        assert scenario["phase4_refresh_optimization"] == "off"

    baseline = next(
        scenario
        for scenario in default_payload["scenarios"]
        if scenario["interaction_variant"] == "baseline"
    )
    assert baseline["phase4_refresh_policy"] == "standard"
    assert baseline["phase4_row_executor"] == "batched"
    assert baseline["row_subchunk_size"] is None
    assert baseline["plan_feature_batch_size"] is False

    optional = next(
        scenario
        for scenario in optional_payload["scenarios"]
        if scenario["interaction_variant"]
        == "deferred_v1_streaming_v1_row_subchunk_512"
    )
    assert optional["phase4_refresh_policy"] == "deferred_v1"
    assert optional["phase4_row_executor"] == "streaming_v1"
    assert optional["row_subchunk_size"] == 512
