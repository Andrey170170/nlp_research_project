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
    build_tier_config,
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
