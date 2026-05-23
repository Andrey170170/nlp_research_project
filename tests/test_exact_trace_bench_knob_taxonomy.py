from __future__ import annotations

import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp_research_project.exact_trace_bench.scenarios import (  # noqa: E402
    ADVANCED_PUBLIC_TUNING_KEYS,
    DEBUG_REPLAY_PUBLIC_KEYS,
    DEPRECATED_COMPAT_KEYS,
    EXACT_MODE_KNOB_KEYS,
    RESOURCE_PROFILE_STANDARD,
    SCENARIO_TIERS,
    STABLE_PUBLIC_SCENARIO_KEYS,
    TELEMETRY_KEYS,
    WAVE2A_PHASE1_VARIANTS,
    WAVE2B_PHASE4_VARIANTS,
    WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS,
    WAVE2C_ROW_ENCODER_VARIANTS,
    WAVE0_CANONICAL_ANOMALY_FIXTURES,
    WAVE0_CANONICAL_FAST_FIXTURES,
    WAVE0_CANONICAL_LATE_FIXTURES,
    WAVE0_NEW_BASE_FIXTURES,
    WAVE0_NEW_LATE_FIXTURES,
    WAVE0_REPEAT_COUNT,
    WAVE4_GENERALIZATION_VARIANTS,
    build_tier_config,
    build_wave0_baseline_config,
    build_wave2a_phase1_config,
    build_wave2b_phase4_config,
    build_wave2c_row_encoder_config,
    build_wave3_interaction_confirmation_config,
    build_wave4_generalization_config,
)
from experiments.run_sparsification_experiment import build_command  # noqa: E402


CLUSTERS = ("ascend", "cardinal")

SCENARIO_PUBLIC_IMPORTS = (
    "ADVANCED_PUBLIC_TUNING_KEYS",
    "DEBUG_REPLAY_PUBLIC_KEYS",
    "DEPRECATED_COMPAT_KEYS",
    "EXACT_MODE_KNOB_KEYS",
    "RESOURCE_PROFILE_LONG_EVAL_HIGH_MEM",
    "RESOURCE_PROFILE_STANDARD",
    "SCENARIO_TIERS",
    "STABLE_PUBLIC_SCENARIO_KEYS",
    "TELEMETRY_KEYS",
    "WAVE2A_PHASE1_TIERS",
    "WAVE2B_PHASE4_TIERS",
    "WAVE2C_ROW_ENCODER_TIERS",
    "WAVE3_INTERACTION_CONFIRMATION_TIERS",
    "WAVE4_GENERALIZATION_TIERS",
    "build_tier_config",
    "build_wave0_baseline_config",
    "build_wave2a_phase1_config",
    "build_wave2b_phase4_config",
    "build_wave2c_row_encoder_config",
    "build_wave3_interaction_confirmation_config",
    "build_wave4_generalization_config",
    "scenario_file_name",
    "wave0_scenario_file_name",
    "wave2a_phase1_scenario_file_name",
    "wave2b_phase4_scenario_file_name",
    "wave2c_row_encoder_scenario_file_name",
    "wave3_interaction_confirmation_scenario_file_name",
    "wave4_generalization_scenario_file_name",
    "write_all_tiers",
    "write_tier_config",
    "write_wave0_baseline_config",
    "write_wave2a_phase1_config",
    "write_wave2b_phase4_config",
    "write_wave2c_row_encoder_config",
    "write_wave3_interaction_confirmation_config",
    "write_wave4_generalization_config",
)

CANONICAL_TIER_SUMMARIES = {
    ("fast", "ascend"): {
        "stage": "exact_trace_bench_fast",
        "resource_profile": "standard",
        "fixtures": ("828_base", "361_base"),
        "names": (
            "ascend_fast_828_base_b128_c2048_cache0g",
            "ascend_fast_361_base_b128_c2048_cache0g",
        ),
    },
    ("fast", "cardinal"): {
        "stage": "exact_trace_bench_fast",
        "resource_profile": "standard",
        "fixtures": ("828_base", "361_base"),
        "names": (
            "cardinal_fast_828_base_b128_c4096_cache0g",
            "cardinal_fast_361_base_b128_c4096_cache0g",
        ),
    },
    ("anomaly", "ascend"): {
        "stage": "exact_trace_bench_anomaly",
        "resource_profile": "standard",
        "fixtures": ("94_base",),
        "names": ("ascend_anomaly_94_base_b256_c4096_cache0g",),
    },
    ("anomaly", "cardinal"): {
        "stage": "exact_trace_bench_anomaly",
        "resource_profile": "standard",
        "fixtures": ("94_base",),
        "names": ("cardinal_anomaly_94_base_b256_c4096_cache0g",),
    },
    ("long_eval", "ascend"): {
        "stage": "exact_trace_bench_long_eval",
        "resource_profile": "long_eval_high_mem",
        "fixtures": ("361_late", "828_late", "94_late", "361_late"),
        "names": (
            "ascend_long_eval_no_cache_361_late_b128_c2048_cache0g",
            "ascend_long_eval_no_cache_828_late_b128_c2048_cache0g",
            "ascend_long_eval_no_cache_94_late_b128_c2048_cache0g",
            "ascend_long_eval_cache_probe_361_late_b128_c2048_cache8g",
        ),
    },
    ("long_eval", "cardinal"): {
        "stage": "exact_trace_bench_long_eval",
        "resource_profile": "long_eval_high_mem",
        "fixtures": ("361_late", "828_late", "94_late", "361_late"),
        "names": (
            "cardinal_long_eval_no_cache_361_late_b256_c4096_cache0g",
            "cardinal_long_eval_no_cache_828_late_b256_c4096_cache0g",
            "cardinal_long_eval_no_cache_94_late_b256_c4096_cache0g",
            "cardinal_long_eval_cache_probe_361_late_b256_c4096_cache8g",
        ),
    },
}

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


def test_scenario_package_reexports_public_generation_surface() -> None:
    import nlp_research_project.exact_trace_bench.scenarios as scenarios

    assert set(SCENARIO_PUBLIC_IMPORTS).issubset(set(scenarios.__all__))
    for name in SCENARIO_PUBLIC_IMPORTS:
        assert hasattr(scenarios, name), name


def test_legacy_experiments_scenarios_namespace_still_forwards() -> None:
    scenarios = importlib.import_module("experiments.exact_trace_bench.scenarios")

    assert getattr(scenarios, "build_tier_config")(
        tier="fast", cluster="ascend"
    ) == build_tier_config(tier="fast", cluster="ascend")
    assert callable(getattr(scenarios, "build_wave0_baseline_config"))
    assert getattr(scenarios, "RESOURCE_PROFILE_STANDARD") == RESOURCE_PROFILE_STANDARD


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


def test_canonical_generated_payload_summaries_are_stable() -> None:
    seen_resource_profiles = set()
    for tier, cluster, payload in _canonical_payloads():
        expected = CANONICAL_TIER_SUMMARIES[(tier, cluster)]
        metadata = payload["metadata"]
        seen_resource_profiles.add(metadata["resource_profile"])

        assert metadata["tier"] == tier
        assert metadata["cluster"] == cluster
        assert metadata["stage"] == expected["stage"]
        assert metadata["resource_profile"] == expected["resource_profile"]
        assert payload["defaults"]["exact_trace_internal_dtype"] == "fp32"

        scenario_summary = tuple(
            (scenario["name"], scenario["fixture_name"])
            for scenario in payload["scenarios"]
        )
        assert len(expected["names"]) == len(expected["fixtures"])
        assert scenario_summary == tuple(zip(expected["names"], expected["fixtures"]))

        for scenario in payload["scenarios"]:
            assert scenario["stage"] == expected["stage"]
            assert scenario["cluster"] == cluster
            assert scenario["resource_profile"] == expected["resource_profile"]
            assert "baseline_check" not in scenario
            assert not any(key in scenario for key in DEBUG_REPLAY_PUBLIC_KEYS)

    assert seen_resource_profiles == {"standard", "long_eval_high_mem"}


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


def test_wave2_scenarios_preserve_variants_and_baseline_checks() -> None:
    catalog = _fake_wave0_catalog()
    expected_builders = (
        (
            build_wave2a_phase1_config,
            "wave2a",
            "phase1_variant",
            {variant["label"] for variant in WAVE2A_PHASE1_VARIANTS},
            len(WAVE2A_PHASE1_VARIANTS),
        ),
        (
            build_wave2b_phase4_config,
            "wave2b",
            "phase4_variant",
            {variant["label"] for variant in WAVE2B_PHASE4_VARIANTS},
            len(WAVE2B_PHASE4_VARIANTS),
        ),
        (
            build_wave2c_row_encoder_config,
            "wave2c",
            "row_encoder_variant",
            {variant["label"] for variant in WAVE2C_ROW_ENCODER_VARIANTS},
            len(WAVE2C_ROW_ENCODER_VARIANTS),
        ),
    )

    for (
        builder,
        wave,
        variant_key,
        expected_variants,
        variants_per_fixture,
    ) in expected_builders:
        payload = builder(tier="fast", cluster="ascend", catalog_by_name=catalog)
        assert payload["metadata"]["wave"] == wave
        assert payload["metadata"]["resource_profile"] == RESOURCE_PROFILE_STANDARD
        assert len(payload["scenarios"]) == (
            len(WAVE0_CANONICAL_FAST_FIXTURES) * variants_per_fixture
        )
        assert {
            scenario[variant_key] for scenario in payload["scenarios"]
        } == expected_variants

        for scenario in payload["scenarios"]:
            assert scenario["baseline_check"] == {
                "enabled": True,
                "mode": "metrics",
                "registry_key": f"wave0/{scenario['fixture_name']}/ascend/fast/fp32_default",
                "baseline_required": True,
                "thresholds": None,
            }

    wave2b = build_wave2b_phase4_config(
        tier="anomaly",
        cluster="cardinal",
        catalog_by_name=catalog,
    )
    assert wave2b["metadata"]["resource_profile"] == RESOURCE_PROFILE_STANDARD
    assert len(wave2b["scenarios"]) == (
        len(WAVE0_CANONICAL_ANOMALY_FIXTURES) * len(WAVE2B_PHASE4_VARIANTS)
    )
    assert {
        scenario["phase1_trace_batch_policy"] for scenario in wave2b["scenarios"]
    } == {"legacy"}

    wave2c_baseline = next(
        scenario
        for scenario in build_wave2c_row_encoder_config(
            tier="fast",
            cluster="ascend",
            catalog_by_name=catalog,
        )["scenarios"]
        if scenario["row_encoder_variant"] == "legacy_default"
    )
    for key, value in WAVE2C_ROW_ENCODER_LEGACY_DEFAULTS.items():
        assert wave2c_baseline[key] == value


def test_wave2_rejects_long_eval_tier() -> None:
    catalog = _fake_wave0_catalog()
    for builder in (
        build_wave2a_phase1_config,
        build_wave2b_phase4_config,
        build_wave2c_row_encoder_config,
    ):
        try:
            builder(tier="long_eval", cluster="ascend", catalog_by_name=catalog)
        except ValueError as exc:
            assert "long_eval" in str(exc)
        else:
            raise AssertionError(f"{builder.__name__} accepted long_eval")


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


def test_wave4_generalization_scenarios_cover_wave0_prompts_and_finalists() -> None:
    catalog = _fake_wave0_catalog()
    expected_fixtures = {
        "fast": WAVE0_CANONICAL_FAST_FIXTURES + WAVE0_NEW_BASE_FIXTURES,
        "anomaly": WAVE0_CANONICAL_ANOMALY_FIXTURES,
        "long_eval": WAVE0_CANONICAL_LATE_FIXTURES + WAVE0_NEW_LATE_FIXTURES,
    }
    expected_variants = {variant["label"] for variant in WAVE4_GENERALIZATION_VARIANTS}

    for tier, fixture_names in expected_fixtures.items():
        payload = build_wave4_generalization_config(
            tier=tier,
            cluster="cardinal",
            catalog_by_name=catalog,
        )

        assert payload["metadata"]["wave"] == "wave4"
        assert payload["metadata"]["sweep_family"] == "prompt_generalization"
        assert "baseline_registry" in payload["metadata"]
        assert "run_name" in payload["metadata"]
        assert "run_goal" in payload["metadata"]
        assert payload["metadata"]["resource_profile"] == (
            "long_eval_high_mem" if tier == "long_eval" else "standard"
        )
        assert len(payload["scenarios"]) == len(fixture_names) * 3
        assert {scenario["fixture_name"] for scenario in payload["scenarios"]} == set(
            fixture_names
        )
        assert {
            scenario["generalization_variant"] for scenario in payload["scenarios"]
        } == expected_variants

        for scenario in payload["scenarios"]:
            assert scenario["baseline_check"] == {
                "enabled": True,
                "mode": "metrics",
                "registry_key": f"wave0/{scenario['fixture_name']}/cardinal/{tier}/fp32_default",
                "baseline_required": True,
                "thresholds": None,
            }
            assert scenario["phase1_trace_batch_policy"] == "legacy"
            assert scenario["row_store_cache_control"] == "off"
            assert scenario["exact_encoder_residency"] == "lazy"
            assert scenario["phase4_scheduler_mode"] == "locality"
            assert scenario["phase4_refresh_policy"] == "standard"
            assert scenario["phase4_ranker"] == "argsort"
            assert scenario["phase4_refresh_optimization"] == "off"
            assert scenario["phase4_row_executor"] == "batched"
            if tier == "long_eval":
                assert scenario["attribution_batch_size"] == 256
                assert scenario["decoder_chunk_size"] == 4096
                assert scenario["cross_batch_decoder_cache_bytes"] == 0

        baseline = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["generalization_variant"] == "baseline"
        )
        assert baseline["row_subchunk_size"] is None
        assert baseline["plan_feature_batch_size"] is False

        row_subchunk = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["generalization_variant"] == "row_subchunk_512"
        )
        assert row_subchunk["row_subchunk_size"] == 512
        assert row_subchunk["plan_feature_batch_size"] is False

        planner = next(
            scenario
            for scenario in payload["scenarios"]
            if scenario["generalization_variant"] == "plan_feature_batch_size"
        )
        assert planner["row_subchunk_size"] is None
        assert planner["plan_feature_batch_size"] is True
