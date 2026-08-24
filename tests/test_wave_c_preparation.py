from __future__ import annotations

from pathlib import Path

from nlp_research_project.exact_trace_bench.cli import build_parser
from nlp_research_project.exact_trace_bench.scenarios.governor_calibration_wave_c import (
    GIB,
    build_wave_c_config,
    split_wave_c_configs,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    CANONICAL_BUCKET_NAMES,
)


def test_wave_c_exact_ten_row_matrix_and_heldouts() -> None:
    payload = build_wave_c_config()
    rows = payload["scenarios"]
    cases = {row["calibration_case"]: row for row in rows}

    assert len(rows) == 10
    assert [row["calibration_case"] for row in rows] == [
        "reference_repeat",
        "session256_execution128",
        "phase1_cap64",
        "phase3_microbatch64",
        "replay2",
        "replay8",
        "cache4gib",
        "cache8gib",
        "heldout_joint_s256_e256_replay8_cache8gib",
        "session128_execution64",
    ]
    assert payload["metadata"]["array_concurrency"] == 1
    assert payload["metadata"]["prefetch_required"] is False
    assert payload["metadata"]["fail_on_baseline_missing"] is True
    assert payload["metadata"]["slurm_by_variant"]["gemma3_4b_plt"] == {
        "account": "rai",
        "partition": "rai-gpu-grn",
        "qos": "rai-gpu-grn",
        "gres": "gpu:h200:1",
        "cpus_per_task": 12,
        "mem": "400G",
        "time": "02:00:00",
    }
    assert payload["metadata"]["slurm_by_variant"]["gemma3_12b_plt"]["mem"] == "600G"
    assert (
        payload["metadata"]["slurm_by_variant"]["gemma3_12b_plt"]["time"] == "08:00:00"
    )
    assert len(payload["metadata"]["declared_heldouts_not_rerun"]) == 2
    split = split_wave_c_configs(payload)
    assert len(split["gemma3_4b_plt"]["scenarios"]) == 9
    assert len(split["gemma3_12b_plt"]["scenarios"]) == 1
    assert split["gemma3_4b_plt"]["metadata"]["slurm"]["mem"] == "400G"
    assert split["gemma3_12b_plt"]["metadata"]["slurm"]["mem"] == "600G"
    assert {row["resource_profile"] for row in rows} == {"governor_calibration_h200"}

    for row in rows:
        semantic = 64 if "12b" in row["name"] else 128
        assert row["attribution_batch_size"] == semantic
        assert row["feature_batch_size"] == semantic
        assert row["logit_batch_size"] == semantic
        assert row["decoder_chunk_size"] == 4096
        assert row["attribution_update_interval"] == 4
        assert row["error_vector_prefetch_lookahead"] == 0
        assert row["governor_admission_mode"] == "advisory"
        assert row["baseline_check"]["baseline_required"] is True
        if row["calibration_case"] == "reference_repeat":
            assert row["governor_fidelity_mode"] == "exact"
        else:
            assert row["governor_fidelity_mode"] == "bounded"
            assert set(row["governor_fidelity_budget"]["metrics"]) == {
                f"overall_mean_bucket_{name.replace('<-', '_').replace('-', '_')}_weighted_jaccard"
                for name in CANONICAL_BUCKET_NAMES
            }

    joint = cases["heldout_joint_s256_e256_replay8_cache8gib"]
    assert joint["calibration_campaign"]["split"] == "heldout"
    assert joint["phase4_execution_batch_max_rows"] == 256
    assert joint["cross_batch_decoder_cache_bytes"] == 8 * GIB
    assert cases["reference_repeat"]["calibration_campaign"]["split"] == "fit"
    assert cases["reference_repeat"]["calibration_campaign"]["role"] == "reference"


def test_wave_c_cli_commands_are_focused() -> None:
    parser = build_parser()
    assert (
        parser.parse_args(["build-governor-wave-c"]).func.__name__
        == "_cmd_build_governor_wave_c"
    )
    assert parser.parse_args(
        ["finalize-calibration", "--job-id", "1", "--root", "/x"]
    ).root == [Path("/x")]
    assert parser.parse_args(
        ["backfill-calibration-observations", "--root", "/x"]
    ).root == [Path("/x")]
    assert parser.parse_args(
        [
            "publish-response-bundle",
            "--observation-manifest",
            "/manifest",
            "--output",
            "/bundle",
        ]
    ).observation_manifest == Path("/manifest")


def test_finalizer_sbatch_has_afterany_contract_and_snapshot_guard() -> None:
    script = Path(
        "slurm/exact_trace_bench/finalize_governor_calibration.granite.sbatch"
    ).read_text()
    assert "--dependency=afterany:123:124" in script
    assert "require_snapshot_workspace.sh" in script
    assert "LIB_WORKSPACE_ROOT" in script
    assert "BUNDLE_OUTPUT" in script
    assert "HISTORICAL_OBSERVATION_MANIFEST" in script
    assert "finalize-calibration --job-id" in script
    assert "publish-response-bundle" in script
    assert "#SBATCH --gres" not in script
