from __future__ import annotations

from pathlib import Path

from nlp_research_project.exact_trace_bench.scenarios.chpc_baseline import (
    build_chpc_baseline_config,
)


def test_12b_baseline_uses_extended_profiled_rerun_envelope(tmp_path: Path) -> None:
    payload = build_chpc_baseline_config(
        variant="gemma3_12b_plt",
        scratch_root=tmp_path / "scratch",
        model_cache_root=tmp_path / "huggingface",
    )

    assert payload["metadata"]["slurm"]["mem"] == "600G"
    assert payload["metadata"]["slurm"]["time"] == "08:30:00"
    assert len(payload["scenarios"]) == 3
    for scenario in payload["scenarios"]:
        assert scenario["timeout_minutes"] == 480
        assert scenario["attribution_batch_size"] == 64
        assert scenario["feature_batch_size"] == 64
        assert scenario["logit_batch_size"] == 64
        assert scenario["decoder_chunk_size"] == 4096
        assert scenario["verbose_attribution"] is True
        assert scenario["profile_attribution"] is True
        assert scenario["profile_log_interval"] == 1
        assert scenario["incremental_telemetry_jsonl"] is True
