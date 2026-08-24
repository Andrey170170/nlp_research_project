from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DEFAULT_GENERATED_DIR, DEFAULT_SCRATCH_ROOT, base_trace_defaults
from ..fixtures import resolve_fixture
from ..io_utils import ensure_dir, write_json
from ..transcoder_config import (
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)
from ..typed_compact_graph import CANONICAL_BUCKET_NAMES
from .governor_calibration import GOVERNOR_CALIBRATION_BASELINE_REGISTRY


GIB = 1024**3
PHASE4_STATIC_COALESCING_FIXTURE = "361_base"
PHASE4_STATIC_COALESCING_PROVIDER_FAMILY = "gemmascope2-plt-1b-small"
PHASE4_STATIC_COALESCING_GOVERNOR_PROFILE = "granite_h200_1b_plt_b128_c4096_cache0"
PHASE4_STATIC_COALESCING_RESOURCE_PROFILE = "governor_calibration_h200"
PHASE4_STATIC_COALESCING_BASELINE_KEY = "governor_calibration/gemma3_1b_plt/361_base"
PHASE4_STATIC_COALESCING_TIMEOUT_MINUTES = 120


def _validation_rows() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "validation_case": f"execution_b{execution_rows}",
            "nnsight_session_capacity": execution_rows,
            "phase4_execution_batch_max_rows": execution_rows,
        }
        for execution_rows in (128, 256, 512)
    )


def build_phase4_static_coalescing_config(
    *,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
    baseline_registry: Path = GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
) -> dict[str, Any]:
    fixture = resolve_fixture(
        PHASE4_STATIC_COALESCING_FIXTURE,
        catalog_by_name=catalog_by_name,
        allow_fallback=True,
    )
    cache_root = model_cache_root or scratch_root.parent / "huggingface"
    provider_config = transcoder_config_to_json(
        resolve_transcoder_load_config(
            transcoder_provider_family=PHASE4_STATIC_COALESCING_PROVIDER_FAMILY,
            transcoder_cache_dir=str(cache_root),
        )
    )
    stage = "phase4_static_coalescing_gemma3_1b_plt"
    output_root = (
        scratch_root
        / "granite"
        / "sweep"
        / "governor_calibration"
        / "phase4_static_coalescing"
        / "gemma3_1b_plt"
    )
    defaults = {
        **base_trace_defaults(),
        **provider_config,
        "stage": stage,
        "cluster": "granite",
        "tier": "sweep",
        "resource_profile": PHASE4_STATIC_COALESCING_RESOURCE_PROFILE,
        "governor_profile_name": PHASE4_STATIC_COALESCING_GOVERNOR_PROFILE,
        "governor_admission_mode": "advisory",
        "attribution_batch_size": 128,
        "feature_batch_size": 128,
        "logit_batch_size": 128,
        "attribution_update_interval": 4,
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "exact_trace_internal_dtype": "fp32",
        "completions": 1,
        "max_steps": 1,
        "save_raw": False,
        "incremental_telemetry_jsonl": True,
        "verbose_attribution": True,
        "profile_attribution": True,
        "profile_log_interval": 1,
    }
    envelope = {
        "total_vram_bytes": 151_397_597_184,
        "vram_fraction": 0.9,
        "host_budget_bytes": 200 * GIB,
        "file_cache_allowance_bytes": 64 * GIB,
        "local_disk_bytes": 64 * GIB,
        "scratch_disk_bytes": 64 * GIB,
        "walltime_seconds": PHASE4_STATIC_COALESCING_TIMEOUT_MINUTES * 60,
    }
    scenarios = []
    for row in _validation_rows():
        case = str(row["validation_case"])
        scenarios.append(
            {
                "name": f"granite_phase4_static_coalescing_gemma3_1b_plt_{case}",
                "run_name": f"phase4-static-coalescing-gemma3-1b-plt-{case}",
                "run_goal": (
                    "Validate static Phase-4 execution coalescing while preserving the "
                    "canonical 128x4 semantic frontier schedule."
                ),
                **fixture.to_source_payload(),
                "phase1_trace_batch_policy": "cap_effective_batches",
                "phase1_trace_batch_size_max": 128,
                "phase3_compute_microbatch_max_rows": 128,
                **row,
                "governor_resource_envelope": dict(envelope),
                "timeout_minutes": PHASE4_STATIC_COALESCING_TIMEOUT_MINUTES,
                "baseline_check": {
                    "enabled": True,
                    "mode": "gate",
                    "registry_key": PHASE4_STATIC_COALESCING_BASELINE_KEY,
                    "baseline_required": True,
                    "thresholds": {
                        "overall_mean_feature_jaccard_min": 1.0,
                        **{
                            f"overall_mean_bucket_{name.replace('<-', '_').replace('-', '_')}_exact_min": 1.0
                            for name in CANONICAL_BUCKET_NAMES
                        },
                    },
                },
            }
        )

    return {
        "schema_version": 1,
        "defaults": defaults,
        "metadata": {
            "cluster": "granite",
            "stage": stage,
            "operational_class": "sweep",
            "resource_profile": PHASE4_STATIC_COALESCING_RESOURCE_PROFILE,
            "array_concurrency": 1,
            "baseline_registry": str(baseline_registry),
            "baseline_registry_pinned": True,
            "fail_on_baseline_missing": True,
            "fail_on_validation_fail": True,
            "immutable_validation_config": True,
            "recommended_output_root": str(output_root),
            "matrix_case_count": len(scenarios),
            "fixed_semantics": {
                "feature_batch_size": 128,
                "frontier_refresh_stride": 4,
                "nominal_frontier_window_rows": 512,
            },
            "execution_batch_ladder": [128, 256, 512],
            "acceptance": {
                "semantic_batch_count_equal": True,
                "refresh_count_equal": True,
                "prepared_frontier_membership_and_order_hashes_equal": True,
                "ranker_pre_locality_order_hash_advisory": True,
                "compact_graph_parity_required": True,
                "typed_bucket_gate": "six_bucket_strict_exact",
                "validator": "experiments/analyze_phase4_static_coalescing.py",
            },
            "slurm": {
                "account": "rai",
                "partition": "rai-gpu-grn",
                "qos": "rai-gpu-grn-short",
                "gres": "gpu:h200:1",
                "cpus_per_task": 12,
                "mem": "200G",
                "time": "02:00:00",
            },
        },
        "scenarios": scenarios,
    }


def phase4_static_coalescing_scenario_file_name() -> str:
    return "exact_trace_phase4_static_coalescing_gemma3_1b_plt_granite_scenarios.json"


def write_phase4_static_coalescing_config(
    payload: dict[str, Any], *, output_dir: Path = DEFAULT_GENERATED_DIR
) -> Path:
    ensure_dir(output_dir)
    path = output_dir / phase4_static_coalescing_scenario_file_name()
    write_json(path, payload)
    return path
