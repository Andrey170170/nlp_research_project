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
from .base import _require_cluster


CHPC_BASELINE_RESOURCE_PROFILE = "baseline_h200"
CHPC_BASELINE_FIXTURES: tuple[str, ...] = ("828_base", "361_base", "94_base")
CHPC_BASELINE_VARIANTS: dict[str, dict[str, Any]] = {
    "gemma3_1b_clt": {
        "provider_family": "gemmascope2-clt-1b-medium-affine",
        "batch": 1000,
        "cache_gib": 8,
        "timeout_minutes": 90,
        "slurm_mem": "200G",
        "slurm_time": "01:30:00",
    },
    "gemma3_1b_plt": {
        "provider_family": "gemmascope2-plt-1b-small",
        "batch": 128,
        "cache_gib": 0,
        "timeout_minutes": 90,
        "slurm_mem": "200G",
        "slurm_time": "01:30:00",
    },
    "gemma3_4b_plt": {
        "provider_family": "gemmascope2-plt-4b-small",
        "batch": 128,
        "cache_gib": 0,
        "timeout_minutes": 120,
        "slurm_mem": "400G",
        "slurm_time": "02:00:00",
    },
    "gemma3_12b_plt": {
        "provider_family": "gemmascope2-plt-12b-small",
        "batch": 64,
        "cache_gib": 0,
        "timeout_minutes": 300,
        "slurm_mem": "400G",
        "slurm_time": "05:00:00",
    },
}


def _default_model_cache_root(scratch_root: Path) -> Path:
    return scratch_root.parent / "huggingface"


def build_chpc_baseline_config(
    *,
    variant: str,
    cluster: str = "granite",
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
) -> dict[str, Any]:
    _require_cluster(cluster)
    if variant not in CHPC_BASELINE_VARIANTS:
        available = ", ".join(sorted(CHPC_BASELINE_VARIANTS))
        raise ValueError(f"Unknown CHPC baseline variant {variant!r}: {available}")

    spec = CHPC_BASELINE_VARIANTS[variant]
    cache_root = model_cache_root or _default_model_cache_root(scratch_root)
    provider_config = transcoder_config_to_json(
        resolve_transcoder_load_config(
            transcoder_provider_family=str(spec["provider_family"]),
            transcoder_cache_dir=str(cache_root),
        )
    )
    batch = int(spec["batch"])
    decoder_chunk_size = 4096
    cache_bytes = int(spec["cache_gib"]) * (1024**3)
    output_root = scratch_root / cluster / "baseline" / variant
    stage = f"exact_trace_bench_baseline_{variant}"

    payload: dict[str, Any] = {
        "defaults": base_trace_defaults(),
        "metadata": {
            "cluster": cluster,
            "stage": stage,
            "tier": "baseline",
            "resource_profile": CHPC_BASELINE_RESOURCE_PROFILE,
            "baseline_variant": variant,
            "recommended_output_root": str(output_root),
            "run_name": f"CHPC baseline {variant}",
            "run_goal": (
                "Validate CHPC H200 setup and export baseline exact-trace "
                "graphs/timings for comparison."
            ),
            "slurm": {
                "account": "rai",
                "partition": "rai-gpu-grn",
                "qos": "rai-gpu-grn",
                "gres": "gpu:h200:1",
                "mem": str(spec["slurm_mem"]),
                "time": str(spec["slurm_time"]),
            },
            "notes": [
                "Canonical base fixtures only: 828_base, 361_base, 94_base.",
                "Run on H200-class GPUs for cross-run consistency.",
                "Raw tensor dumps remain disabled; compact graph artifacts, scenario.json, result.json, and scenario_metrics are exported.",
            ],
        },
        "scenarios": [],
    }

    for fixture_name in CHPC_BASELINE_FIXTURES:
        fixture = resolve_fixture(
            fixture_name,
            catalog_by_name=catalog_by_name,
            allow_fallback=True,
        )
        row = {
            "name": (
                f"{cluster}_baseline_{variant}_{fixture.fixture_name}"
                f"_b{batch}_c{decoder_chunk_size}_cache{spec['cache_gib']}g"
            ),
            "stage": stage,
            "cluster": cluster,
            "tier": "baseline",
            "resource_profile": CHPC_BASELINE_RESOURCE_PROFILE,
            "baseline_variant": variant,
            "recommended_output_root": str(output_root),
            **fixture.to_source_payload(),
            **provider_config,
            "attribution_batch_size": batch,
            "feature_batch_size": batch,
            "logit_batch_size": batch,
            "decoder_chunk_size": decoder_chunk_size,
            "cross_batch_decoder_cache_bytes": cache_bytes,
            "exact_trace_internal_dtype": "fp32",
            "capture_phase0_donor_bundle": True,
            "capture_phase3_seed_bundle": True,
            "capture_feature_semantic_descriptors": False,
            "timeout_minutes": int(spec["timeout_minutes"]),
        }
        payload["scenarios"].append(row)

    return payload


def chpc_baseline_scenario_file_name(*, variant: str, cluster: str = "granite") -> str:
    return f"exact_trace_bench_baseline_{variant}_{cluster}_scenarios.json"


def write_chpc_baseline_config(
    payload: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_GENERATED_DIR,
    variant: str,
    cluster: str = "granite",
) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / chpc_baseline_scenario_file_name(
        variant=variant,
        cluster=cluster,
    )
    write_json(output_path, payload)
    return output_path
