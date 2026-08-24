"""Wave C governor calibration matrix, separate from historical Waves A/B."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

from ..config import DEFAULT_GENERATED_DIR, DEFAULT_SCRATCH_ROOT, base_trace_defaults
from ..fixtures import resolve_fixture
from ..io_utils import ensure_dir, write_json
from ..transcoder_config import (
    resolve_transcoder_load_config,
    transcoder_config_to_json,
)
from ..typed_compact_graph import CANONICAL_BUCKET_NAMES
from .governor_calibration import (
    GIB,
    GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
    GOVERNOR_CALIBRATION_FIXTURE,
)

WAVE_C_VARIANTS = {
    "gemma3_4b_plt": {
        "provider_family": "gemmascope2-plt-4b-small",
        "semantic_batch": 128,
        "profile": "granite_h200_4b_plt_b128_c4096_cache0",
        "mem": "400G",
        "time": "02:00:00",
        "host_memory_bytes": 400 * GIB,
        "walltime_seconds": 2 * 60 * 60,
        "baseline_key": "governor_calibration/gemma3_4b_plt/361_base",
    },
    "gemma3_12b_plt": {
        "provider_family": "gemmascope2-plt-12b-small",
        "semantic_batch": 64,
        "profile": "granite_h200_12b_plt_b64_c4096_cache0",
        "mem": "600G",
        "time": "08:00:00",
        "host_memory_bytes": 600 * GIB,
        "walltime_seconds": 8 * 60 * 60,
        "baseline_key": "governor_calibration/gemma3_12b_plt/361_base",
    },
}

_PARITY_BUDGET = {
    "metrics": {
        f"overall_mean_bucket_{name.replace('<-', '_').replace('-', '_')}_weighted_jaccard": {
            "minimum": 0.99,
            "confidence": 0.95,
        }
        for name in CANONICAL_BUCKET_NAMES
    },
}


def _bounded(axes: tuple[str, ...]) -> dict[str, Any]:
    return {
        "governor_fidelity_mode": "bounded",
        "governor_fidelity_evidence_name": "wave-c-planning-vector-calibration",
        "governor_fidelity_evidence_version": "1",
        "governor_fidelity_budget": {
            **_PARITY_BUDGET,
            "allowed_sensitive_axes": list(axes),
        },
    }


def _fixed(semantic_batch: int) -> dict[str, Any]:
    return {
        "attribution_batch_size": semantic_batch,
        "feature_batch_size": semantic_batch,
        "logit_batch_size": semantic_batch,
        "decoder_chunk_size": 4096,
        "attribution_update_interval": 4,
        "error_vector_prefetch_lookahead": 0,
        "nnsight_session_capacity": semantic_batch,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": semantic_batch,
        "phase3_compute_microbatch_max_rows": semantic_batch,
        "phase4_execution_batch_max_rows": semantic_batch,
        "chunked_feature_replay_window": 4,
        "cross_batch_decoder_cache_bytes": 0,
    }


def _campaign(
    spec: dict[str, Any], *, split: str, role: str, holdout_group: str | None = None
) -> dict[str, Any]:
    semantic = int(spec["semantic_batch"])
    return {
        "campaign_id": "governor_calibration_wave_c",
        "split": split,
        "role": role,
        "fit_group": "wave_c_planning_vectors",
        "holdout_group": holdout_group,
        "fixed_controls": {
            "semantic_batch": semantic,
            "decoder_chunk_size": 4096,
            "frontier_refresh_stride": 4,
            "hardware": "single_h200",
            "admission_mode": "advisory",
            "baseline": "pinned",
            "prefetch_depth": 0,
        },
        "reference": {
            "kind": "baseline_registry",
            "registry_key": spec["baseline_key"],
        },
    }


def _rows() -> list[tuple[str, str, dict[str, Any], tuple[str, ...], str, str]]:
    return [
        ("gemma3_4b_plt", "reference_repeat", {}, (), "fit", "reference"),
        (
            "gemma3_4b_plt",
            "session256_execution128",
            {"nnsight_session_capacity": 256},
            ("session_capacity",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "phase1_cap64",
            {"phase1_trace_batch_size_max": 64},
            ("phase1_source_batch_size",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "phase3_microbatch64",
            {"phase3_compute_microbatch_max_rows": 64},
            ("logit_microbatch_size",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "replay2",
            {"chunked_feature_replay_window": 2},
            ("replay_window",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "replay8",
            {"chunked_feature_replay_window": 8},
            ("replay_window",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "cache4gib",
            {"cross_batch_decoder_cache_bytes": 4 * GIB},
            ("decoder_cache_bytes",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "cache8gib",
            {"cross_batch_decoder_cache_bytes": 8 * GIB},
            ("decoder_cache_bytes",),
            "fit",
            "fit",
        ),
        (
            "gemma3_4b_plt",
            "heldout_joint_s256_e256_replay8_cache8gib",
            {
                "nnsight_session_capacity": 256,
                "phase4_execution_batch_max_rows": 256,
                "chunked_feature_replay_window": 8,
                "cross_batch_decoder_cache_bytes": 8 * GIB,
            },
            (
                "session_capacity",
                "feature_microbatch_size",
                "replay_window",
                "decoder_cache_bytes",
            ),
            "heldout",
            "heldout",
        ),
        (
            "gemma3_12b_plt",
            "session128_execution64",
            {"nnsight_session_capacity": 128},
            ("session_capacity",),
            "fit",
            "fit",
        ),
    ]


def build_wave_c_config(
    *,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    model_cache_root: Path | None = None,
    baseline_registry: Path = GOVERNOR_CALIBRATION_BASELINE_REGISTRY,
) -> dict[str, Any]:
    fixture = resolve_fixture(
        GOVERNOR_CALIBRATION_FIXTURE,
        catalog_by_name=catalog_by_name,
        allow_fallback=True,
    )
    scenarios: list[dict[str, Any]] = []
    for variant, case, changes, axes, split, role in _rows():
        spec = WAVE_C_VARIANTS[variant]
        semantic = int(spec["semantic_batch"])
        provider = transcoder_config_to_json(
            resolve_transcoder_load_config(
                transcoder_provider_family=str(spec["provider_family"]),
                transcoder_cache_dir=str(
                    model_cache_root or scratch_root.parent / "huggingface"
                ),
            )
        )
        fidelity = {"governor_fidelity_mode": "exact"} if not axes else _bounded(axes)
        scenarios.append(
            {
                "name": f"granite_governor_calibration_wave_c_{variant}_{case}",
                "stage": "governor_calibration_wave_c",
                "cluster": "granite",
                "tier": "sweep",
                "resource_profile": "governor_calibration_h200",
                "calibration_stage": "wave_c_response_model",
                "calibration_case": case,
                "calibration_factor": "reference_repeat"
                if not axes
                else "planning_vector",
                **fixture.to_source_payload(),
                **provider,
                **_fixed(semantic),
                **changes,
                **fidelity,
                "governor_profile_name": spec["profile"],
                "governor_admission_mode": "advisory",
                "governor_resource_envelope": {
                    "total_vram_bytes": 151_397_597_184,
                    "vram_fraction": 0.9,
                    "host_budget_bytes": spec["host_memory_bytes"],
                    "file_cache_allowance_bytes": 64 * GIB,
                    "local_disk_bytes": 64 * GIB,
                    "scratch_disk_bytes": 64 * GIB,
                    "walltime_seconds": spec["walltime_seconds"],
                },
                "exact_trace_internal_dtype": "fp32",
                "timeout_minutes": 120 if variant == "gemma3_4b_plt" else 480,
                "baseline_check": {
                    "enabled": True,
                    "mode": "metrics",
                    "registry_key": spec["baseline_key"],
                    "baseline_required": True,
                },
                "calibration_campaign": _campaign(
                    spec,
                    split=split,
                    role=role,
                    holdout_group="wave_c_joint" if split == "heldout" else None,
                ),
            }
        )
    return {
        "schema_version": 1,
        "defaults": {
            **base_trace_defaults(),
            "completions": 1,
            "max_steps": 1,
            "save_raw": False,
            "incremental_telemetry_jsonl": True,
        },
        "metadata": {
            "stage": "governor_calibration_wave_c",
            "cluster": "granite",
            "calibration_wave": "C",
            "matrix_case_count": 10,
            "array_concurrency": 1,
            "baseline_registry": str(baseline_registry),
            "baseline_registry_pinned": True,
            "fail_on_baseline_missing": True,
            "prefetch_required": False,
            "declared_heldouts_not_rerun": [
                {"variant": "gemma3_4b_plt", "wave_b_case": "physical_envelope_b256"},
                {"variant": "gemma3_12b_plt", "wave_b_case": "physical_envelope_b128"},
            ],
            "slurm_by_variant": {
                variant: {
                    "account": "rai",
                    "partition": "rai-gpu-grn",
                    "qos": "rai-gpu-grn",
                    "gres": "gpu:h200:1",
                    "cpus_per_task": 12,
                    "mem": spec["mem"],
                    "time": spec["time"],
                }
                for variant, spec in WAVE_C_VARIANTS.items()
            },
            "recommended_output_root": str(
                scratch_root / "granite" / "sweep" / "governor_calibration" / "wave_c"
            ),
        },
        "scenarios": scenarios,
    }


def write_wave_c_config(
    payload: dict[str, Any], *, output_dir: Path = DEFAULT_GENERATED_DIR
) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "exact_trace_governor_calibration_wave_c_granite_scenarios.json"
    write_json(path, payload)
    return path


def split_wave_c_configs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return launcher-ready model arrays with one resource envelope each."""

    result: dict[str, dict[str, Any]] = {}
    slurm_by_variant = payload["metadata"]["slurm_by_variant"]
    for variant in WAVE_C_VARIANTS:
        selected = [
            row for row in payload["scenarios"] if f"_{variant}_" in row["name"]
        ]
        variant_payload = deepcopy(payload)
        variant_payload["scenarios"] = selected
        variant_payload["metadata"]["variant"] = variant
        variant_payload["metadata"]["matrix_case_count"] = len(selected)
        variant_payload["metadata"]["slurm"] = deepcopy(slurm_by_variant[variant])
        result[variant] = variant_payload
    return result


def write_wave_c_configs(
    payload: dict[str, Any], *, output_dir: Path = DEFAULT_GENERATED_DIR
) -> dict[str, Path]:
    ensure_dir(output_dir)
    outputs: dict[str, Path] = {}
    for variant, variant_payload in split_wave_c_configs(payload).items():
        path = (
            output_dir
            / f"exact_trace_governor_calibration_wave_c_{variant}_granite_scenarios.json"
        )
        write_json(path, variant_payload)
        outputs[variant] = path
    return outputs
