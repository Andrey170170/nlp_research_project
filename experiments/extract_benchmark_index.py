from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from extract_utils import ensure_dir, flatten_dict, read_json, write_csv, write_jsonl


DEFAULT_INPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")


def _infer_cluster(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
    return None


def _special_case_label(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    stage = str(scenario.get("stage") or "")
    joined = "/".join(scenario_root.parts)
    if "prompt94_compare" in stage or "prompt94_compare" in joined:
        return "prompt94_compare"
    return None


def _load_prompt_meta(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta_files = sorted(artifact_dir.glob("prompt_*/prompt_meta.json"))
    if not prompt_meta_files:
        return {}
    return read_json(prompt_meta_files[0])


def _load_completion_manifests(artifact_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("prompt_*/completion_*/completion.json")):
        manifests.append(read_json(path))
    return manifests


def _summarize_artifacts(artifact_dir: Path) -> dict[str, Any]:
    prompt_meta = _load_prompt_meta(artifact_dir)
    manifests = _load_completion_manifests(artifact_dir)

    steps = [
        step
        for manifest in manifests
        for step in manifest.get("steps", [])
        if isinstance(step, dict)
    ]
    diagnostics = [
        step.get("transcoder_diagnostics", {})
        for step in steps
        if isinstance(step.get("transcoder_diagnostics"), dict)
    ]

    resource_snapshot = manifests[0].get("resource_snapshot") if manifests else None
    first_manifest = manifests[0] if manifests else {}
    max_active_features = max(
        (
            step.get("n_active_features")
            for step in steps
            if step.get("n_active_features") is not None
        ),
        default=None,
    )
    max_edges_retained = max(
        (
            step.get("n_edges_retained")
            for step in steps
            if step.get("n_edges_retained") is not None
        ),
        default=None,
    )
    decoder_cache_hit_count = max(
        (
            diag.get("decoder_cache_hit_count")
            for diag in diagnostics
            if diag.get("decoder_cache_hit_count") is not None
        ),
        default=None,
    )
    decoder_cache_miss_count = max(
        (
            diag.get("decoder_cache_miss_count")
            for diag in diagnostics
            if diag.get("decoder_cache_miss_count") is not None
        ),
        default=None,
    )
    decoder_cache_eviction_count = max(
        (
            diag.get("decoder_cache_eviction_count")
            for diag in diagnostics
            if diag.get("decoder_cache_eviction_count") is not None
        ),
        default=None,
    )
    decoder_load_count = max(
        (
            diag.get("decoder_load_count")
            for diag in diagnostics
            if diag.get("decoder_load_count") is not None
        ),
        default=None,
    )
    decoder_load_seconds = max(
        (
            diag.get("decoder_load_seconds")
            for diag in diagnostics
            if diag.get("decoder_load_seconds") is not None
        ),
        default=None,
    )
    reconstruction_chunk_count = max(
        (
            diag.get("reconstruction_chunk_count")
            for diag in diagnostics
            if diag.get("reconstruction_chunk_count") is not None
        ),
        default=None,
    )
    reconstruction_seconds = max(
        (
            diag.get("reconstruction_seconds")
            for diag in diagnostics
            if diag.get("reconstruction_seconds") is not None
        ),
        default=None,
    )
    encode_sparse_seconds = max(
        (
            diag.get("encode_sparse_seconds")
            for diag in diagnostics
            if diag.get("encode_sparse_seconds") is not None
        ),
        default=None,
    )

    return {
        "prompt_count": len(list(artifact_dir.glob("prompt_*"))),
        "completion_count": len(manifests),
        "gsm8k_index": prompt_meta.get("gsm8k_index"),
        "prompt_source": first_manifest.get(
            "prompt_source", prompt_meta.get("prompt_source")
        ),
        "fixture_name": first_manifest.get(
            "fixture_name", prompt_meta.get("fixture_name")
        ),
        "fixture_kind": first_manifest.get(
            "fixture_kind", prompt_meta.get("fixture_kind")
        ),
        "prepared_prompt_file": prompt_meta.get("prepared_prompt_file"),
        "prepared_prompt_meta_file": prompt_meta.get("prepared_prompt_meta_file"),
        "prompt_token_count": first_manifest.get(
            "prompt_token_count", prompt_meta.get("prompt_token_count")
        ),
        "initial_input_token_count": first_manifest.get(
            "initial_input_token_count", prompt_meta.get("initial_input_token_count")
        ),
        "generated_token_count": first_manifest.get("generated_token_count"),
        "completion_duration_seconds": first_manifest.get("duration_seconds"),
        "n_steps_traced": first_manifest.get("n_steps_traced"),
        "max_active_features": max_active_features,
        "max_edges_retained": max_edges_retained,
        "decoder_cache_hit_count": decoder_cache_hit_count,
        "decoder_cache_miss_count": decoder_cache_miss_count,
        "decoder_cache_eviction_count": decoder_cache_eviction_count,
        "decoder_load_count": decoder_load_count,
        "decoder_load_seconds": decoder_load_seconds,
        "reconstruction_chunk_count": reconstruction_chunk_count,
        "reconstruction_seconds": reconstruction_seconds,
        "encode_sparse_seconds": encode_sparse_seconds,
        **flatten_dict(resource_snapshot, prefix="resource_snapshot_"),
    }


def build_row(result_path: Path) -> dict[str, Any]:
    scenario_root = result_path.parent
    result = read_json(result_path)
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.exists() else {}
    artifact_dir = Path(result.get("output_dir") or scenario_root / "artifacts")
    profiling = result.get("profiling_summary", {})
    artifact_summary = _summarize_artifacts(artifact_dir)
    special_case = _special_case_label(scenario_root, scenario)
    cache_bytes = scenario.get("cross_batch_decoder_cache_bytes")

    row = {
        "scenario_root": str(scenario_root),
        "scenario_name": result.get("name")
        or scenario.get("name")
        or scenario_root.name,
        "stage": result.get("stage") or scenario.get("stage"),
        "cluster": _infer_cluster(scenario_root, scenario),
        "method": result.get("method") or scenario.get("method"),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "duration_seconds": result.get("duration_seconds"),
        "timeout_minutes": result.get("timeout_minutes"),
        "scenario_file": str(scenario_path) if scenario_path.exists() else None,
        "result_file": str(result_path),
        "run_log_path": result.get("log_path") or str(scenario_root / "run.log"),
        "artifacts_dir": str(artifact_dir),
        "attribution_batch_size": scenario.get("attribution_batch_size"),
        "feature_batch_size": scenario.get("feature_batch_size"),
        "logit_batch_size": scenario.get("logit_batch_size"),
        "decoder_chunk_size": scenario.get("decoder_chunk_size"),
        "decoder_cache_bytes": cache_bytes,
        "decoder_cache_gib": None if cache_bytes is None else cache_bytes / (1024**3),
        "max_feature_nodes": scenario.get("max_feature_nodes"),
        "max_edges": scenario.get("max_edges"),
        "max_steps": scenario.get("max_steps"),
        "temperature": scenario.get("temperature"),
        "max_n_logits": scenario.get("max_n_logits"),
        "desired_logit_prob": scenario.get("desired_logit_prob"),
        "attribution_update_interval": scenario.get("attribution_update_interval"),
        "lazy_encoder": None
        if scenario.get("no_lazy_encoder") is None
        else not scenario.get("no_lazy_encoder"),
        "lazy_decoder": None
        if scenario.get("no_lazy_decoder") is None
        else not scenario.get("no_lazy_decoder"),
        "offload_enabled": None
        if scenario.get("no_offload") is None
        else not scenario.get("no_offload"),
        "verbose_attribution": scenario.get("verbose_attribution"),
        "profile_attribution": scenario.get("profile_attribution"),
        "profile_log_interval": scenario.get("profile_log_interval"),
        "save_raw": scenario.get("save_raw"),
        "is_special_case": special_case is not None,
        "special_case_label": special_case,
        **artifact_summary,
        **profiling,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a flat scenario-level index from weekend exact chunked benchmark results"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root benchmark directory containing scenario subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV/JSONL files will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    rows = [build_row(path) for path in sorted(args.input_root.glob("**/result.json"))]

    preferred_headers = [
        "scenario_root",
        "scenario_name",
        "stage",
        "cluster",
        "method",
        "status",
        "returncode",
        "duration_seconds",
        "gsm8k_index",
        "prompt_source",
        "fixture_name",
        "fixture_kind",
        "attribution_batch_size",
        "feature_batch_size",
        "logit_batch_size",
        "decoder_chunk_size",
        "decoder_cache_gib",
        "max_feature_nodes",
        "max_edges",
        "max_steps",
        "prompt_token_count",
        "initial_input_token_count",
        "generated_token_count",
        "n_steps_traced",
        "max_active_features",
        "phase3_duration_seconds",
        "phase4_duration_seconds",
        "phase4_avg_batch_seconds",
        "peak_rss_gib",
        "peak_cuda_reserved_gib",
        "run_log_path",
        "artifacts_dir",
        "is_special_case",
        "special_case_label",
    ]
    write_csv(
        args.output_dir / "benchmark_index.csv",
        rows,
        preferred_headers=preferred_headers,
    )
    write_jsonl(args.output_dir / "benchmark_index.jsonl", rows)
    print(f"Wrote {len(rows)} scenario rows to {args.output_dir}")


if __name__ == "__main__":
    main()
