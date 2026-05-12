from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean
from typing import Any

from extract_utils import (
    ensure_dir,
    parse_memory_value_to_gib,
    read_json,
    to_json_string,
    write_csv,
)


DEFAULT_INPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")

MEMORY_RE = re.compile(
    r"rss=(?P<rss>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"(?:rss_current=(?P<rss_current>n/a|\d+(?:\.\d+)?(?: GiB)?), )?"
    r"cuda_alloc=(?P<cuda_alloc>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_reserved=(?P<cuda_reserved>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_peak_alloc=(?P<cuda_peak_alloc>n/a|\d+(?:\.\d+)?(?: GiB)?), "
    r"cuda_peak_reserved=(?P<cuda_peak_reserved>n/a|\d+(?:\.\d+)?(?: GiB)?)"
)
PRECOMP_RE = re.compile(r"Precomputation completed in (?P<seconds>\d+(?:\.\d+)?)s")
FORWARD_RE = re.compile(r"Forward pass completed in (?P<seconds>\d+(?:\.\d+)?)s")
INPUT_VECTOR_RE = re.compile(
    r"Input vector build completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE3_DONE_RE = re.compile(
    r"(?P<count>\d+) logit attribution\(s\) completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
PHASE4_DONE_RE = re.compile(
    r"Feature attributions completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
ATTRIBUTION_DONE_RE = re.compile(
    r"Attribution completed in (?P<seconds>\d+(?:\.\d+)?)s"
)
GPU_RE = re.compile(r"GPU: (?P<gpu>.+)")
VRAM_RE = re.compile(r"VRAM: (?P<vram>\d+(?:\.\d+)?) GB")
PHASE0_ENCODE_LAYER_RE = re.compile(
    r"TRACE phase0\.encode_sparse\.layer_done \| layer=(?P<layer>\d+), "
    r"active_features=(?P<active_features>\d+), elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE0_ENCODE_DONE_RE = re.compile(
    r"TRACE phase0\.encode_sparse\.done \| total_active_features=(?P<active_features>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE0_RECON_START_RE = re.compile(
    r"TRACE phase0\.reconstruction\.start \| n_layers=(?P<n_layers>\d+), "
    r"nnz=(?P<nnz>\d+), chunk_size=(?P<chunk_size>\d+)"
)
PHASE0_RECON_LAYER_START_RE = re.compile(
    r"TRACE phase0\.reconstruction\.layer_start \| layer=(?P<layer>\d+), "
    r"active_features=(?P<active_features>\d+)"
)
PHASE0_RECON_LAYER_DONE_RE = re.compile(
    r"TRACE phase0\.reconstruction\.layer_done \| layer=(?P<layer>\d+), "
    r"chunks=(?P<chunks>\d+), elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE0_RECON_DONE_RE = re.compile(
    r"TRACE phase0\.reconstruction\.done \| total_chunks=(?P<chunks>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
COMPUTE_BATCH_DONE_RE = re.compile(
    r"TRACE compute_batch\.done \| phase=(?P<phase>[^,]+), batch_nodes=(?P<batch_nodes>\d+), "
    r"elapsed_s=(?P<seconds>\d+(?:\.\d+)?)"
)
PHASE4_BATCH_RE = re.compile(
    r"Phase 4 batch (?P<batch_idx>\d+)/(?P<total_batches>\d+) in (?P<seconds>\d+(?:\.\d+)?)s"
    r"(?: \| ctx\[(?P<ctx>.*?)\])?"
    r"(?: \| transcoder\[(?P<transcoder>.*?)\])?"
)
CACHE_EVENT_RE = re.compile(
    r"TRACE decoder\.cache\.(?P<event>hit|miss|eviction) \| .*?"
    r"(?P<counter_name>hit_count|miss_count|eviction_count)=(?P<count>\d+)"
    r"(?:, resident_bytes=(?P<resident_bytes>\d+))?"
)
CUDA_OOM_RE = re.compile(
    r"torch\.OutOfMemoryError: CUDA out of memory\. Tried to allocate (?P<requested>[0-9.]+\s(?:GiB|MiB|KiB))\. "
    r"GPU 0 has a total capacity of (?P<total>[0-9.]+\s(?:GiB|MiB|KiB)) of which (?P<free>[0-9.]+\s(?:GiB|MiB|KiB)) is free\. "
    r"Including non-PyTorch memory, this process has (?P<in_use>[0-9.]+\s(?:GiB|MiB|KiB)) memory in use\."
)
FEATURE_ATTR_LAYER_KEY_RE = re.compile(r"feature_attr_seconds_by_layer\.(?P<layer>\d+)")


def _infer_cluster(scenario_root: Path, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
    return None


def _parse_key_value_blob(blob: str | None) -> dict[str, float | int | str]:
    if not blob:
        return {}

    values: dict[str, float | int | str] = {}
    for item in blob.split(", "):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        if re.fullmatch(r"-?\d+", raw_value):
            values[key] = int(raw_value)
        else:
            try:
                values[key] = float(raw_value)
            except ValueError:
                values[key] = raw_value
    return values


def _max_or_none(
    current: float | int | None, candidate: float | int | None
) -> float | int | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _guess_failure_stage(
    summary: dict[str, Any], result_status: str | None
) -> str | None:
    if result_status == "success":
        return None
    if summary.get("cuda_oom_requested_gib") is not None:
        if summary.get("phase0_encode_total_active_features") is None:
            return "phase0_encode"
        if (
            summary.get("phase0_reconstruction_seconds") is None
            and summary.get("precomputation_seconds") is None
        ):
            return "phase0_reconstruction"
        if (
            summary.get("phase3_logit_attribution_seconds") is None
            and summary.get("phase4_batches_observed", 0) == 0
        ):
            return "phase3"
        if summary.get("phase4_feature_attribution_seconds") is None:
            return "phase4"
    if (
        summary.get("phase4_batches_observed", 0) > 0
        and summary.get("phase4_feature_attribution_seconds") is None
    ):
        return "phase4"
    if (
        summary.get("phase3_logit_attribution_seconds") is None
        and summary.get("precomputation_seconds") is not None
    ):
        return "phase3"
    if (
        summary.get("precomputation_seconds") is None
        and summary.get("phase0_encode_total_active_features") is not None
    ):
        return "phase0_reconstruction"
    if summary.get("phase0_encode_total_active_features") is None:
        return "phase0_encode"
    return "unknown"


def parse_run_log(
    *,
    scenario_root: Path,
    scenario_name: str,
    stage: str | None,
    cluster: str | None,
    result_status: str | None,
    log_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary: dict[str, Any] = {
        "scenario_root": str(scenario_root),
        "scenario_name": scenario_name,
        "stage": stage,
        "cluster": cluster,
        "result_status": result_status,
        "log_path": str(log_path),
        "gpu_name": None,
        "gpu_vram_gib": None,
        "precomputation_seconds": None,
        "forward_pass_seconds": None,
        "input_vector_build_seconds": None,
        "phase0_encode_seconds": None,
        "phase0_encode_total_active_features": None,
        "phase0_reconstruction_seconds": None,
        "phase0_reconstruction_total_chunks": None,
        "phase0_reconstruction_chunk_size": None,
        "phase0_reconstruction_nnz": None,
        "phase3_logit_count": None,
        "phase3_logit_attribution_seconds": None,
        "phase4_feature_attribution_seconds": None,
        "attribution_total_seconds": None,
        "phase3_compute_batch_seconds_total": 0.0,
        "phase4_compute_batch_seconds_total": 0.0,
        "phase4_batches_observed": 0,
        "phase4_batch_seconds_mean": None,
        "phase4_batch_seconds_max": None,
        "phase4_ctx_compute_batch_seconds_total": 0.0,
        "phase4_transcoder_decoder_load_seconds_total": 0.0,
        "log_peak_rss_gib": None,
        "log_peak_cuda_allocated_gib": None,
        "log_peak_cuda_reserved_gib": None,
        "log_peak_cuda_peak_allocated_gib": None,
        "log_peak_cuda_peak_reserved_gib": None,
        "log_max_decoder_cache_hit_count": None,
        "log_max_decoder_cache_miss_count": None,
        "log_max_decoder_cache_eviction_count": None,
        "log_max_decoder_cache_resident_bytes": None,
        "cuda_oom_requested_gib": None,
        "cuda_oom_total_gib": None,
        "cuda_oom_free_gib": None,
        "cuda_oom_in_use_gib": None,
    }

    phase0_encode_rows: list[dict[str, Any]] = []
    phase0_reconstruction_rows: list[dict[str, Any]] = []
    phase4_batch_rows: list[dict[str, Any]] = []
    phase4_batch_layer_rows: list[dict[str, Any]] = []

    phase4_batch_seconds: list[float] = []
    phase0_recon_active_features: dict[int, int] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            gpu_match = GPU_RE.search(line)
            if gpu_match:
                summary["gpu_name"] = gpu_match.group("gpu")

            vram_match = VRAM_RE.search(line)
            if vram_match:
                summary["gpu_vram_gib"] = float(vram_match.group("vram"))

            precomp_match = PRECOMP_RE.search(line)
            if precomp_match:
                summary["precomputation_seconds"] = float(
                    precomp_match.group("seconds")
                )

            forward_match = FORWARD_RE.search(line)
            if forward_match:
                summary["forward_pass_seconds"] = float(forward_match.group("seconds"))

            input_match = INPUT_VECTOR_RE.search(line)
            if input_match:
                summary["input_vector_build_seconds"] = float(
                    input_match.group("seconds")
                )

            phase3_match = PHASE3_DONE_RE.search(line)
            if phase3_match:
                summary["phase3_logit_count"] = int(phase3_match.group("count"))
                summary["phase3_logit_attribution_seconds"] = float(
                    phase3_match.group("seconds")
                )

            phase4_match = PHASE4_DONE_RE.search(line)
            if phase4_match:
                summary["phase4_feature_attribution_seconds"] = float(
                    phase4_match.group("seconds")
                )

            attribution_match = ATTRIBUTION_DONE_RE.search(line)
            if attribution_match:
                summary["attribution_total_seconds"] = float(
                    attribution_match.group("seconds")
                )

            encode_layer_match = PHASE0_ENCODE_LAYER_RE.search(line)
            if encode_layer_match:
                phase0_encode_rows.append(
                    {
                        "scenario_root": str(scenario_root),
                        "scenario_name": scenario_name,
                        "stage": stage,
                        "cluster": cluster,
                        "layer": int(encode_layer_match.group("layer")),
                        "active_features": int(
                            encode_layer_match.group("active_features")
                        ),
                        "elapsed_seconds": float(encode_layer_match.group("seconds")),
                    }
                )

            encode_done_match = PHASE0_ENCODE_DONE_RE.search(line)
            if encode_done_match:
                summary["phase0_encode_total_active_features"] = int(
                    encode_done_match.group("active_features")
                )
                summary["phase0_encode_seconds"] = float(
                    encode_done_match.group("seconds")
                )

            recon_start_match = PHASE0_RECON_START_RE.search(line)
            if recon_start_match:
                summary["phase0_reconstruction_chunk_size"] = int(
                    recon_start_match.group("chunk_size")
                )
                summary["phase0_reconstruction_nnz"] = int(
                    recon_start_match.group("nnz")
                )

            recon_layer_start_match = PHASE0_RECON_LAYER_START_RE.search(line)
            if recon_layer_start_match:
                phase0_recon_active_features[
                    int(recon_layer_start_match.group("layer"))
                ] = int(recon_layer_start_match.group("active_features"))

            recon_layer_done_match = PHASE0_RECON_LAYER_DONE_RE.search(line)
            if recon_layer_done_match:
                layer = int(recon_layer_done_match.group("layer"))
                phase0_reconstruction_rows.append(
                    {
                        "scenario_root": str(scenario_root),
                        "scenario_name": scenario_name,
                        "stage": stage,
                        "cluster": cluster,
                        "layer": layer,
                        "active_features": phase0_recon_active_features.get(layer),
                        "chunks": int(recon_layer_done_match.group("chunks")),
                        "elapsed_seconds": float(
                            recon_layer_done_match.group("seconds")
                        ),
                    }
                )

            recon_done_match = PHASE0_RECON_DONE_RE.search(line)
            if recon_done_match:
                summary["phase0_reconstruction_total_chunks"] = int(
                    recon_done_match.group("chunks")
                )
                summary["phase0_reconstruction_seconds"] = float(
                    recon_done_match.group("seconds")
                )

            compute_batch_match = COMPUTE_BATCH_DONE_RE.search(line)
            if compute_batch_match:
                phase = compute_batch_match.group("phase")
                seconds = float(compute_batch_match.group("seconds"))
                if phase == "phase3_logits":
                    summary["phase3_compute_batch_seconds_total"] += seconds
                elif phase == "phase4_features":
                    summary["phase4_compute_batch_seconds_total"] += seconds

            phase4_batch_match = PHASE4_BATCH_RE.search(line)
            if phase4_batch_match:
                batch_idx = int(phase4_batch_match.group("batch_idx"))
                total_batches = int(phase4_batch_match.group("total_batches"))
                batch_seconds = float(phase4_batch_match.group("seconds"))
                ctx_values = _parse_key_value_blob(phase4_batch_match.group("ctx"))
                transcoder_values = _parse_key_value_blob(
                    phase4_batch_match.group("transcoder")
                )
                phase4_batch_seconds.append(batch_seconds)
                phase4_batch_rows.append(
                    {
                        "scenario_root": str(scenario_root),
                        "scenario_name": scenario_name,
                        "stage": stage,
                        "cluster": cluster,
                        "batch_idx": batch_idx,
                        "total_batches": total_batches,
                        "batch_seconds": batch_seconds,
                        "ctx_compute_batch_seconds": ctx_values.get(
                            "compute_batch_seconds"
                        ),
                        "ctx_compute_batch_calls": ctx_values.get(
                            "compute_batch_calls"
                        ),
                        "transcoder_decoder_load_seconds": transcoder_values.get(
                            "decoder_load_seconds"
                        ),
                        "transcoder_decoder_load_count": transcoder_values.get(
                            "decoder_load_count"
                        ),
                        "transcoder_decoder_cache_hit_count": transcoder_values.get(
                            "decoder_cache_hit_count"
                        ),
                        "transcoder_decoder_cache_miss_count": transcoder_values.get(
                            "decoder_cache_miss_count"
                        ),
                        "transcoder_decoder_cache_eviction_count": transcoder_values.get(
                            "decoder_cache_eviction_count"
                        ),
                        "ctx_values_json": to_json_string(ctx_values),
                        "transcoder_values_json": to_json_string(transcoder_values),
                    }
                )
                summary["phase4_ctx_compute_batch_seconds_total"] += float(
                    ctx_values.get("compute_batch_seconds", 0.0) or 0.0
                )
                summary["phase4_transcoder_decoder_load_seconds_total"] += float(
                    transcoder_values.get("decoder_load_seconds", 0.0) or 0.0
                )

                for key, value in ctx_values.items():
                    layer_match = FEATURE_ATTR_LAYER_KEY_RE.fullmatch(key)
                    if layer_match and isinstance(value, (int, float)):
                        phase4_batch_layer_rows.append(
                            {
                                "scenario_root": str(scenario_root),
                                "scenario_name": scenario_name,
                                "stage": stage,
                                "cluster": cluster,
                                "batch_idx": batch_idx,
                                "total_batches": total_batches,
                                "layer": int(layer_match.group("layer")),
                                "elapsed_seconds": float(value),
                            }
                        )

            cache_event_match = CACHE_EVENT_RE.search(line)
            if cache_event_match:
                event = cache_event_match.group("event")
                count = int(cache_event_match.group("count"))
                resident_bytes = cache_event_match.group("resident_bytes")
                if event == "hit":
                    summary["log_max_decoder_cache_hit_count"] = _max_or_none(
                        summary["log_max_decoder_cache_hit_count"], count
                    )
                elif event == "miss":
                    summary["log_max_decoder_cache_miss_count"] = _max_or_none(
                        summary["log_max_decoder_cache_miss_count"], count
                    )
                elif event == "eviction":
                    summary["log_max_decoder_cache_eviction_count"] = _max_or_none(
                        summary["log_max_decoder_cache_eviction_count"], count
                    )
                if resident_bytes is not None:
                    summary["log_max_decoder_cache_resident_bytes"] = _max_or_none(
                        summary["log_max_decoder_cache_resident_bytes"],
                        int(resident_bytes),
                    )

            memory_match = MEMORY_RE.search(line)
            if memory_match:
                summary["log_peak_rss_gib"] = _max_or_none(
                    summary["log_peak_rss_gib"],
                    parse_memory_value_to_gib(memory_match.group("rss")),
                )
                summary["log_peak_cuda_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_allocated_gib"],
                    parse_memory_value_to_gib(memory_match.group("cuda_alloc")),
                )
                summary["log_peak_cuda_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_reserved_gib"],
                    parse_memory_value_to_gib(memory_match.group("cuda_reserved")),
                )
                summary["log_peak_cuda_peak_allocated_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_allocated_gib"],
                    parse_memory_value_to_gib(memory_match.group("cuda_peak_alloc")),
                )
                summary["log_peak_cuda_peak_reserved_gib"] = _max_or_none(
                    summary["log_peak_cuda_peak_reserved_gib"],
                    parse_memory_value_to_gib(memory_match.group("cuda_peak_reserved")),
                )

            cuda_oom_match = CUDA_OOM_RE.search(line)
            if cuda_oom_match:
                summary["cuda_oom_requested_gib"] = parse_memory_value_to_gib(
                    cuda_oom_match.group("requested")
                )
                summary["cuda_oom_total_gib"] = parse_memory_value_to_gib(
                    cuda_oom_match.group("total")
                )
                summary["cuda_oom_free_gib"] = parse_memory_value_to_gib(
                    cuda_oom_match.group("free")
                )
                summary["cuda_oom_in_use_gib"] = parse_memory_value_to_gib(
                    cuda_oom_match.group("in_use")
                )

    if phase4_batch_seconds:
        summary["phase4_batches_observed"] = len(phase4_batch_seconds)
        summary["phase4_batch_seconds_mean"] = mean(phase4_batch_seconds)
        summary["phase4_batch_seconds_max"] = max(phase4_batch_seconds)

    summary["failure_stage_guess"] = _guess_failure_stage(summary, result_status)
    return (
        summary,
        phase0_encode_rows,
        phase0_reconstruction_rows,
        phase4_batch_rows,
        phase4_batch_layer_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse run.log files from weekend exact chunked benchmarks"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root benchmark directory containing scenario result.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV files will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary_rows: list[dict[str, Any]] = []
    phase0_encode_rows: list[dict[str, Any]] = []
    phase0_reconstruction_rows: list[dict[str, Any]] = []
    phase4_batch_rows: list[dict[str, Any]] = []
    phase4_batch_layer_rows: list[dict[str, Any]] = []

    for result_path in sorted(args.input_root.glob("**/result.json")):
        scenario_root = result_path.parent
        result = read_json(result_path)
        scenario = (
            read_json(scenario_root / "scenario.json")
            if (scenario_root / "scenario.json").exists()
            else {}
        )
        log_path = Path(result.get("log_path") or scenario_root / "run.log")
        if not log_path.exists():
            continue

        parsed = parse_run_log(
            scenario_root=scenario_root,
            scenario_name=result.get("name")
            or scenario.get("name")
            or scenario_root.name,
            stage=result.get("stage") or scenario.get("stage"),
            cluster=_infer_cluster(scenario_root, scenario),
            result_status=result.get("status"),
            log_path=log_path,
        )
        summary_rows.append(parsed[0])
        phase0_encode_rows.extend(parsed[1])
        phase0_reconstruction_rows.extend(parsed[2])
        phase4_batch_rows.extend(parsed[3])
        phase4_batch_layer_rows.extend(parsed[4])

    write_csv(
        args.output_dir / "runlog_phase_summary.csv",
        summary_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "result_status",
            "gpu_name",
            "gpu_vram_gib",
            "precomputation_seconds",
            "forward_pass_seconds",
            "input_vector_build_seconds",
            "phase0_encode_seconds",
            "phase0_encode_total_active_features",
            "phase0_reconstruction_seconds",
            "phase0_reconstruction_total_chunks",
            "phase3_logit_attribution_seconds",
            "phase4_feature_attribution_seconds",
            "attribution_total_seconds",
            "phase4_batches_observed",
            "phase4_batch_seconds_mean",
            "phase4_batch_seconds_max",
            "log_peak_rss_gib",
            "log_peak_cuda_reserved_gib",
            "cuda_oom_requested_gib",
            "cuda_oom_total_gib",
            "cuda_oom_free_gib",
            "cuda_oom_in_use_gib",
            "failure_stage_guess",
            "log_path",
        ],
    )
    write_csv(
        args.output_dir / "phase0_encode_layers.csv",
        phase0_encode_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "layer",
            "active_features",
            "elapsed_seconds",
        ],
    )
    write_csv(
        args.output_dir / "phase0_reconstruction_layers.csv",
        phase0_reconstruction_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "layer",
            "active_features",
            "chunks",
            "elapsed_seconds",
        ],
    )
    write_csv(
        args.output_dir / "phase4_batches.csv",
        phase4_batch_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "batch_idx",
            "total_batches",
            "batch_seconds",
            "ctx_compute_batch_seconds",
            "transcoder_decoder_load_seconds",
            "transcoder_decoder_cache_hit_count",
            "transcoder_decoder_cache_miss_count",
            "transcoder_decoder_cache_eviction_count",
            "ctx_values_json",
            "transcoder_values_json",
        ],
    )
    write_csv(
        args.output_dir / "phase4_batch_layer_times.csv",
        phase4_batch_layer_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "batch_idx",
            "total_batches",
            "layer",
            "elapsed_seconds",
        ],
    )
    print(
        "Wrote "
        f"{len(summary_rows)} summaries, "
        f"{len(phase0_encode_rows)} encode-layer rows, "
        f"{len(phase0_reconstruction_rows)} reconstruction-layer rows, "
        f"{len(phase4_batch_rows)} phase4 batch rows, and "
        f"{len(phase4_batch_layer_rows)} phase4 batch-layer rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
