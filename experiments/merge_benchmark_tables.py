from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from extract_utils import ensure_dir, write_csv, write_jsonl


DEFAULT_INPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in {None, "", "None"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def build_slurm_lookup(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        scenario_root = row.get("scenario_root") or ""
        scenario_name = row.get("scenario_name") or ""
        if scenario_root:
            by_root[scenario_root].append(row)
        if scenario_name:
            by_name[scenario_name].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        families = sorted(
            {row.get("failure_family") for row in group if row.get("failure_family")}
        )
        job_ids = sorted({row.get("job_id") for row in group if row.get("job_id")})
        nodes = sorted({row.get("node") for row in group if row.get("node")})
        excerpts = [row.get("err_excerpt") for row in group if row.get("err_excerpt")]
        return {
            "slurm_err_file_count": len(group),
            "slurm_failure_families": "|".join(families) if families else None,
            "slurm_any_ram_oom": any(
                row.get("failure_family") == "ram_oom" for row in group
            ),
            "slurm_any_cuda_oom": any(
                row.get("failure_family") == "cuda_oom" for row in group
            ),
            "slurm_any_timeout": any(
                row.get("failure_family") == "timeout" for row in group
            ),
            "slurm_oom_kill_event_count": sum(
                _to_int(row.get("oom_kill_event_count")) or 0 for row in group
            ),
            "slurm_job_ids": "|".join(job_ids) if job_ids else None,
            "slurm_nodes": "|".join(nodes) if nodes else None,
            "slurm_err_excerpt": excerpts[0] if excerpts else None,
        }

    return (
        {key: summarize(group) for key, group in by_root.items()},
        {key: summarize(group) for key, group in by_name.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge weekend benchmark extractor outputs into one enriched table"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing benchmark_index.csv and companion extractor outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory where merged CSV/JSONL outputs will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    benchmark_rows = read_csv_rows(args.input_dir / "benchmark_index.csv")
    runlog_rows = {
        row["scenario_root"]: row
        for row in read_csv_rows(args.input_dir / "runlog_phase_summary.csv")
        if row.get("scenario_root")
    }
    slurm_by_root, slurm_by_name = build_slurm_lookup(
        read_csv_rows(args.input_dir / "slurm_err_summary.csv")
    )

    enriched_rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        scenario_root = row.get("scenario_root") or ""
        scenario_name = row.get("scenario_name") or ""
        merged = dict(row)
        runlog = runlog_rows.get(scenario_root, {})
        slurm = slurm_by_root.get(scenario_root) or slurm_by_name.get(scenario_name, {})
        for key, value in runlog.items():
            if key in {
                "scenario_root",
                "scenario_name",
                "stage",
                "cluster",
                "result_status",
                "log_path",
            }:
                continue
            merged[key] = value
        merged.update(slurm)

        max_active_features = _to_float(merged.get("max_active_features"))
        duration_seconds = _to_float(merged.get("duration_seconds"))
        prompt_tokens = _to_int(merged.get("prompt_token_count"))
        initial_input_tokens = _to_int(merged.get("initial_input_token_count"))
        fixture_kind = merged.get("fixture_kind")

        merged["runtime_per_million_active_features"] = (
            None
            if duration_seconds is None or max_active_features in {None, 0.0}
            else duration_seconds / (max_active_features / 1_000_000.0)
        )

        if fixture_kind == "late_prefix":
            prompt_regime = "late_prefix"
        elif (initial_input_tokens or prompt_tokens or 0) >= 150:
            prompt_regime = "long"
        else:
            prompt_regime = "base_or_short"
        merged["prompt_regime"] = prompt_regime

        status = merged.get("status")
        slurm_any_ram_oom = _to_bool(merged.get("slurm_any_ram_oom"))
        slurm_any_timeout = _to_bool(merged.get("slurm_any_timeout"))
        cuda_oom_seen = _to_float(merged.get("cuda_oom_requested_gib")) is not None
        if status == "success":
            failure_family = "success"
        elif slurm_any_ram_oom:
            failure_family = "ram_oom"
        elif status == "oom" or cuda_oom_seen:
            failure_family = "cuda_oom"
        elif status == "timeout" or slurm_any_timeout:
            failure_family = "timeout"
        else:
            failure_family = "other_fail"
        merged["failure_family_final"] = failure_family

        enriched_rows.append(merged)

    write_csv(
        args.output_dir / "benchmark_enriched.csv",
        enriched_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "status",
            "failure_family_final",
            "slurm_failure_families",
            "slurm_any_ram_oom",
            "returncode",
            "duration_seconds",
            "runtime_per_million_active_features",
            "gsm8k_index",
            "prompt_source",
            "fixture_name",
            "fixture_kind",
            "prompt_regime",
            "attribution_batch_size",
            "feature_batch_size",
            "logit_batch_size",
            "decoder_chunk_size",
            "decoder_cache_gib",
            "prompt_token_count",
            "initial_input_token_count",
            "max_active_features",
            "phase0_encode_seconds",
            "phase0_reconstruction_seconds",
            "precomputation_seconds",
            "phase3_logit_attribution_seconds",
            "phase4_feature_attribution_seconds",
            "attribution_total_seconds",
            "phase4_batches_observed",
            "phase4_batch_seconds_mean",
            "peak_rss_gib",
            "log_peak_rss_gib",
            "peak_cuda_reserved_gib",
            "log_peak_cuda_reserved_gib",
            "cuda_oom_requested_gib",
            "failure_stage_guess",
            "is_special_case",
            "special_case_label",
            "run_log_path",
        ],
    )
    write_jsonl(args.output_dir / "benchmark_enriched.jsonl", enriched_rows)
    print(f"Wrote {len(enriched_rows)} merged benchmark rows to {args.output_dir}")


if __name__ == "__main__":
    main()
