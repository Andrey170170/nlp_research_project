from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from extract_utils import (
    ensure_dir,
    flatten_dict,
    iter_jsonl,
    read_json,
    write_csv,
    write_jsonl,
)


DEFAULT_INPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/feature_distribution_analysis")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/feature_distribution_analysis")


def build_batch_row(batch_dir: Path) -> dict[str, Any]:
    summary = read_json(batch_dir / "summary.json")
    run_config = summary.get("run_config", read_json(batch_dir / "run_config.json"))
    return {
        "batch_dir": str(batch_dir),
        "batch_name": summary.get("batch_name")
        or run_config.get("batch_name")
        or batch_dir.name,
        "batch_index": summary.get("batch_index") or run_config.get("batch_index"),
        "analysis_name": summary.get("analysis_name")
        or run_config.get("analysis_name"),
        "duration_seconds": summary.get("duration_seconds"),
        "prompt_count": summary.get("prompt_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "max_total_active_features": summary.get("max_total_active_features"),
        "mean_total_active_features": summary.get("mean_total_active_features"),
        "max_token_count": summary.get("max_token_count"),
        "mean_token_count": summary.get("mean_token_count"),
        "decoder_chunk_size": run_config.get("decoder_chunk_size"),
        "lazy_encoder": run_config.get("lazy_encoder"),
        "lazy_decoder": run_config.get("lazy_decoder"),
        "started_at": run_config.get("started_at"),
        "results_jsonl": str(batch_dir / "results.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract prompt/layer/token tables from feature distribution analysis results"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing feature_distribution_batch_* subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV/JSONL files will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    prompt_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []

    for batch_dir in sorted(
        path for path in args.input_root.iterdir() if path.is_dir()
    ):
        summary_path = batch_dir / "summary.json"
        results_path = batch_dir / "results.jsonl"
        if not summary_path.exists() or not results_path.exists():
            continue

        batch_row = build_batch_row(batch_dir)
        batch_rows.append(batch_row)
        run_config = (
            read_json(batch_dir / "run_config.json")
            if (batch_dir / "run_config.json").exists()
            else {}
        )

        for result in iter_jsonl(results_path):
            token_count = result.get("token_count") or result.get("prompt_token_count")
            total_active_features = result.get("total_active_features")
            active_features_per_token = result.get("active_features_per_token")
            if (
                active_features_per_token is None
                and isinstance(token_count, int)
                and token_count > 0
                and isinstance(total_active_features, int)
            ):
                active_features_per_token = total_active_features / token_count

            prompt_row = {
                "batch_dir": str(batch_dir),
                "batch_name": result.get("batch_name") or batch_row["batch_name"],
                "batch_index": result.get("batch_index") or batch_row["batch_index"],
                "analysis_name": result.get("analysis_name")
                or batch_row["analysis_name"],
                "gsm8k_index": result.get("gsm8k_index"),
                "prompt_index_within_batch": result.get("prompt_index_within_batch"),
                "status": result.get("status"),
                "token_count": token_count,
                "prompt_token_count": result.get("prompt_token_count") or token_count,
                "total_active_features": total_active_features,
                "active_features_per_token": active_features_per_token,
                "phase0_encode_seconds": result.get("phase0_encode_seconds"),
                "phase0_reconstruction_seconds": result.get(
                    "phase0_reconstruction_seconds"
                ),
                "duration_seconds": result.get("duration_seconds"),
                "decoder_chunk_size": run_config.get("decoder_chunk_size"),
                "lazy_encoder": run_config.get("lazy_encoder"),
                "lazy_decoder": run_config.get("lazy_decoder"),
                "error": result.get("error"),
                **flatten_dict(
                    result.get("resource_snapshot_before"), prefix="resource_before_"
                ),
                **flatten_dict(
                    result.get("resource_snapshot_after"), prefix="resource_after_"
                ),
            }
            prompt_rows.append(prompt_row)

            for layer_idx, active_features in enumerate(
                result.get("active_features_by_layer", []) or []
            ):
                layer_rows.append(
                    {
                        "batch_dir": str(batch_dir),
                        "batch_name": prompt_row["batch_name"],
                        "batch_index": prompt_row["batch_index"],
                        "analysis_name": prompt_row["analysis_name"],
                        "gsm8k_index": prompt_row["gsm8k_index"],
                        "prompt_index_within_batch": prompt_row[
                            "prompt_index_within_batch"
                        ],
                        "layer": layer_idx,
                        "active_features": active_features,
                    }
                )

            for token_idx, active_features in enumerate(
                result.get("active_features_by_token", []) or []
            ):
                token_rows.append(
                    {
                        "batch_dir": str(batch_dir),
                        "batch_name": prompt_row["batch_name"],
                        "batch_index": prompt_row["batch_index"],
                        "analysis_name": prompt_row["analysis_name"],
                        "gsm8k_index": prompt_row["gsm8k_index"],
                        "prompt_index_within_batch": prompt_row[
                            "prompt_index_within_batch"
                        ],
                        "token_position": token_idx,
                        "active_features": active_features,
                    }
                )

    write_csv(
        args.output_dir / "feature_distribution_prompts.csv",
        prompt_rows,
        preferred_headers=[
            "batch_dir",
            "batch_name",
            "batch_index",
            "analysis_name",
            "gsm8k_index",
            "prompt_index_within_batch",
            "status",
            "token_count",
            "prompt_token_count",
            "total_active_features",
            "active_features_per_token",
            "phase0_encode_seconds",
            "phase0_reconstruction_seconds",
            "duration_seconds",
            "decoder_chunk_size",
            "lazy_encoder",
            "lazy_decoder",
            "error",
        ],
    )
    write_jsonl(args.output_dir / "feature_distribution_prompts.jsonl", prompt_rows)
    write_csv(
        args.output_dir / "feature_distribution_layers.csv",
        layer_rows,
        preferred_headers=[
            "batch_dir",
            "batch_name",
            "batch_index",
            "analysis_name",
            "gsm8k_index",
            "prompt_index_within_batch",
            "layer",
            "active_features",
        ],
    )
    write_csv(
        args.output_dir / "feature_distribution_tokens.csv",
        token_rows,
        preferred_headers=[
            "batch_dir",
            "batch_name",
            "batch_index",
            "analysis_name",
            "gsm8k_index",
            "prompt_index_within_batch",
            "token_position",
            "active_features",
        ],
    )
    write_csv(
        args.output_dir / "feature_distribution_batches.csv",
        batch_rows,
        preferred_headers=[
            "batch_dir",
            "batch_name",
            "batch_index",
            "analysis_name",
            "duration_seconds",
            "prompt_count",
            "success_count",
            "failure_count",
            "max_total_active_features",
            "mean_total_active_features",
            "max_token_count",
            "mean_token_count",
            "decoder_chunk_size",
            "lazy_encoder",
            "lazy_decoder",
            "started_at",
            "results_jsonl",
        ],
    )
    print(
        "Wrote "
        f"{len(batch_rows)} batch rows, {len(prompt_rows)} prompt rows, "
        f"{len(layer_rows)} layer rows, and {len(token_rows)} token rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
