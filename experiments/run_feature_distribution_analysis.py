from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from circuit_tracer.attribution.attribute import attribute_phase0_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import trace_pipeline as base  # noqa: E402


DEFAULT_BATCHES_FILE = (
    Path(__file__).with_name("generated") / "feature_distribution_analysis_batches.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/fs/scratch/PAS3272/kopanev.1/feature_distribution_analysis"
)


def load_batches(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_prompt_result(result: dict[str, Any]) -> dict[str, Any]:
    total_active_features = result.get("total_active_features")
    token_count = result.get("token_count")
    active_features_per_token = None
    if (
        isinstance(total_active_features, int)
        and isinstance(token_count, int)
        and token_count > 0
    ):
        active_features_per_token = total_active_features / token_count

    return {
        "gsm8k_index": result["gsm8k_index"],
        "status": result["status"],
        "token_count": token_count,
        "total_active_features": total_active_features,
        "active_features_per_token": active_features_per_token,
        "phase0_encode_seconds": result.get("phase0_encode_seconds"),
        "phase0_reconstruction_seconds": result.get("phase0_reconstruction_seconds"),
    }


def build_job_summary(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [result for result in prompt_results if result["status"] == "success"]
    total_active_features = [
        result["total_active_features"]
        for result in successes
        if isinstance(result.get("total_active_features"), int)
    ]
    token_counts = [
        result["token_count"]
        for result in successes
        if isinstance(result.get("token_count"), int)
    ]
    return {
        "prompt_count": len(prompt_results),
        "success_count": len(successes),
        "failure_count": len(prompt_results) - len(successes),
        "max_total_active_features": max(total_active_features)
        if total_active_features
        else None,
        "mean_total_active_features": (
            sum(total_active_features) / len(total_active_features)
            if total_active_features
            else None
        ),
        "max_token_count": max(token_counts) if token_counts else None,
        "mean_token_count": sum(token_counts) / len(token_counts)
        if token_counts
        else None,
        "results": [summarize_prompt_result(result) for result in prompt_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase-0 feature distribution analysis for a batch of GSM8K prompts"
    )
    parser.add_argument(
        "--batches-file",
        type=Path,
        default=DEFAULT_BATCHES_FILE,
        help="JSON file containing feature-distribution analysis batches",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        required=True,
        help="Which batch entry from the batches file to execute",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Scratch/output directory where per-batch results will be written",
    )
    parser.add_argument(
        "--decoder-chunk-size",
        type=int,
        default=256,
        help="Decoder chunk size used while loading the model",
    )
    parser.add_argument(
        "--no-lazy-encoder",
        action="store_true",
        help="Eagerly load encoder weights",
    )
    parser.add_argument(
        "--no-lazy-decoder",
        action="store_true",
        help="Eagerly load decoder weights",
    )
    args = parser.parse_args()

    batches_payload = load_batches(args.batches_file)
    batches = batches_payload["batches"]
    if not (0 <= args.batch_index < len(batches)):
        raise IndexError(
            f"--batch-index {args.batch_index} out of range for {len(batches)} batches"
        )

    batch = batches[args.batch_index]
    batch_name = str(batch["name"])
    gsm8k_indices = [int(index) for index in batch["gsm8k_indices"]]
    output_dir = args.output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "analysis_name": batches_payload.get(
            "analysis_name", "feature_distribution_analysis"
        ),
        "batch_name": batch_name,
        "batch_index": int(batch["batch_index"]),
        "gsm8k_indices": gsm8k_indices,
        "decoder_chunk_size": args.decoder_chunk_size,
        "lazy_encoder": not args.no_lazy_encoder,
        "lazy_decoder": not args.no_lazy_decoder,
        "output_dir": str(output_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(f"Batch: {batch_name}")
    print(f"Prompt count: {len(gsm8k_indices)}")
    print(f"Output dir: {output_dir}")

    model = base.load_model(
        lazy_encoder=not args.no_lazy_encoder,
        lazy_decoder=not args.no_lazy_decoder,
        decoder_chunk_size=args.decoder_chunk_size,
    )
    examples = base.load_gsm8k_examples(len(gsm8k_indices), indices=gsm8k_indices)

    prompt_results: list[dict[str, Any]] = []
    results_jsonl = output_dir / "results.jsonl"
    results_jsonl.write_text("")
    batch_start = time.time()

    for prompt_idx, example in enumerate(examples):
        prompt_start = time.time()
        prompt_text = base.format_prompt(model.tokenizer, example["question"])  # type: ignore[unresolved-attribute]
        prompt_result: dict[str, Any] = {
            "analysis_name": batches_payload.get(
                "analysis_name", "feature_distribution_analysis"
            ),
            "batch_name": batch_name,
            "batch_index": int(batch["batch_index"]),
            "prompt_index_within_batch": prompt_idx,
            "gsm8k_index": int(example["gsm8k_index"]),
            "question": example["question"],
            "ground_truth_answer": example["answer"],
            "status": "success",
            "resource_snapshot_before": base.capture_resource_snapshot(),
        }

        try:
            stats = attribute_phase0_stats(prompt_text, cast(Any, model))
        except torch.OutOfMemoryError as exc:
            prompt_result.update(
                {
                    "status": "oom",
                    "error": str(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - defensive runtime capture
            prompt_result.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            prompt_result.update(stats)

        prompt_result["duration_seconds"] = round(time.time() - prompt_start, 2)
        prompt_result["resource_snapshot_after"] = base.capture_resource_snapshot()
        prompt_results.append(prompt_result)

        with results_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prompt_result) + "\n")

        print(
            f"[{prompt_idx + 1:02d}/{len(examples):02d}] GSM8K {prompt_result['gsm8k_index']} "
            f"status={prompt_result['status']} duration={prompt_result['duration_seconds']:.2f}s "
            f"features={prompt_result.get('total_active_features')}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "analysis_name": batches_payload.get(
            "analysis_name", "feature_distribution_analysis"
        ),
        "batch_name": batch_name,
        "batch_index": int(batch["batch_index"]),
        "duration_seconds": round(time.time() - batch_start, 2),
        "run_config": run_config,
        **build_job_summary(prompt_results),
    }
    (output_dir / "results.json").write_text(json.dumps(prompt_results, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
