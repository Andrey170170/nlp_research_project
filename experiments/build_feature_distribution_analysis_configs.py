from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated")
DEFAULT_TOTAL_SAMPLES = 360
DEFAULT_SAMPLES_PER_JOB = 20
DEFAULT_SEED = 42


def build_selection(total_samples: int, *, seed: int) -> list[int]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if total_samples > len(dataset):
        raise ValueError(
            f"Requested {total_samples} samples but GSM8K test only has {len(dataset)}"
        )

    rng = random.Random(seed)
    return rng.sample(list(range(len(dataset))), total_samples)


def build_batches(
    indices: list[int], *, samples_per_job: int
) -> list[dict[str, object]]:
    batches: list[dict[str, object]] = []
    for batch_idx, start in enumerate(range(0, len(indices), samples_per_job)):
        batch_indices = indices[start : start + samples_per_job]
        batches.append(
            {
                "name": f"feature_distribution_batch_{batch_idx:03d}",
                "batch_index": batch_idx,
                "gsm8k_indices": batch_indices,
                "sample_count": len(batch_indices),
            }
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sample-selection and batch config files for feature distribution analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated selection/config JSON files will be written",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="How many GSM8K test prompts to sample in total",
    )
    parser.add_argument(
        "--samples-per-job",
        type=int,
        default=DEFAULT_SAMPLES_PER_JOB,
        help="How many prompts each Ascend job should process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic GSM8K sampling",
    )
    args = parser.parse_args()

    if args.total_samples <= 0:
        raise ValueError("--total-samples must be positive")
    if args.samples_per_job <= 0:
        raise ValueError("--samples-per-job must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampled_indices = build_selection(args.total_samples, seed=args.seed)
    batches = build_batches(sampled_indices, samples_per_job=args.samples_per_job)

    selection_payload = {
        "dataset": "openai/gsm8k:test",
        "seed": args.seed,
        "total_samples": args.total_samples,
        "samples_per_job": args.samples_per_job,
        "sampled_indices": sampled_indices,
    }
    batches_payload = {
        "analysis_name": "feature_distribution_analysis",
        "dataset": "openai/gsm8k:test",
        "seed": args.seed,
        "total_samples": args.total_samples,
        "samples_per_job": args.samples_per_job,
        "job_count": len(batches),
        "batches": batches,
    }

    selection_path = args.output_dir / "feature_distribution_analysis_selection.json"
    batches_path = args.output_dir / "feature_distribution_analysis_batches.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2))
    batches_path.write_text(json.dumps(batches_payload, indent=2))

    print(f"Wrote selection to {selection_path}")
    print(f"Wrote batches to {batches_path}")
    print(f"Total jobs: {len(batches)}")


if __name__ == "__main__":
    main()
