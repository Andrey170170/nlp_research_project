from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import trace_pipeline as base  # noqa: E402


DEFAULT_TARGET_SPEC = Path(__file__).with_name(
    "weekend_exact_chunked_fixture_targets.json"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).with_name("generated") / "weekend_exact_chunked_fixtures"
)


def load_target_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def get_stop_token_ids(tokenizer) -> set[int]:
    candidate_stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn != tokenizer.unk_token_id:
        candidate_stop_ids.append(end_of_turn)
    return {token_id for token_id in candidate_stop_ids if token_id is not None}


def generate_greedy_completion(
    model,
    prompt_text: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    stop_token_ids = get_stop_token_ids(tokenizer)
    input_ids = model.ensure_tokenized(prompt_text).unsqueeze(0)
    prompt_token_ids = input_ids[0].tolist()
    generated_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        token_result = base.generate_next_token(model, input_ids, temperature=0.0)
        next_token_id = token_result["token_id"]
        generated_token_ids.append(next_token_id)
        input_ids = token_result["next_input_ids"]
        if next_token_id in stop_token_ids:
            break

    return {
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "completion_text": tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        ),
        "full_input_text": tokenizer.decode(input_ids[0], skip_special_tokens=False),
    }


def choose_late_prefix_token_count(
    *,
    prompt_token_count: int,
    completion_token_count: int,
    fraction_min: float,
    fraction_max: float,
    prompt_token_multiplier_cap: float,
) -> tuple[int, dict[str, Any]]:
    if completion_token_count <= 0:
        return 0, {
            "selection_reason": "empty_completion",
            "target_fraction": 0.0,
            "fraction_token_target": 0,
            "fraction_token_min": 0,
            "fraction_token_max": 0,
            "multiplier_cap_total_tokens": int(
                round(prompt_token_multiplier_cap * prompt_token_count)
            ),
        }

    target_fraction = (fraction_min + fraction_max) / 2
    fraction_token_target = max(1, round(target_fraction * completion_token_count))
    fraction_token_min = max(1, math.ceil(fraction_min * completion_token_count))
    fraction_token_max = max(
        fraction_token_min, math.floor(fraction_max * completion_token_count)
    )
    multiplier_cap_total_tokens = max(
        prompt_token_count + 1,
        int(round(prompt_token_multiplier_cap * prompt_token_count)),
    )
    multiplier_cap_generated_tokens = max(
        1, multiplier_cap_total_tokens - prompt_token_count
    )
    strict_completion_cap = max(1, completion_token_count - 1)

    selected_token_count = min(
        fraction_token_target,
        multiplier_cap_generated_tokens,
        strict_completion_cap,
    )
    selection_reason = "fraction_target"
    if selected_token_count == multiplier_cap_generated_tokens:
        selection_reason = "prompt_multiplier_cap"
    elif (
        selected_token_count == strict_completion_cap
        and strict_completion_cap < fraction_token_target
    ):
        selection_reason = "keep_completion_nonterminal"

    return selected_token_count, {
        "selection_reason": selection_reason,
        "target_fraction": target_fraction,
        "fraction_token_target": fraction_token_target,
        "fraction_token_min": fraction_token_min,
        "fraction_token_max": fraction_token_max,
        "multiplier_cap_total_tokens": multiplier_cap_total_tokens,
        "multiplier_cap_generated_tokens": multiplier_cap_generated_tokens,
    }


def write_fixture_files(
    fixture_dir: Path,
    *,
    prompt_text: str,
    metadata: dict[str, Any],
    greedy_completion_text: str | None = None,
    late_prefix_text: str | None = None,
) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "prompt.txt").write_text(prompt_text)
    (fixture_dir / "fixture_meta.json").write_text(json.dumps(metadata, indent=2))
    if greedy_completion_text is not None:
        (fixture_dir / "greedy_completion.txt").write_text(greedy_completion_text)
    if late_prefix_text is not None:
        (fixture_dir / "late_prefix_only.txt").write_text(late_prefix_text)


def build_fixture_catalog(
    model,
    *,
    target_spec: dict[str, Any],
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    gsm8k_indices = [int(index) for index in target_spec["gsm8k_indices"]]
    examples = base.load_gsm8k_examples(len(gsm8k_indices), indices=gsm8k_indices)
    tokenizer = model.tokenizer

    catalog: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_spec_file": str(DEFAULT_TARGET_SPEC),
        "output_dir": str(output_dir),
        "fixtures": [],
    }

    for example in examples:
        gsm8k_index = int(example["gsm8k_index"])
        question = example["question"]
        ground_truth_answer = example["answer"]
        prompt_text = base.format_prompt(tokenizer, question)
        prompt_token_ids = model.ensure_tokenized(prompt_text)  # type: ignore[unresolved-attribute]
        prompt_token_count = int(prompt_token_ids.shape[0])

        greedy_completion = generate_greedy_completion(
            model,
            prompt_text,
            max_new_tokens=max_new_tokens,
        )
        generated_token_ids = greedy_completion["generated_token_ids"]
        completion_token_count = len(generated_token_ids)
        late_prefix_token_count, truncation_meta = choose_late_prefix_token_count(
            prompt_token_count=prompt_token_count,
            completion_token_count=completion_token_count,
            fraction_min=float(target_spec["late_prefix_fraction_min"]),
            fraction_max=float(target_spec["late_prefix_fraction_max"]),
            prompt_token_multiplier_cap=float(
                target_spec["prompt_token_multiplier_cap"]
            ),
        )

        late_prefix_token_ids = generated_token_ids[:late_prefix_token_count]
        late_prompt_token_ids = prompt_token_ids.tolist() + late_prefix_token_ids
        late_prompt_text = tokenizer.decode(
            late_prompt_token_ids, skip_special_tokens=False
        )
        late_prefix_text = tokenizer.decode(
            late_prefix_token_ids, skip_special_tokens=False
        )
        late_fraction = (
            (late_prefix_token_count / completion_token_count)
            if completion_token_count > 0
            else 0.0
        )

        base_fixture_name = f"{gsm8k_index}_base"
        late_fixture_name = f"{gsm8k_index}_late"
        base_fixture_dir = output_dir / base_fixture_name
        late_fixture_dir = output_dir / late_fixture_name

        base_metadata = {
            "fixture_name": base_fixture_name,
            "fixture_kind": "base",
            "prompt_source": "gsm8k",
            "gsm8k_index": gsm8k_index,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "prompt_token_count": prompt_token_count,
            "initial_input_token_count": prompt_token_count,
            "greedy_completion_token_count": completion_token_count,
        }
        write_fixture_files(
            base_fixture_dir,
            prompt_text=prompt_text,
            metadata=base_metadata,
            greedy_completion_text=greedy_completion["completion_text"],
        )

        late_metadata = {
            "fixture_name": late_fixture_name,
            "fixture_kind": "late_prefix",
            "prompt_source": "prepared_prefix",
            "gsm8k_index": gsm8k_index,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "prompt_token_count": prompt_token_count,
            "initial_input_token_count": len(late_prompt_token_ids),
            "late_prefix_token_count": late_prefix_token_count,
            "late_prefix_fraction_of_completion_tokens": late_fraction,
            "greedy_completion_token_count": completion_token_count,
            "truncation": truncation_meta,
        }
        write_fixture_files(
            late_fixture_dir,
            prompt_text=late_prompt_text,
            metadata=late_metadata,
            greedy_completion_text=greedy_completion["completion_text"],
            late_prefix_text=late_prefix_text,
        )

        catalog["fixtures"].extend(
            [
                {
                    **base_metadata,
                    "prompt_text_file": str(base_fixture_dir / "prompt.txt"),
                    "prepared_prompt_file": str(base_fixture_dir / "prompt.txt"),
                    "prepared_prompt_meta_file": str(
                        base_fixture_dir / "fixture_meta.json"
                    ),
                },
                {
                    **late_metadata,
                    "prompt_text_file": str(late_fixture_dir / "prompt.txt"),
                    "prepared_prompt_file": str(late_fixture_dir / "prompt.txt"),
                    "prepared_prompt_meta_file": str(
                        late_fixture_dir / "fixture_meta.json"
                    ),
                },
            ]
        )

        print(
            f"Prepared fixtures for GSM8K {gsm8k_index}: "
            f"prompt_tokens={prompt_token_count} completion_tokens={completion_token_count} "
            f"late_prefix_tokens={late_prefix_token_count}"
        )

    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic late-prefix fixtures for the weekend exact chunked benchmark"
    )
    parser.add_argument(
        "--target-spec-file",
        type=Path,
        default=DEFAULT_TARGET_SPEC,
        help="JSON file describing GSM8K fixture targets and truncation policy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prepared prompt fixtures and the catalog will be written",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for deterministic completion length",
    )
    parser.add_argument(
        "--decoder-chunk-size",
        type=int,
        default=256,
        help="Fork-native decoder chunk size used while loading the tracing model",
    )
    parser.add_argument(
        "--cross-batch-decoder-cache-bytes",
        type=int,
        default=None,
        help="Optional cache budget when loading the tracing model for fixture generation",
    )
    parser.add_argument(
        "--no-lazy-encoder",
        action="store_true",
        help="Eagerly load encoder weights when generating fixtures",
    )
    parser.add_argument(
        "--no-lazy-decoder",
        action="store_true",
        help="Eagerly load decoder weights when generating fixtures",
    )
    args = parser.parse_args()

    target_spec = load_target_spec(args.target_spec_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(
        target_spec["max_new_tokens"]
        if args.max_new_tokens is None
        else args.max_new_tokens
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = base.load_model(
        lazy_encoder=not args.no_lazy_encoder,
        lazy_decoder=not args.no_lazy_decoder,
        decoder_chunk_size=args.decoder_chunk_size,
        cross_batch_decoder_cache_bytes=args.cross_batch_decoder_cache_bytes,
    )
    catalog = build_fixture_catalog(
        model,
        target_spec=target_spec,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
    )

    catalog_path = output_dir / "fixture_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote fixture catalog to {catalog_path}")


if __name__ == "__main__":
    main()
