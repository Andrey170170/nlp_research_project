"""Prompt selection and project provenance records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    metadata: dict[str, Any]


def load_prompts(model: Any, scenario: Mapping[str, Any]) -> list[PreparedPrompt]:
    prepared_file = scenario.get("prepared_prompt_file")
    if prepared_file is not None:
        examples = [_load_prepared_example(prepared_file, scenario.get("prepared_prompt_meta_file"))]
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        indices = scenario.get("gsm8k_indices")
        if not indices:
            indices = range(min(int(scenario.get("prompts", 1)), len(dataset)))
        examples = [{**dataset[int(index)], "gsm8k_index": int(index)} for index in indices]

    prompts = []
    for example in examples:
        text = example.get("prompt_text") or _format_prompt(
            model.tokenizer, example["question"]
        )
        token_count = int(model.ensure_tokenized(text).shape[0])
        prompts.append(
            PreparedPrompt(
                text=text,
                metadata=_prompt_metadata(example, text, token_count),
            )
        )
    return prompts


def write_prompt_record(output_dir: Path, index: int, prompt: PreparedPrompt) -> None:
    prompt_dir = output_dir / f"prompt_{index:03d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt_meta.json").write_text(json.dumps(prompt.metadata, indent=2))


def _load_prepared_example(prompt_file: Any, metadata_file: Any) -> dict[str, Any]:
    metadata = {} if metadata_file is None else json.loads(Path(metadata_file).read_text())
    return {
        "question": metadata.get("question", ""),
        "answer": metadata.get("ground_truth_answer", ""),
        "gsm8k_index": metadata.get("gsm8k_index"),
        "prompt_text": Path(prompt_file).read_text(),
        "prompt_source": metadata.get("prompt_source", "prepared_prompt"),
        "fixture_name": metadata.get("fixture_name"),
        "fixture_kind": metadata.get("fixture_kind", "prepared_prompt"),
        "prompt_token_count": metadata.get("prompt_token_count"),
        "prepared_prompt_file": str(prompt_file),
        "prepared_prompt_meta_file": metadata_file,
        "prepared_prompt_metadata": metadata,
    }


def _format_prompt(tokenizer: Any, question: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nPlease solve this step by step and end with "
                    "'Final answer: <number>'."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _prompt_metadata(
    example: Mapping[str, Any], text: str, initial_token_count: int
) -> dict[str, Any]:
    record = {
        "gsm8k_index": example.get("gsm8k_index"),
        "question": example.get("question", ""),
        "ground_truth_answer": example.get("answer", ""),
        "prompt_text": text,
        "prompt_source": example.get("prompt_source", "gsm8k"),
        "fixture_name": example.get("fixture_name"),
        "fixture_kind": example.get("fixture_kind"),
        "prompt_token_count": int(example.get("prompt_token_count") or initial_token_count),
        "initial_input_token_count": initial_token_count,
    }
    for key in (
        "prepared_prompt_file",
        "prepared_prompt_meta_file",
        "prepared_prompt_metadata",
    ):
        if example.get(key) is not None:
            record[key] = example[key]
    return record
