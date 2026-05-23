from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..io_utils import read_json
from .schemas import SCHEMA_VERSION, GeneratedToken, Trajectory, write_trajectory


def get_stop_token_ids(tokenizer: Any) -> set[int]:
    candidate_stop_ids = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ]
    try:
        end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    except Exception:  # pragma: no cover - tokenizer-specific fallback
        end_of_turn = None
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(end_of_turn, int) and end_of_turn != unk:
        candidate_stop_ids.append(end_of_turn)
    return {int(token_id) for token_id in candidate_stop_ids if token_id is not None}


def load_fixture_prompt(
    *,
    prompt_path: Path | None = None,
    fixture_catalog: Path | None = None,
    fixture_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    resolved_prompt_path = prompt_path
    if fixture_catalog is not None:
        catalog = read_json(fixture_catalog)
        fixtures = catalog.get("fixtures") if isinstance(catalog, dict) else None
        if not isinstance(fixtures, list):
            raise ValueError(f"fixture catalog has no fixtures list: {fixture_catalog}")
        matches = [row for row in fixtures if row.get("fixture_name") == fixture_name]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one fixture named {fixture_name!r} in {fixture_catalog}"
            )
        metadata = dict(matches[0])
        prompt_value = (
            metadata.get("prepared_prompt_file")
            or metadata.get("prompt_text_file")
            or metadata.get("prompt_file")
        )
        fixture_dir = metadata.get("fixture_dir") or metadata.get("path")
        if prompt_value is not None:
            resolved_prompt_path = Path(str(prompt_value))
        elif fixture_dir is not None:
            resolved_prompt_path = Path(str(fixture_dir)) / "prompt.txt"
        if resolved_prompt_path is not None and not resolved_prompt_path.is_absolute():
            resolved_prompt_path = fixture_catalog.parent / resolved_prompt_path
    if resolved_prompt_path is None:
        raise ValueError("provide --prompt-path or --fixture-catalog/--fixture-name")
    prompt_text = resolved_prompt_path.read_text(encoding="utf-8")
    meta_path = resolved_prompt_path.parent / "fixture_meta.json"
    if meta_path.exists():
        metadata.update(json.loads(meta_path.read_text(encoding="utf-8")))
    metadata.setdefault("prompt_path", str(resolved_prompt_path))
    return prompt_text, metadata


def build_trajectory(
    *,
    trajectory_id: str,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    token_texts: list[str],
    token_logprobs: list[float | None] | None = None,
    stop_token_ids: set[int] | None = None,
    prompt_text: str | None = None,
    fixture_metadata: dict[str, Any] | None = None,
) -> Trajectory:
    stop_token_ids = stop_token_ids or set()
    prompt_token_count = len(prompt_token_ids)
    generated_tokens: list[GeneratedToken] = []
    for index, token_id in enumerate(generated_token_ids):
        logprob = None if token_logprobs is None else token_logprobs[index]
        generated_tokens.append(
            {
                "generated_index": index,
                "absolute_token_position": prompt_token_count + index,
                "token_id": int(token_id),
                "token_text": token_texts[index],
                "logprob": logprob,
                "probability": None,
                "rank": None,
                "is_stop": int(token_id) in stop_token_ids,
            }
        )
    trajectory: Trajectory = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "prompt_token_count": prompt_token_count,
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "generated_tokens": generated_tokens,
    }
    if prompt_text is not None:
        trajectory["prompt_text"] = prompt_text
    if fixture_metadata:
        trajectory["fixture_metadata"] = fixture_metadata
    return trajectory


def generate_trajectory(
    *,
    output: Path,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
    include_prompt_text: bool = False,
    prompt_path: Path | None = None,
    fixture_catalog: Path | None = None,
    fixture_name: str | None = None,
    trajectory_id: str | None = None,
) -> Trajectory:
    """SLURM-only model generation entry point.

    Heavy imports and model loading intentionally live inside this function so CLI
    help/imports remain login-node safe. Do not call outside an allocated job.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if seed is not None:
        random.seed(seed)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    import trace_pipeline as base

    prompt_text, metadata = load_fixture_prompt(
        prompt_path=prompt_path,
        fixture_catalog=fixture_catalog,
        fixture_name=fixture_name,
    )
    model: Any = base.load_model(exact_chunked_decoder=True)
    tokenizer = model.tokenizer
    stop_token_ids = get_stop_token_ids(tokenizer)
    input_ids = model.ensure_tokenized(prompt_text).unsqueeze(0)
    prompt_token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
    generated_token_ids: list[int] = []
    token_texts: list[str] = []
    token_logprobs: list[float | None] = []
    for _ in range(max_new_tokens):
        token_result = base.generate_next_token(
            model,
            input_ids,
            temperature=temperature,
        )
        token_id = int(token_result["token_id"])
        generated_token_ids.append(token_id)
        token_texts.append(str(token_result.get("token_text", "")))
        token_logprobs.append(token_result.get("token_logprob"))
        input_ids = token_result["next_input_ids"]
        if token_id in stop_token_ids:
            break
    resolved_id = trajectory_id or str(
        metadata.get("fixture_name") or output.with_suffix("").name
    )
    trajectory = build_trajectory(
        trajectory_id=resolved_id,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        token_texts=token_texts,
        token_logprobs=token_logprobs,
        stop_token_ids=stop_token_ids,
        prompt_text=prompt_text if include_prompt_text else None,
        fixture_metadata=metadata,
    )
    write_trajectory(output, trajectory)
    return trajectory
