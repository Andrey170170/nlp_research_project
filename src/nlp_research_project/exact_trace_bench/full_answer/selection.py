from __future__ import annotations

import re
from typing import Any

from .schemas import SCHEMA_VERSION, Trajectory, validate_trajectory

NUMERIC_RE = re.compile(r"[-+]?\d|\d[\d,]*(?:\.\d+)?")


def parse_indices_csv(text: str | None) -> list[int]:
    if not text:
        return []
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def select_tokens(
    trajectory: Trajectory,
    *,
    explicit_indices: list[int] | None = None,
    uniform_every_k: int | None = None,
    include_numeric: bool = False,
    include_final_answer: bool = False,
    high_surprisal_top_k: int | None = None,
) -> dict[str, Any]:
    """Build a merged trace selection from simple login-safe policies.

    The first final-answer heuristic intentionally selects the last non-stop
    generated token (or the last token if all are marked stop). This captures the
    answer-final position without requiring tokenizer/model imports.
    """

    validate_trajectory(trajectory)
    tokens = trajectory["generated_tokens"]
    reasons: dict[int, list[str]] = {}

    def add(index: int, reason: str) -> None:
        if index < 0 or index >= len(tokens):
            raise ValueError(f"selected generated index out of bounds: {index}")
        reasons.setdefault(index, [])
        if reason not in reasons[index]:
            reasons[index].append(reason)

    for index in explicit_indices or []:
        add(index, "explicit")

    if uniform_every_k is not None:
        if uniform_every_k <= 0:
            raise ValueError("uniform_every_k must be positive")
        for index in range(uniform_every_k, len(tokens), uniform_every_k):
            add(index, "uniform_every_k")

    if include_numeric:
        for index, token in enumerate(tokens):
            if NUMERIC_RE.search(str(token.get("token_text", ""))):
                add(index, "numeric")

    if include_final_answer and tokens:
        final_index = len(tokens) - 1
        for index in range(len(tokens) - 1, -1, -1):
            if not tokens[index].get("is_stop", False):
                final_index = index
                break
        add(final_index, "final_answer")

    if high_surprisal_top_k is not None:
        if high_surprisal_top_k < 0:
            raise ValueError("high_surprisal_top_k must be non-negative")
        ranked = []
        for index, token in enumerate(tokens):
            logprob = token.get("logprob")
            if logprob is not None:
                ranked.append((float(logprob), index))
        for _, index in sorted(ranked)[:high_surprisal_top_k]:
            add(index, "high_surprisal")

    selected = sorted(reasons)
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory["trajectory_id"],
        "selection_policy": {
            "explicit_indices": explicit_indices or [],
            "include_final_answer": include_final_answer,
            "include_numeric": include_numeric,
            "high_surprisal_top_k": high_surprisal_top_k,
            "uniform_every_k": uniform_every_k,
        },
        "selected_indices": selected,
        "selection_reasons": {str(index): reasons[index] for index in selected},
    }
