from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenRow:
    source: dict[str, str]
    char_start: int
    char_end: int

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.source["wave"],
            self.source["prompt_id"],
            self.source["label"],
            self.source["trajectory_name"],
        )

    @property
    def text(self) -> str:
        return self.source.get("target_token_text", "")

    @property
    def generated_index(self) -> int:
        return int(self.source["generated_index"])


def read_tokens(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_trajectory_texts(path: Path | None) -> dict[tuple[str, str, str, str], str]:
    if path is None or not path.exists():
        return {}
    texts: dict[tuple[str, str, str, str], str] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["wave"]),
                str(row["prompt_id"]),
                str(row["label"]),
                str(row["name"]),
            )
            texts[key] = row.get("generated_text", "")
    return texts


def reconstruct_token_offsets(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], list[TokenRow]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["wave"], row["prompt_id"], row["label"], row["trajectory_name"])
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, str, str, str], list[TokenRow]] = {}
    for key, group in grouped.items():
        pos = 0
        ordered = sorted(group, key=lambda r: int(r["generated_index"]))
        built = []
        for row in ordered:
            text = row.get("target_token_text", "")
            built.append(TokenRow(row, pos, pos + len(text)))
            pos += len(text)
        out[key] = built
    return out


def load_inventory(
    tokens_csv: Path, trajectory_texts_jsonl: Path | None = None
) -> tuple[
    dict[tuple[str, str, str, str], list[TokenRow]],
    dict[tuple[str, str, str, str], str],
    list[str],
]:
    grouped = reconstruct_token_offsets(read_tokens(tokens_csv))
    texts = read_trajectory_texts(trajectory_texts_jsonl)
    diagnostics = []
    for key, rows in grouped.items():
        reconstructed = "".join(row.text for row in rows)
        expected = texts.get(key)
        if expected is not None and reconstructed != expected:
            diagnostics.append(f"generated_text_mismatch:{key}:tokens={len(rows)}")
    return grouped, texts, diagnostics
