"""Filesystem ownership for one generated completion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompletionWorkspace:
    root: Path
    prompt_id: str
    completion_id: str

    @classmethod
    def create(
        cls,
        output_dir: Path,
        *,
        prompt_index: int,
        completion_index: int,
    ) -> CompletionWorkspace:
        prompt_id = f"prompt_{prompt_index:03d}"
        completion_id = f"completion_{completion_index:03d}"
        root = output_dir / prompt_id / completion_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, prompt_id=prompt_id, completion_id=completion_id)

    @property
    def live_telemetry_path(self) -> Path:
        return self.root / "telemetry.live.jsonl"

    @property
    def telemetry_path(self) -> Path:
        return self.root / "telemetry.jsonl"

    def step_path(self, step_index: int) -> Path:
        return self.root / f"step_{step_index:03d}.npz"

    def sidecar_path(self, step_index: int, suffix: str) -> Path:
        return self.root / f"step_{step_index:03d}_{suffix}.npz"

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def append_jsonl(self, path: Path, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")
        return len(records)
