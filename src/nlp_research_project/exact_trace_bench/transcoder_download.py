from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .transcoder_config import TranscoderLoadConfig, transcoder_config_to_json


def load_env_file(path: Path | None) -> bool:
    """Load simple KEY=VALUE entries without printing secrets.

    This intentionally avoids shell-sourcing `.env`: only plain assignments are
    accepted, optional quotes are stripped, and existing environment variables win.
    """

    if path is None or not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    return True


def transcoder_allow_patterns(config: TranscoderLoadConfig) -> list[str]:
    if config.transcoder_architecture == "clt":
        if not config.clt_subfolder:
            raise ValueError("CLT download requires clt_subfolder")
        return [f"{config.clt_subfolder}/params_layer_*.safetensors"]
    if not config.plt_subfolder_template:
        raise ValueError("PLT download requires plt_subfolder_template")
    if isinstance(config.layer_count, int):
        return [
            config.plt_subfolder_template.format(layer=layer)
            for layer in range(config.layer_count)
        ]
    return [config.plt_subfolder_template.format(layer="*")]


def _count_matching_files(local_dir: Path, allow_patterns: list[str]) -> int:
    count = 0
    for pattern in allow_patterns:
        if "*" in pattern:
            count += sum(1 for path in local_dir.glob(pattern) if path.is_file())
        elif (local_dir / pattern).is_file():
            count += 1
    return count


def download_transcoder_snapshot(
    config: TranscoderLoadConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    patterns = transcoder_allow_patterns(config)
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "config": transcoder_config_to_json(config),
        "repo_id": config.repo_id,
        "revision": config.revision,
        "cache_dir": config.transcoder_cache_dir,
        "allow_patterns": patterns,
        "pattern_count": len(patterns),
    }
    if dry_run:
        return summary

    from huggingface_hub import snapshot_download

    local_dir = Path(
        snapshot_download(
            config.repo_id,
            revision=config.revision,
            cache_dir=config.transcoder_cache_dir,
            allow_patterns=patterns,
        )
    )
    summary.update(
        {
            "local_dir": str(local_dir),
            "matched_file_count": _count_matching_files(local_dir, patterns),
        }
    )
    return summary
