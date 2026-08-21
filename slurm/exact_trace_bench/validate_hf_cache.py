#!/usr/bin/env python3
"""Fail early when a selected model cannot resolve from the batch cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from nlp_research_project.exact_trace_bench.full_answer.prepared_workload import (
    validate_prepared_workload,
)


def _selected_configs_from_launch_spec(
    record: Mapping[str, Any], *, label: str
) -> Iterable[Mapping[str, Any]]:
    selections = record.get("selected_configs")
    if not isinstance(selections, list) or not selections:
        raise ValueError(f"launch specification lacks selected_configs: {label}")
    for index, selection in enumerate(selections):
        if not isinstance(selection, Mapping) or not isinstance(
            selection.get("selected_config"), Mapping
        ):
            raise ValueError(
                f"launch specification selected_configs[{index}] is invalid: {label}"
            )
        yield selection["selected_config"]


def _selected_configs_from_trace_specs(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read trace specs {path}: {exc}") from exc
    found = False
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid trace-spec JSON at {path}:{line_number}: {exc}"
            ) from exc
        config = record.get("graph_knobs") if isinstance(record, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError(f"trace spec lacks graph_knobs at {path}:{line_number}")
        found = True
        yield config
    if not found:
        raise ValueError(f"trace specs are empty: {path}")


def _validated_cache_root() -> Path:
    names = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
    raw_values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in raw_values.items() if not value]
    if missing:
        raise ValueError(
            "missing Hugging Face cache environment: " + ", ".join(missing)
        )
    paths = {name: Path(str(value)) for name, value in raw_values.items()}
    relative = [name for name, path in paths.items() if not path.is_absolute()]
    if relative:
        raise ValueError(
            "Hugging Face cache paths must be absolute: " + ", ".join(relative)
        )
    resolved = {name: path.resolve() for name, path in paths.items()}
    if len(set(resolved.values())) != 1:
        rendered = ", ".join(f"{name}={path}" for name, path in resolved.items())
        raise ValueError(f"Hugging Face cache paths disagree: {rendered}")
    root = next(iter(resolved.values()))
    if not root.is_dir():
        raise ValueError(f"Hugging Face cache root is not a directory: {root}")
    if not os.access(root, os.R_OK | os.X_OK):
        raise ValueError(f"Hugging Face cache root is not readable/searchable: {root}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("scientific cache preflight requires HF_HUB_OFFLINE=1")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("scientific cache preflight requires TRANSFORMERS_OFFLINE=1")
    return root


def validate_selected_models(
    configs: Iterable[Mapping[str, Any]],
    *,
    expected_model_root: Path | None = None,
) -> dict[str, Any]:
    cache_root = _validated_cache_root()
    identities: dict[tuple[str, str | None], Mapping[str, Any]] = {}
    for config in configs:
        model_name = config.get("model_name")
        revision = config.get("revision")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("selected config lacks model_name")
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"selected model revision must be a string: {model_name}")
        configured_cache = config.get("transcoder_cache_dir")
        if (
            configured_cache is not None
            and Path(str(configured_cache)).resolve() != cache_root
        ):
            raise ValueError(
                "selected transcoder_cache_dir disagrees with authoritative cache: "
                f"{configured_cache} != {cache_root}"
            )
        identities[(model_name, revision)] = config
    if not identities:
        raise ValueError("no selected model configurations found")

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    resolved_models = []
    for model_name, revision in sorted(identities):
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    cache_dir=str(cache_root),
                    local_files_only=True,
                )
            ).resolve()
            AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"selected model {model_name!r} revision {revision!r} cannot "
                f"resolve offline from {cache_root}: {type(exc).__name__}: {exc}"
            ) from exc
        resolved_models.append(
            {"model_name": model_name, "revision": revision, "snapshot": str(snapshot)}
        )

    if expected_model_root is not None:
        expected = expected_model_root.resolve()
        actual = {Path(item["snapshot"]) for item in resolved_models}
        if actual != {expected}:
            raise ValueError(
                f"preheat model root {expected} does not match resolved models "
                f"{sorted(str(path) for path in actual)}"
            )
    return {
        "schema_version": 1,
        "operation": "offline_huggingface_cache_preflight",
        "cache_root": str(cache_root),
        "offline": True,
        "resolved_models": resolved_models,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--prepared-workload", type=Path)
    sources.add_argument("--trace-specs", type=Path, action="append")
    parser.add_argument("--expected-model-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.prepared_workload is not None:
        workload = validate_prepared_workload(args.prepared_workload)
        configs = list(
            _selected_configs_from_launch_spec(
                workload.launch_spec, label=str(workload.launch_spec_path)
            )
        )
    else:
        configs = [
            config
            for path in args.trace_specs
            for config in _selected_configs_from_trace_specs(path)
        ]
    result = validate_selected_models(
        configs,
        expected_model_root=args.expected_model_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
