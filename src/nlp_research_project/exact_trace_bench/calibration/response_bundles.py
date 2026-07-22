"""Project orchestration for publishing and validating response bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .response_bundle_adapter import PublicSiblingResponseModelApi, ResponseModelApi


def publish_response_bundle(
    observation_paths: Iterable[Path],
    *,
    output: Path,
    api: ResponseModelApi | None = None,
) -> dict[str, Any]:
    observations = tuple(Path(path) for path in observation_paths)
    if not observations:
        raise ValueError("at least one explicit observation path is required")
    missing = [str(path) for path in observations if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing observations: " + ", ".join(missing))
    adapter = api or PublicSiblingResponseModelApi()
    publish_result = adapter.publish(observations=observations, output=output)
    validation_result = adapter.validate(bundle=output)
    return {
        "bundle": str(output),
        "observation_count": len(observations),
        "publish": publish_result,
        "validation": validation_result,
    }


def validate_response_bundle(
    bundle: Path, *, api: ResponseModelApi | None = None
) -> Any:
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    return (api or PublicSiblingResponseModelApi()).validate(bundle=bundle)
