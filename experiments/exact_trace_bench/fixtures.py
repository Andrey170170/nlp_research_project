from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_FIXTURE_CATALOG, REPO_ROOT
from .io_utils import read_json


BASE_FIXTURES: tuple[str, ...] = ("828_base", "361_base")
ANOMALY_FIXTURES: tuple[str, ...] = ("94_base",)
LATE_FIXTURES: tuple[str, ...] = ("828_late", "361_late", "94_late")

PROMPT_TIERS: dict[str, tuple[str, ...]] = {
    "base": BASE_FIXTURES,
    "late": LATE_FIXTURES,
    "anomaly": ANOMALY_FIXTURES,
    "long_eval": LATE_FIXTURES,
}


@dataclass(frozen=True)
class FixtureRef:
    fixture_name: str
    fixture_kind: str
    gsm8k_index: int
    prepared_prompt_file: str
    prepared_prompt_meta_file: str

    def to_source_payload(self) -> dict[str, Any]:
        return {
            "fixture_name": self.fixture_name,
            "fixture_kind": self.fixture_kind,
            "prepared_prompt_file": self.prepared_prompt_file,
            "prepared_prompt_meta_file": self.prepared_prompt_meta_file,
            "gsm8k_indices": [self.gsm8k_index],
        }


def load_fixture_catalog(
    path: Path = DEFAULT_FIXTURE_CATALOG,
) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {fixture["fixture_name"]: fixture for fixture in payload.get("fixtures", [])}


def _fallback_fixture(fixture_name: str) -> FixtureRef:
    index_str, fixture_kind = fixture_name.split("_", maxsplit=1)
    fixture_dir = (
        REPO_ROOT / "experiments" / "generated" / "weekend_exact_chunked_fixtures"
    )
    fixture_subdir = fixture_dir / fixture_name
    return FixtureRef(
        fixture_name=fixture_name,
        fixture_kind="late_prefix" if fixture_kind == "late" else "base",
        gsm8k_index=int(index_str),
        prepared_prompt_file=str(fixture_subdir / "prompt.txt"),
        prepared_prompt_meta_file=str(fixture_subdir / "fixture_meta.json"),
    )


def resolve_fixture(
    fixture_name: str,
    *,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    allow_fallback: bool = True,
) -> FixtureRef:
    catalog_by_name = catalog_by_name or {}
    fixture = catalog_by_name.get(fixture_name)
    if fixture is None:
        if not allow_fallback:
            raise KeyError(f"Fixture not found in catalog: {fixture_name}")
        return _fallback_fixture(fixture_name)

    return FixtureRef(
        fixture_name=fixture_name,
        fixture_kind=str(fixture.get("fixture_kind") or "base"),
        gsm8k_index=int(fixture["gsm8k_index"]),
        prepared_prompt_file=str(fixture["prepared_prompt_file"]),
        prepared_prompt_meta_file=str(fixture["prepared_prompt_meta_file"]),
    )


def resolve_tier_fixtures(
    tier: str,
    *,
    catalog_by_name: dict[str, dict[str, Any]] | None = None,
    allow_fallback: bool = True,
) -> list[FixtureRef]:
    if tier not in PROMPT_TIERS:
        available = ", ".join(sorted(PROMPT_TIERS))
        raise ValueError(f"Unknown tier '{tier}'. Expected one of: {available}")
    return [
        resolve_fixture(
            fixture_name,
            catalog_by_name=catalog_by_name,
            allow_fallback=allow_fallback,
        )
        for fixture_name in PROMPT_TIERS[tier]
    ]


def describe_fixture_tiers() -> dict[str, list[dict[str, Any]]]:
    return {
        tier_name: [
            asdict(_fallback_fixture(fixture_name)) for fixture_name in fixtures
        ]
        for tier_name, fixtures in PROMPT_TIERS.items()
    }
