"""Explicit-root, idempotent Wave A/B observation backfill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..calibration_observations import (
    build_calibration_observation,
    write_calibration_observation,
)
from ..io_utils import read_json


_HISTORICAL_HELDOUTS = {
    ("wave_b_independent_semantic_physical", "google/gemma-3-4b-it", "physical_envelope_b256"),
    ("wave_b_independent_semantic_physical", "google/gemma-3-12b-it", "physical_envelope_b128"),
}


def _historical_reference_key(scenario: Mapping[str, Any]) -> str | None:
    baseline = scenario.get("baseline_check")
    if isinstance(baseline, Mapping):
        value = baseline.get("registry_key")
        if isinstance(value, str) and value:
            return value
    model = str(scenario.get("model_name") or scenario.get("model_id") or "")
    architecture = str(scenario.get("transcoder_architecture") or "")
    sizes = {"google/gemma-3-1b-it": "1b", "google/gemma-3-4b-it": "4b", "google/gemma-3-12b-it": "12b"}
    size = sizes.get(model)
    if size and architecture in {"clt", "plt"}:
        return f"governor_calibration/gemma3_{size}_{architecture}/361_base"
    return None


def _historical_campaign_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade immutable Wave A/B scenario metadata without rewriting it."""

    normalized = dict(scenario)
    stage = str(normalized.get("calibration_stage") or "")
    case = str(normalized.get("calibration_case") or "")
    model = str(normalized.get("model_name") or normalized.get("model_id") or "")
    reference_key = _historical_reference_key(normalized)
    if not isinstance(reference_key, str) or not reference_key:
        return normalized

    heldout = (stage, model, case) in _HISTORICAL_HELDOUTS
    role = "heldout" if heldout else ("reference" if case.startswith("reference") else "fit")
    normalized["calibration_campaign"] = {
        "campaign_id": f"historical_{stage}",
        "split": "heldout" if heldout else "fit",
        "role": role,
        "fit_group": stage,
        "holdout_group": "wave_b_physical_envelope" if heldout else None,
        "fixed_controls": {
            key: normalized[key]
            for key in (
                "fixture_name",
                "exact_trace_internal_dtype",
                "attribution_batch_size",
                "feature_batch_size",
                "logit_batch_size",
                "attribution_update_interval",
            )
            if key in normalized
        },
        "reference": {"kind": "baseline_registry", "registry_key": reference_key},
    }
    if normalized.get("governor_fidelity_mode") == "strict":
        normalized["governor_fidelity_mode"] = "exact"
    return normalized


def _candidate_roots(explicit_root: Path) -> tuple[Path, ...]:
    """Return only the explicit root or its immediate children; never search broadly."""

    if (explicit_root / "scenario.json").is_file():
        return (explicit_root,)
    if not explicit_root.is_dir():
        return ()
    return tuple(
        child
        for child in sorted(explicit_root.iterdir())
        if child.is_dir() and (child / "scenario.json").is_file()
    )


def backfill_observations(
    roots: Iterable[Path],
    *,
    baseline_entries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    written: list[str] = []
    unchanged: list[str] = []
    unsupported: list[dict[str, str]] = []
    seen: set[Path] = set()
    entries = baseline_entries or {}

    for explicit in roots:
        candidates = _candidate_roots(explicit)
        if not candidates:
            unsupported.append(
                {"root": str(explicit), "reason": "no_direct_scenario_rows"}
            )
            continue
        for root in candidates:
            resolved = root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            result_path = root / "result.json"
            if not result_path.is_file():
                unsupported.append({"root": str(root), "reason": "missing_result_json"})
                continue
            scenario = read_json(root / "scenario.json")
            result = read_json(result_path)
            if result.get("status") == "probe_completed":
                unsupported.append(
                    {"root": str(root), "reason": "diagnostic_probe"}
                )
                continue
            stage = str(scenario.get("calibration_stage") or "")
            if not stage.startswith(("wave_a_", "wave_b_")):
                unsupported.append(
                    {"root": str(root), "reason": "not_historical_wave_a_or_b"}
                )
                continue
            scenario = _historical_campaign_scenario(scenario)
            campaign = scenario.get("calibration_campaign")
            if not isinstance(campaign, Mapping):
                unsupported.append(
                    {"root": str(root), "reason": "not_calibration_campaign"}
                )
                continue
            key = (campaign.get("reference") or {}).get("registry_key")
            payload = build_calibration_observation(
                scenario_root=root,
                scenario=scenario,
                result=result,
                baseline_entry=entries.get(str(key)),
            )
            if payload is None:
                unsupported.append(
                    {"root": str(root), "reason": "unsupported_observation"}
                )
                continue
            output = root / "calibration_observation.json"
            if output.is_file() and read_json(output) == payload:
                unchanged.append(str(output))
            else:
                write_calibration_observation(root, payload)
                written.append(str(output))
    return {"written": written, "unchanged": unchanged, "unsupported": unsupported}
