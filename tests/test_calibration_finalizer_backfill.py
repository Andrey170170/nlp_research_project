from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_research_project.exact_trace_bench.calibration.backfill import (
    backfill_observations,
)
from nlp_research_project.exact_trace_bench.calibration.finalizer import (
    finalize_observations,
    parse_sacct,
)


def _scenario() -> dict[str, object]:
    return {
        "calibration_case": "wave_a_row",
        "calibration_stage": "wave_a_upward_1b",
        "calibration_campaign": {
            "campaign_id": "historical-wave-a",
            "reference": {"kind": "baseline_registry", "registry_key": "provider/ref"},
        },
        "governor_fidelity_mode": "exact",
    }


def _write_root(root: Path, *, result: bool = True) -> None:
    root.mkdir(parents=True)
    (root / "scenario.json").write_text(json.dumps(_scenario()))
    if result:
        (root / "result.json").write_text(
            json.dumps({"status": "success", "duration_seconds": 2})
        )


def test_backfill_is_explicit_idempotent_and_reports_unsupported(
    tmp_path: Path,
) -> None:
    supported = tmp_path / "campaign" / "row"
    unsupported = tmp_path / "campaign" / "no-result"
    nested = tmp_path / "campaign" / "deep" / "ignored"
    _write_root(supported)
    _write_root(unsupported, result=False)
    _write_root(nested)

    first = backfill_observations([tmp_path / "campaign"])
    second = backfill_observations([tmp_path / "campaign"])

    assert len(first["written"]) == 1
    assert second["written"] == []
    assert len(second["unchanged"]) == 1
    assert {row["reason"] for row in first["unsupported"]} == {"missing_result_json"}
    assert not (nested / "calibration_observation.json").exists()


def test_backfill_upgrades_legacy_wave_b_and_reserves_heldout(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    scenario = {
        "calibration_case": "physical_envelope_b256",
        "calibration_stage": "wave_b_independent_semantic_physical",
        "model_name": "google/gemma-3-4b-it",
        "transcoder_architecture": "plt",
        "governor_fidelity_mode": "strict",
        "baseline_check": {"enabled": False, "mode": "off"},
    }
    (root / "scenario.json").write_text(json.dumps(scenario))
    (root / "result.json").write_text(
        json.dumps({"status": "success", "duration_seconds": 2})
    )

    summary = backfill_observations([root])
    observation = json.loads((root / "calibration_observation.json").read_text())

    assert len(summary["written"]) == 1
    assert observation["campaign"]["split"] == "heldout"
    assert observation["campaign"]["role"] == "heldout"
    assert observation["fidelity"]["mode"] == "exact"
    assert observation["campaign"]["reference"]["registry_key"] == (
        "governor_calibration/gemma3_4b_plt/361_base"
    )


def test_sacct_parser_and_finalizer_regenerate_after_runner_death(
    tmp_path: Path,
) -> None:
    root = tmp_path / "row"
    _write_root(root, result=False)
    sacct = (
        "123|TIMEOUT|0:0|7200|20G|30G|cpu=1,mem=400G|400G|start|end|node|TimeLimit\n"
    )

    summary = finalize_observations([root], job_id="123", sacct_runner=lambda _: sacct)
    observation = json.loads((root / "calibration_observation.json").read_text())

    assert summary["accounting"][0]["classification"] == "censored"
    assert (
        json.loads((root / "result.json").read_text())["result_origin"]
        == "scheduler_finalizer"
    )
    assert observation["schema_version"] == 2
    assert observation["provenance"]["scheduler_accounting"]["state"] == "TIMEOUT"
    assert (
        observation["provenance"]["finalization"]["regenerated_after_runner_loss"]
        is True
    )


def test_finalizer_rejects_nonterminal_accounting(tmp_path: Path) -> None:
    root = tmp_path / "row"
    _write_root(root)
    sacct = "123|RUNNING|0:0|1||||||||\n"
    with pytest.raises(ValueError, match="not terminal"):
        finalize_observations([root], job_id="123", sacct_runner=lambda _: sacct)


def test_parse_sacct_rejects_short_rows() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_sacct("123|FAILED\n")
