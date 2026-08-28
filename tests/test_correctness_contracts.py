from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from nlp_research_project.exact_trace_bench.correctness import (
    BehavioralFaithfulnessReport,
    BehavioralFaithfulnessStatus,
    BehavioralSufficiency,
    CorrectnessReport,
    NumericalScopeReport,
    NumericalStabilityReport,
    NumericalStabilityStatus,
    StructuralCheck,
    StructuralCheckStatus,
    StructuralConformanceReport,
    StructuralVerdict,
    load_correctness_report,
    save_correctness_report,
)


def _report() -> CorrectnessReport:
    return CorrectnessReport(
        structural=StructuralConformanceReport(
            verdict=StructuralVerdict.INVALID,
            checks=(
                StructuralCheck(
                    check_id="strict_typed_graph_reopen",
                    status=StructuralCheckStatus.VIOLATED,
                    reason_code="typed_graph_reopen_failed",
                ),
            ),
            evidence=None,
        ),
        numerical_stability=NumericalStabilityReport(
            scopes=(
                NumericalScopeReport(
                    scope_id="typed_graph",
                    status=NumericalStabilityStatus.UNKNOWN,
                    metrics={"repeat_count": 0},
                    reason_codes=("structural_invalid",),
                ),
            )
        ),
        behavioral_faithfulness=BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.NOT_APPLICABLE,
            reason_codes=("structural_invalid",),
        ),
    )


def test_axis_status_values_are_explicit_and_independent() -> None:
    assert {status.value for status in NumericalStabilityStatus} == {
        "exact_stable",
        "alias_stable",
        "bounded",
        "review",
        "divergent",
        "unknown",
    }
    assert {status.value for status in BehavioralFaithfulnessStatus} == {
        "supported",
        "contradicted",
        "inconclusive",
        "unknown",
        "not_applicable",
    }

    payload = _report().to_json()
    assert payload["structural"]["verdict"] == "invalid"
    assert payload["numerical_stability"]["scopes"][0]["status"] == "unknown"
    assert payload["behavioral_faithfulness"] == {
        "status": "not_applicable",
        "sufficiency": "unknown",
        "metrics": {},
        "reason_codes": ["structural_invalid"],
        "evidence_fingerprint": None,
    }
    assert "correct" not in payload
    assert "passed" not in payload


def test_structural_verdict_must_agree_with_typed_checks() -> None:
    with pytest.raises(ValueError, match="does not agree"):
        StructuralConformanceReport(
            verdict=StructuralVerdict.CONFORMANT,
            checks=(
                StructuralCheck(
                    check_id="binding",
                    status=StructuralCheckStatus.VIOLATED,
                    reason_code="mismatch",
                ),
            ),
            evidence=None,
        )


def test_behavioral_sufficiency_is_fixed_to_unknown_in_v1() -> None:
    with pytest.raises(ValueError, match="sufficiency must remain unknown"):
        BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.SUPPORTED,
            sufficiency=cast(BehavioralSufficiency, "supported"),
        )


def test_report_persistence_is_deterministic_and_strict(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    report = _report()

    save_correctness_report(report, first)
    save_correctness_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    assert load_correctness_report(first) == report
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["numerical_stability"]["scopes"][0]["metrics"]["repeat_count"] = 1
    first.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_correctness_report(first)


def test_report_loader_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = _report().to_json()
    payload["aggregate_verdict"] = "correct"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown=\['aggregate_verdict'\]"):
        load_correctness_report(path)
