import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from nlp_research_project.exact_trace_bench.correctness.contracts import (
    NumericalScopeReport,
    NumericalStabilityStatus,
)
from nlp_research_project.exact_trace_bench.full_answer.correctness_gate import (
    _closed_telemetry_evidence,
    _qualified_alias_pairs,
    _required_numerical_policy_satisfied,
    _row_denominator_evidence_from_compact_result,
    _run_declared_numerical_comparison,
)
from nlp_research_project.exact_trace_bench.correctness.numerical import (
    build_reference_identity_receipt,
    file_sha256,
    prepare_numerical_manifest_declaration,
)


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def test_gate_persists_declared_numerical_comparison_sidecar(tmp_path: Path) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    reference = tmp_path / "reference.npz"
    write_typed_graph(candidate)
    write_typed_graph(reference)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "exact_trace_numerical_reference_manifest_v1",
                "references": [
                    {
                        "reference_id": "repeat-1",
                        "role": "repeat",
                        "graph_path": str(reference.resolve()),
                        "graph_sha256": file_sha256(reference),
                        "identity": build_reference_identity_receipt(reference),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    declaration = prepare_numerical_manifest_declaration(manifest)
    token_dir = tmp_path / "token"
    token_dir.mkdir()

    result = _run_declared_numerical_comparison(
        knobs={
            "correctness_numerical_manifest_path": declaration["manifest_path"],
            "correctness_numerical_manifest_sha256": declaration["manifest_sha256"],
        },
        graph_path=candidate,
        token_dir=token_dir,
    )

    assert result is not None
    report, artifact = result
    assert report.scopes[0].status is NumericalStabilityStatus.EXACT_STABLE
    assert report.scopes[1].reason_codes == ("declared_repeat_frontier_absent",)
    assert report.scopes[2].reason_codes == ("declared_canonical_reference_absent",)
    sidecar = Path(str(artifact["path"]))
    assert sidecar.name == "correctness_numerical_details.json"
    assert artifact["artifact_sha256"] == file_sha256(sidecar)


def test_full_answer_alias_seam_accepts_only_explicit_bounded_comparison_evidence() -> None:
    row = {
        "source": {"layer": 1, "position": 2, "feature_id": 3},
        "substitute": {"layer": 1, "position": 2, "feature_id": 4},
        "selection_policy_id": "frontier_alias_calibration_v1",
        "calibration_fingerprint": _fingerprint("calibration"),
        "comparison_evidence_fingerprint": _fingerprint("comparison"),
        "baseline_graph_fingerprint": _fingerprint("baseline"),
        "candidate_graph_fingerprint": _fingerprint("candidate"),
        "qualified_decoder_cosine": 0.95,
    }

    assert _qualified_alias_pairs({}) == ()
    aliases = _qualified_alias_pairs({"correctness_qualified_alias_pairs": [row]})

    assert len(aliases) == 1
    assert aliases[0].source.feature_id == 3
    assert aliases[0].comparison_evidence_fingerprint == _fingerprint("comparison")
    with pytest.raises(ValueError, match="bound of two"):
        _qualified_alias_pairs(
            {"correctness_qualified_alias_pairs": [row, row, row]}
        )


def _telemetry_spec() -> dict[str, object]:
    return {
        "trace_id": "trace-7",
        "trajectory_id": "trajectory-3",
        "generated_index": 7,
        "target_position": 11,
        "target_token_id": 42,
    }


def _telemetry_record(
    *, index: int, name: str, attrs: dict[str, object]
) -> dict[str, object]:
    return {
        **_telemetry_spec(),
        "event_index": index,
        "sequence": index + 1,
        "scope": "run" if name == "attribute.done" else "batch",
        "name": name,
        "attrs": attrs,
    }


def test_closed_telemetry_evidence_streams_identity_bound_full_sink(
    tmp_path: Path,
) -> None:
    sink_path = tmp_path / "telemetry_live.jsonl"
    records = [
        _telemetry_record(
            index=0,
            name="phase4.refresh",
            attrs={"refresh_index": 0},
        ),
        _telemetry_record(
            index=1,
            name="phase4.refresh",
            attrs={"refresh_index": 1},
        ),
        _telemetry_record(
            index=2,
            name="attribute.done",
            attrs={"status": "succeeded"},
        ),
    ]
    sink_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    errors: list[dict[str, str]] = []
    trace_result = SimpleNamespace(
        telemetry_summary={
            "sink_status": "closed",
            "sink_path": str(sink_path),
            "sink_event_count": 3,
            "event_count": 3,
            "sink_error_count": 0,
        },
        # The in-memory mirror may be capped before phase 4 and is not evidence.
        telemetry_events=({"name": "unrelated.capped.event"},),
    )

    terminal_done, frontier = _closed_telemetry_evidence(
        trace_result,
        spec=_telemetry_spec(),
        expected_path=sink_path,
        expected_refresh_count=2,
        errors=errors,
    )

    assert terminal_done is True
    assert frontier is not None
    assert frontier.complete is True
    assert frontier.events_checked == 2
    assert frontier.monotonic is True
    assert frontier.duplicate_event_count == 0
    assert errors == []


def test_closed_telemetry_evidence_does_not_fallback_to_capped_events(
    tmp_path: Path,
) -> None:
    errors: list[dict[str, str]] = []
    trace_result = SimpleNamespace(
        telemetry_summary={"sink_status": "disabled"},
        telemetry_events=(
            {"name": "attribute.done", "attrs": {"status": "succeeded"}},
        ),
    )

    terminal_done, frontier = _closed_telemetry_evidence(
        trace_result,
        spec=_telemetry_spec(),
        expected_path=tmp_path / "telemetry_live.jsonl",
        expected_refresh_count=None,
        errors=errors,
    )

    assert terminal_done is False
    assert frontier is None
    assert errors[0]["stage"] == "closed_telemetry_sink"


def test_closed_telemetry_evidence_requires_explicit_zero_sink_errors(
    tmp_path: Path,
) -> None:
    sink_path = tmp_path / "telemetry_live.jsonl"
    record = _telemetry_record(
        index=0,
        name="attribute.done",
        attrs={"status": "succeeded"},
    )
    sink_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors: list[dict[str, str]] = []

    terminal_done, frontier = _closed_telemetry_evidence(
        SimpleNamespace(
            telemetry_summary={
                "sink_status": "closed",
                "sink_path": str(sink_path),
                "sink_event_count": 1,
                "event_count": 1,
            }
        ),
        spec=_telemetry_spec(),
        expected_path=sink_path,
        expected_refresh_count=0,
        errors=errors,
    )

    assert terminal_done is False
    assert frontier is None
    assert "write errors" in errors[0]["detail"]


@pytest.mark.parametrize(
    ("refresh_indices", "expected_refresh_count"),
    [([1, 2], 2), ([0, 2], 2), ([0, 0], 2), ([0, 1], 3)],
)
def test_closed_telemetry_evidence_rejects_unbound_refresh_sequences(
    tmp_path: Path,
    refresh_indices: list[int],
    expected_refresh_count: int,
) -> None:
    sink_path = tmp_path / "telemetry_live.jsonl"
    records = [
        _telemetry_record(
            index=index,
            name="phase4.refresh",
            attrs={"refresh_index": refresh_index},
        )
        for index, refresh_index in enumerate(refresh_indices)
    ]
    records.append(
        _telemetry_record(
            index=len(records),
            name="attribute.done",
            attrs={"status": "succeeded"},
        )
    )
    sink_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    errors: list[dict[str, str]] = []

    terminal_done, frontier = _closed_telemetry_evidence(
        SimpleNamespace(
            telemetry_summary={
                "sink_status": "closed",
                "sink_path": str(sink_path),
                "sink_event_count": len(records),
                "event_count": len(records),
                "sink_error_count": 0,
            }
        ),
        spec=_telemetry_spec(),
        expected_path=sink_path,
        expected_refresh_count=expected_refresh_count,
        errors=errors,
    )

    assert terminal_done is True
    assert frontier is None
    assert errors[-1]["stage"] == "frontier_refresh_sequence"


def test_closed_telemetry_evidence_requires_explicit_refresh_count(
    tmp_path: Path,
) -> None:
    sink_path = tmp_path / "telemetry_live.jsonl"
    record = _telemetry_record(
        index=0,
        name="attribute.done",
        attrs={"status": "succeeded"},
    )
    sink_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    errors: list[dict[str, str]] = []

    terminal_done, frontier = _closed_telemetry_evidence(
        SimpleNamespace(
            telemetry_summary={
                "sink_status": "closed",
                "sink_path": str(sink_path),
                "sink_event_count": 1,
                "event_count": 1,
                "sink_error_count": 0,
            }
        ),
        spec=_telemetry_spec(),
        expected_path=sink_path,
        expected_refresh_count=None,
        errors=errors,
    )

    assert terminal_done is True
    assert frontier is None
    assert errors[-1]["stage"] == "frontier_refresh_sequence"


@pytest.mark.parametrize(
    ("status", "metrics", "satisfied"),
    [
        (NumericalStabilityStatus.UNKNOWN, {}, False),
        (NumericalStabilityStatus.DIVERGENT, {}, False),
        (NumericalStabilityStatus.EXACT_STABLE, {}, True),
        (NumericalStabilityStatus.ALIAS_STABLE, {}, True),
        (NumericalStabilityStatus.BOUNDED, {}, False),
        (
            NumericalStabilityStatus.BOUNDED,
            {
                "classification_policy_id": "frontier_alias_calibration_v1",
                "calibration_fingerprint": "sha256:" + "a" * 64,
            },
            True,
        ),
    ],
)
def test_required_numerical_policy_uses_versioned_admission(
    status: NumericalStabilityStatus,
    metrics: dict[str, int | float | str | None],
    satisfied: bool,
) -> None:
    scope = NumericalScopeReport(
        scope_id="typed_graph",
        status=status,
        metrics=metrics,
    )

    assert (
        _required_numerical_policy_satisfied(
            policy_id="behavioral_closure_v1",
            scopes=(scope,),
        )
        is satisfied
    )
    assert (
        _required_numerical_policy_satisfied(
            policy_id="unknown_policy_v9",
            scopes=(scope,),
        )
        is False
    )


def test_row_denominator_evidence_uses_only_explicit_compact_receipt() -> None:
    errors: list[dict[str, str]] = []

    evidence = _row_denominator_evidence_from_compact_result(
        {
            "correctness_row_denominator_evidence": {
                "complete": True,
                "policy_id": "canonical_scaled_l1_row_evidence_sha256_v2",
                "rows_checked": 17,
                "violation_count": 0,
                "authoritative_batch_count": 2,
            },
            "telemetry_summary": {
                "row_denominator_evidence": {
                    "complete": False,
                    "policy_id": "untrusted_telemetry_v1",
                    "rows_checked": 0,
                    "violation_count": 99,
                }
            },
        },
        errors=errors,
    )

    assert evidence is not None
    assert evidence.complete is True
    assert evidence.rows_checked == 17
    assert evidence.violation_count == 0
    assert errors == []


def test_row_denominator_evidence_missing_compact_receipt_remains_unknown() -> None:
    errors: list[dict[str, str]] = []

    evidence = _row_denominator_evidence_from_compact_result(
        {
            "telemetry_summary": {
                "row_denominator_evidence": {
                    "complete": True,
                    "policy_id": "canonical_scaled_l1_row_evidence_sha256_v2",
                    "rows_checked": 17,
                    "violation_count": 0,
                }
            }
        },
        errors=errors,
    )

    assert evidence is None
    assert errors == []


def test_row_denominator_evidence_rejects_unknown_policy_identity() -> None:
    errors: list[dict[str, str]] = []

    evidence = _row_denominator_evidence_from_compact_result(
        {
            "correctness_row_denominator_evidence": {
                "complete": True,
                "policy_id": "unknown_policy_v9",
                "rows_checked": 17,
                "violation_count": 0,
                "authoritative_batch_count": 2,
            }
        },
        errors=errors,
    )

    assert evidence is None
    assert errors[0]["stage"] == "correctness_row_denominator_evidence"
    assert "unsupported correctness row-denominator policy_id" in errors[0]["detail"]
