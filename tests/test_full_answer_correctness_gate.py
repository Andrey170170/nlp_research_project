import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from circuit_tracer.verification import FrozenBehavioralCalibration

from nlp_research_project.exact_trace_bench.correctness.behavioral import (
    BehavioralProbeMode,
    QualifiedAliasPair,
)
from nlp_research_project.exact_trace_bench.correctness.contracts import (
    BehavioralFaithfulnessReport,
    BehavioralFaithfulnessStatus,
    NumericalScopeReport,
    NumericalStabilityStatus,
)
from nlp_research_project.exact_trace_bench.correctness.frontier import FeatureKey
from nlp_research_project.exact_trace_bench.correctness.numerical import (
    build_reference_identity_receipt,
    file_sha256,
    prepare_numerical_manifest_declaration,
)
from nlp_research_project.exact_trace_bench.full_answer.correctness_gate import (
    _behavioral_alias_handoff,
    _behavioral_calibration_admitted,
    _behavioral_probe_policy,
    _closed_telemetry_evidence,
    _qualified_alias_pairs,
    _required_alias_policy_satisfied,
    _required_numerical_policy_satisfied,
    _row_denominator_evidence_from_compact_result,
    _run_declared_numerical_comparison,
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
    evaluation, artifact = result
    report = evaluation.report
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


def test_calibrated_churn_handoff_uses_typed_numerical_aliases() -> None:
    numerical_alias = QualifiedAliasPair(
        source=FeatureKey(layer=3, position=5, feature_id=7),
        substitute=FeatureKey(layer=3, position=5, feature_id=11),
        selection_policy_id="behavioral_closure_v1",
        calibration_fingerprint=_fingerprint("calibration"),
        comparison_evidence_fingerprint=_fingerprint("comparison"),
        baseline_graph_fingerprint=_fingerprint("baseline"),
        candidate_graph_fingerprint=_fingerprint("candidate"),
        qualified_decoder_cosine=0.97,
    )

    aliases = _behavioral_alias_handoff(
        mode=BehavioralProbeMode.REQUIRED,
        calibration=SimpleNamespace(),
        numerical_aliases=(numerical_alias,),
        compact_result={"correctness_qualified_alias_pairs": "must-not-be-read"},
    )

    assert aliases == (numerical_alias,)
    assert _behavioral_alias_handoff(
        mode=BehavioralProbeMode.REQUIRED,
        calibration=SimpleNamespace(),
        numerical_aliases=(numerical_alias,),
        compact_result={},
        required_alias=False,
    ) == ()


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
        (NumericalStabilityStatus.ALIAS_STABLE, {}, False),
        (
            NumericalStabilityStatus.ALIAS_STABLE,
            {
                "classification_policy_id": "behavioral_closure_v1",
                "calibration_fingerprint": "sha256:" + "a" * 64,
            },
            True,
        ),
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


def test_calibrated_numerical_admission_ignores_optional_unknown_scope() -> None:
    fingerprint = "sha256:" + "a" * 64
    required = (
        NumericalScopeReport(
            scope_id="typed_graph.repeat",
            status=NumericalStabilityStatus.BOUNDED,
            metrics={
                "classification_policy_id": "behavioral_closure_v1",
                "calibration_fingerprint": fingerprint,
            },
        ),
        NumericalScopeReport(
            scope_id="typed_graph.canonical",
            status=NumericalStabilityStatus.EXACT_STABLE,
        ),
        NumericalScopeReport(
            scope_id="feature_frontier.repeat",
            status=NumericalStabilityStatus.ALIAS_STABLE,
            metrics={
                "classification_policy_id": "behavioral_closure_v1",
                "calibration_fingerprint": fingerprint,
            },
        ),
    )
    optional = NumericalScopeReport(
        scope_id="feature_frontier.canonical",
        status=NumericalStabilityStatus.UNKNOWN,
        reason_codes=("declared_canonical_frontier_absent",),
    )

    assert _required_numerical_policy_satisfied(
        policy_id="behavioral_closure_v1",
        scopes=(*required, optional),
        required_scope_ids=tuple(scope.scope_id for scope in required),
        calibration_fingerprint=fingerprint,
    )
    assert not _required_numerical_policy_satisfied(
        policy_id="behavioral_closure_v1",
        scopes=(required[0], required[2], optional),
        required_scope_ids=tuple(scope.scope_id for scope in required),
        calibration_fingerprint=fingerprint,
    )
    mismatched_policy = NumericalScopeReport(
        scope_id="typed_graph.repeat",
        status=NumericalStabilityStatus.BOUNDED,
        metrics={
            "classification_policy_id": "different_policy_v1",
            "calibration_fingerprint": fingerprint,
        },
    )
    assert not _required_numerical_policy_satisfied(
        policy_id="behavioral_closure_v1",
        scopes=(mismatched_policy, required[1], required[2]),
        required_scope_ids=tuple(scope.scope_id for scope in required),
        calibration_fingerprint=fingerprint,
    )


def test_required_alias_policy_requires_bound_pair_and_complete_comparator() -> None:
    fingerprint = _fingerprint("calibration")
    alias_scope = NumericalScopeReport(
        scope_id="feature_frontier.repeat",
        status=NumericalStabilityStatus.ALIAS_STABLE,
        metrics={
            "classification_policy_id": "behavioral_closure_v1",
            "calibration_fingerprint": fingerprint,
        },
    )
    alias = QualifiedAliasPair(
        source=FeatureKey(layer=3, position=5, feature_id=7),
        substitute=FeatureKey(layer=3, position=5, feature_id=11),
        selection_policy_id="behavioral_closure_v1",
        calibration_fingerprint=fingerprint,
        comparison_evidence_fingerprint=_fingerprint("comparison"),
        baseline_graph_fingerprint=_fingerprint("baseline"),
        candidate_graph_fingerprint=_fingerprint("candidate"),
        qualified_decoder_cosine=0.97,
    )
    supported = BehavioralFaithfulnessReport(
        status=BehavioralFaithfulnessStatus.SUPPORTED,
        metrics={
            "policy_id": "behavioral_closure_v1",
            "calibration_id": "correctness_calibration_v1",
            "evidence_completeness": "complete",
            "runtime_status": "complete",
        },
    )
    kwargs = {
        "policy_id": "behavioral_closure_v1",
        "scopes": (alias_scope,),
        "required_scope_ids": (alias_scope.scope_id,),
        "calibration_id": "correctness_calibration_v1",
        "calibration_fingerprint": fingerprint,
        "behavioral": supported,
        "alias_comparator_status": "complete",
        "behavioral_alias_selection_count": 1,
    }

    assert not _required_alias_policy_satisfied(
        **kwargs,
        qualified_aliases=(),
    )
    assert _required_alias_policy_satisfied(
        **kwargs,
        qualified_aliases=(alias,),
    )
    assert not _required_alias_policy_satisfied(
        **{**kwargs, "alias_comparator_status": "not_applicable"},
        qualified_aliases=(alias,),
    )
    assert not _required_alias_policy_satisfied(
        **{**kwargs, "behavioral_alias_selection_count": 0},
        qualified_aliases=(alias,),
    )
    assert not _required_alias_policy_satisfied(
        **kwargs,
        qualified_aliases=(
            replace(alias, calibration_fingerprint=_fingerprint("other")),
        ),
    )
    assert not _required_alias_policy_satisfied(
        **kwargs,
        qualified_aliases=(
            replace(alias, selection_policy_id="different_policy_v1"),
        ),
    )
    assert not _required_alias_policy_satisfied(
        **{
            **kwargs,
            "behavioral": replace(
                supported,
                metrics={
                    **supported.metrics,
                    "calibration_id": "other_calibration_v1",
                },
            ),
        },
        qualified_aliases=(alias,),
    )


def test_required_alias_policy_ignores_optional_alias_scope() -> None:
    optional_alias = NumericalScopeReport(
        scope_id="feature_frontier.canonical",
        status=NumericalStabilityStatus.ALIAS_STABLE,
    )
    required_exact = NumericalScopeReport(
        scope_id="typed_graph.repeat",
        status=NumericalStabilityStatus.EXACT_STABLE,
    )

    assert _required_alias_policy_satisfied(
        policy_id="behavioral_closure_v1",
        scopes=(required_exact, optional_alias),
        required_scope_ids=(required_exact.scope_id,),
        calibration_id="correctness_calibration_v1",
        calibration_fingerprint=_fingerprint("calibration"),
        qualified_aliases=(),
        behavioral=BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.SUPPORTED,
            metrics={},
        ),
        alias_comparator_status="not_applicable",
        behavioral_alias_selection_count=0,
    )


def test_behavioral_probe_policy_receives_frozen_calibration_projection() -> None:
    sibling_calibration = FrozenBehavioralCalibration(
        calibration_id="correctness_calibration_v1",
        policy_id="behavioral_closure_v1",
        direct_max_mean_relative_closure=0.05,
    )
    calibration = SimpleNamespace(
        behavioral_probe_policy_kwargs=lambda: {
            "calibration": sibling_calibration,
            "no_op_absolute_tolerance": 1e-6,
            "no_op_relative_tolerance": 0.0,
        }
    )

    policy = _behavioral_probe_policy(
        policy_id="behavioral_closure_v1",
        calibration=calibration,
    )

    assert policy.calibration is sibling_calibration
    assert policy.no_op_absolute_tolerance == 1e-6
    assert policy.no_op_relative_tolerance == 0.0


def test_required_behavioral_preparation_fails_closed_without_calibration() -> None:
    assert _behavioral_calibration_admitted(
        mode=BehavioralProbeMode.SMOKE,
        calibration=None,
    )
    assert not _behavioral_calibration_admitted(
        mode=BehavioralProbeMode.REQUIRED,
        calibration=None,
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
