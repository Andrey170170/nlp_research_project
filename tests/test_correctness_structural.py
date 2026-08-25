from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.correctness import (
    ExpectedGraphIdentity,
    StructuralCheckStatus,
    StructuralVerdict,
    evaluate_structural_conformance,
)
from nlp_research_project.exact_trace_bench.correctness.contracts import (
    CompletedTraceEvidence,
    EnvelopeCleanupEvidence,
    FiniteValueEvidence,
    FrontierRefreshSequenceEvidence,
    MechanismRequirementObservation,
    RowDenominatorEvidence,
    SignedAccountingEvidence,
    StructuralAssessmentScope,
    StructuralConformanceReport,
    TerminalSuccessEvidence,
    TraceScopeCleanupEvidence,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    load_typed_compact_graph,
)
from typed_graph_fixtures import write_typed_graph


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, np.ndarray] = {
            name: np.asarray(data[name]) for name in data.files
        }
    payload.update(updates)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _complete_trace_evidence(*, edge_weights_checked: int) -> CompletedTraceEvidence:
    return CompletedTraceEvidence(
        terminal_success=TerminalSuccessEvidence(
            terminal_state="completed",
            successful=True,
        ),
        trace_scope_cleanup=TraceScopeCleanupEvidence(
            complete=True,
            resources_released=3,
        ),
        finite_values=FiniteValueEvidence(
            complete=True,
            tensor_values_checked=120,
            edge_weights_checked=edge_weights_checked,
            nonfinite_value_count=0,
        ),
        signed_accounting=SignedAccountingEvidence(
            complete=True,
            buckets_checked=6,
            signed_weight_buckets=6,
        ),
        row_denominator=RowDenominatorEvidence(
            complete=True,
            policy_id="stable_row_l1_v1",
            rows_checked=12,
            violation_count=0,
        ),
        frontier_refresh_sequence=FrontierRefreshSequenceEvidence(
            complete=True,
            events_checked=7,
            monotonic=True,
            duplicate_event_count=0,
        ),
        mechanism_requirements=(
            MechanismRequirementObservation(
                requirement_id="target_lane_chunk_size",
                expected="64",
                observed="64",
                satisfied=True,
            ),
            MechanismRequirementObservation(
                requirement_id="phase4_execution_mode",
                expected="cpu_exact",
                observed="cpu_exact",
                satisfied=True,
            ),
        ),
    )


def test_strict_reopen_reports_conformant_with_portable_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)
    graph = load_typed_compact_graph(path)

    report = evaluate_structural_conformance(
        path,
        expected=ExpectedGraphIdentity(
            step_idx=0,
            graph_fingerprint=graph.graph_fingerprint,
            target_fingerprint=graph.target_fingerprint,
            provider_fingerprint=graph.provider_fingerprint,
            trace_fingerprint=graph.trace_fingerprint,
            step_fingerprint=graph.step_fingerprint,
        ),
    )

    assert report.verdict is StructuralVerdict.CONFORMANT
    assert report.scope is StructuralAssessmentScope.ARTIFACT_ONLY
    assert all(
        check.status is StructuralCheckStatus.SATISFIED for check in report.checks
    )
    assert report.evidence is not None
    assert report.evidence.artifact_sha256.startswith("sha256:")
    assert report.evidence.graph_fingerprint == graph.graph_fingerprint
    assert report.evidence.step_idx == 0
    assert report.to_json()["scope"] == "artifact_only"


def test_completed_trace_requires_all_trace_local_evidence(tmp_path: Path) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)

    report = evaluate_structural_conformance(
        path,
        trace_evidence=CompletedTraceEvidence(),
    )

    assert report.scope is StructuralAssessmentScope.COMPLETED_TRACE
    assert report.verdict is StructuralVerdict.INVALID
    failed = {
        check.check_id: check.reason_code
        for check in report.checks
        if check.status is StructuralCheckStatus.VIOLATED
    }
    assert failed == {
        "terminal_success": "missing_terminal_success_evidence",
        "trace_scope_cleanup": "missing_trace_scope_cleanup_evidence",
        "finite_values": "missing_finite_value_evidence",
        "signed_accounting": "missing_signed_accounting_evidence",
        "row_denominator": "missing_row_denominator_evidence",
        "frontier_refresh_sequence": "missing_frontier_refresh_sequence_evidence",
        "mechanism_requirements_declared": ("missing_mechanism_requirement_evidence"),
    }


def test_complete_trace_evidence_is_conformant_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)
    graph = load_typed_compact_graph(path)
    envelope = EnvelopeCleanupEvidence(
        complete=False,
        envelope_id="window-session-7",
        detail="outer window remains open for the next token",
    )

    report = evaluate_structural_conformance(
        path,
        trace_evidence=_complete_trace_evidence(edge_weights_checked=graph.edge_count),
        envelope_cleanup=envelope,
    )

    assert report.scope is StructuralAssessmentScope.COMPLETED_TRACE
    assert report.verdict is StructuralVerdict.CONFORMANT
    assert all(
        check.status is StructuralCheckStatus.SATISFIED for check in report.checks
    )
    assert report.envelope_cleanup is envelope
    assert report.to_json()["envelope_cleanup"]["scope"] == (
        "outer_window_session_envelope"
    )
    assert StructuralConformanceReport.from_json(report.to_json()) == report


def test_failed_trace_evidence_and_named_mechanism_invalidate(tmp_path: Path) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)
    graph = load_typed_compact_graph(path)
    complete = _complete_trace_evidence(edge_weights_checked=graph.edge_count)
    trace_evidence = CompletedTraceEvidence(
        terminal_success=complete.terminal_success,
        trace_scope_cleanup=complete.trace_scope_cleanup,
        finite_values=FiniteValueEvidence(
            complete=True,
            tensor_values_checked=120,
            edge_weights_checked=graph.edge_count,
            nonfinite_value_count=1,
        ),
        signed_accounting=complete.signed_accounting,
        row_denominator=complete.row_denominator,
        frontier_refresh_sequence=complete.frontier_refresh_sequence,
        mechanism_requirements=(
            MechanismRequirementObservation(
                requirement_id="phase4_execution_mode",
                expected="cpu_exact",
                observed="cuda_windowed",
                satisfied=False,
            ),
        ),
    )

    report = evaluate_structural_conformance(path, trace_evidence=trace_evidence)

    assert report.verdict is StructuralVerdict.INVALID
    failed = {
        check.check_id: check.reason_code
        for check in report.checks
        if check.status is StructuralCheckStatus.VIOLATED
    }
    assert failed == {
        "finite_values": "finite_value_check_failed",
        "mechanism_requirement:phase4_execution_mode": (
            "mechanism_requirement_not_satisfied"
        ),
    }


def test_envelope_cleanup_cannot_masquerade_as_trace_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)

    with pytest.raises(
        ValueError, match="cannot be attached to an artifact-only assessment"
    ):
        evaluate_structural_conformance(
            path,
            envelope_cleanup=EnvelopeCleanupEvidence(
                complete=True,
                envelope_id="window-session-7",
            ),
        )


def test_canonical_loader_failure_becomes_typed_invalid_report(tmp_path: Path) -> None:
    path = tmp_path / "step_000.npz"
    write_typed_graph(path)
    _rewrite_npz(path, unexpected_field=np.asarray(1))

    report = evaluate_structural_conformance(path)

    assert report.verdict is StructuralVerdict.INVALID
    assert report.evidence is None
    assert len(report.checks) == 1
    assert report.checks[0].check_id == "strict_typed_graph_reopen"
    assert report.checks[0].status is StructuralCheckStatus.VIOLATED
    assert report.checks[0].reason_code == "typed_graph_reopen_failed"
    assert "fields do not match schema v2" in (report.checks[0].detail or "")


def test_external_identity_mismatch_invalidates_without_discarding_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.npz"
    write_typed_graph(path)

    report = evaluate_structural_conformance(
        path,
        expected=ExpectedGraphIdentity(
            target_fingerprint="sha256:" + "0" * 64,
        ),
    )

    assert report.verdict is StructuralVerdict.INVALID
    assert report.evidence is not None
    assert report.checks[0].status is StructuralCheckStatus.SATISFIED
    assert report.checks[1].status is StructuralCheckStatus.VIOLATED
    assert report.checks[1].reason_code == "evidence_identity_mismatch"
    assert "target_fingerprint" in (report.checks[1].detail or "")


def test_step_path_mismatch_is_owned_by_canonical_loader(tmp_path: Path) -> None:
    path = tmp_path / "step_003.npz"
    write_typed_graph(path, step_idx=2)

    report = evaluate_structural_conformance(path)

    assert report.verdict is StructuralVerdict.INVALID
    assert report.evidence is None
    assert "does not match expected step 3" in (report.checks[0].detail or "")
