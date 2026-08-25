"""Structural conformance through the canonical typed-graph seam."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    TypedCompactGraph,
    load_typed_compact_graph,
)

from .contracts import (
    CompletedTraceEvidence,
    EnvelopeCleanupEvidence,
    ExpectedGraphIdentity,
    FiniteValueEvidence,
    FrontierRefreshSequenceEvidence,
    GraphEvidenceIdentity,
    MechanismRequirementObservation,
    RowDenominatorEvidence,
    SignedAccountingEvidence,
    StructuralAssessmentScope,
    StructuralCheck,
    StructuralCheckStatus,
    StructuralConformanceReport,
    StructuralVerdict,
    TerminalSuccessEvidence,
    TraceScopeCleanupEvidence,
)


def evaluate_structural_conformance(
    graph_path: Path,
    *,
    expected: ExpectedGraphIdentity | None = None,
    trace_evidence: CompletedTraceEvidence | None = None,
    envelope_cleanup: EnvelopeCleanupEvidence | None = None,
) -> StructuralConformanceReport:
    """Strictly reopen one canonical graph and bind it to optional trace pins.

    Artifact validation remains owned by ``load_typed_compact_graph``.  This
    layer translates that canonical verdict into the correctness report and
    adds only cross-artifact identity binding.
    """

    if envelope_cleanup is not None and trace_evidence is None:
        raise ValueError(
            "outer envelope cleanup cannot be attached to an artifact-only assessment"
        )
    scope = (
        StructuralAssessmentScope.ARTIFACT_ONLY
        if trace_evidence is None
        else StructuralAssessmentScope.COMPLETED_TRACE
    )
    path = Path(graph_path)
    expectation = expected or ExpectedGraphIdentity()
    try:
        graph = load_typed_compact_graph(
            path,
            expected_step_idx=expectation.step_idx,
        )
        artifact_sha256 = _file_sha256(path)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        trace_checks = () if trace_evidence is None else _trace_checks(trace_evidence)
        return StructuralConformanceReport(
            verdict=StructuralVerdict.INVALID,
            scope=scope,
            checks=(
                StructuralCheck(
                    check_id="strict_typed_graph_reopen",
                    status=StructuralCheckStatus.VIOLATED,
                    reason_code="typed_graph_reopen_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
                *trace_checks,
            ),
            evidence=None,
            trace_evidence=trace_evidence,
            envelope_cleanup=envelope_cleanup,
        )

    evidence = _evidence_identity(graph, artifact_sha256=artifact_sha256)
    trace_checks = (
        ()
        if trace_evidence is None
        else _trace_checks(
            trace_evidence,
            expected_edge_weight_count=graph.edge_count,
            expected_bucket_count=len(graph.bucket_names),
        )
    )
    checks = [
        StructuralCheck(
            check_id="strict_typed_graph_reopen",
            status=StructuralCheckStatus.SATISFIED,
        )
    ]
    if expectation.has_pins:
        mismatches = _identity_mismatches(evidence, expectation)
        if mismatches:
            checks.append(
                StructuralCheck(
                    check_id="expected_evidence_identity",
                    status=StructuralCheckStatus.VIOLATED,
                    reason_code="evidence_identity_mismatch",
                    detail="; ".join(mismatches),
                )
            )
        else:
            checks.append(
                StructuralCheck(
                    check_id="expected_evidence_identity",
                    status=StructuralCheckStatus.SATISFIED,
                )
            )

    checks.extend(trace_checks)

    verdict = (
        StructuralVerdict.INVALID
        if any(check.status is StructuralCheckStatus.VIOLATED for check in checks)
        else StructuralVerdict.CONFORMANT
    )
    return StructuralConformanceReport(
        verdict=verdict,
        checks=tuple(checks),
        evidence=evidence,
        scope=scope,
        trace_evidence=trace_evidence,
        envelope_cleanup=envelope_cleanup,
    )


def _trace_checks(
    evidence: CompletedTraceEvidence,
    *,
    expected_edge_weight_count: int | None = None,
    expected_bucket_count: int | None = None,
) -> tuple[StructuralCheck, ...]:
    checks = [
        _terminal_check(evidence.terminal_success),
        _cleanup_check(evidence.trace_scope_cleanup),
        _finite_values_check(
            evidence.finite_values,
            expected_edge_weight_count=expected_edge_weight_count,
        ),
        _signed_accounting_check(
            evidence.signed_accounting,
            expected_bucket_count=expected_bucket_count,
        ),
        _row_denominator_check(evidence.row_denominator),
        _frontier_sequence_check(evidence.frontier_refresh_sequence),
    ]
    requirements = evidence.mechanism_requirements
    if requirements is None:
        checks.append(
            _missing_check(
                "mechanism_requirements_declared",
                "missing_mechanism_requirement_evidence",
            )
        )
    else:
        checks.append(
            StructuralCheck(
                check_id="mechanism_requirements_declared",
                status=StructuralCheckStatus.SATISFIED,
            )
        )
        for requirement in requirements:
            checks.append(
                StructuralCheck(
                    check_id=f"mechanism_requirement:{requirement.requirement_id}",
                    status=(
                        StructuralCheckStatus.SATISFIED
                        if requirement.satisfied
                        else StructuralCheckStatus.VIOLATED
                    ),
                    reason_code=(
                        None
                        if requirement.satisfied
                        else "mechanism_requirement_not_satisfied"
                    ),
                    detail=(
                        None
                        if requirement.satisfied
                        else _mechanism_detail(requirement)
                    ),
                )
            )
    return tuple(checks)


def _terminal_check(
    evidence: TerminalSuccessEvidence | None,
) -> StructuralCheck:
    if evidence is None:
        return _missing_check("terminal_success", "missing_terminal_success_evidence")
    return StructuralCheck(
        check_id="terminal_success",
        status=(
            StructuralCheckStatus.SATISFIED
            if evidence.successful
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=None if evidence.successful else "trace_terminal_not_successful",
        detail=(
            None
            if evidence.successful
            else f"terminal_state={evidence.terminal_state!r}; {evidence.detail or ''}".rstrip(
                "; "
            )
        ),
    )


def _cleanup_check(evidence: TraceScopeCleanupEvidence | None) -> StructuralCheck:
    if evidence is None:
        return _missing_check(
            "trace_scope_cleanup", "missing_trace_scope_cleanup_evidence"
        )
    return StructuralCheck(
        check_id="trace_scope_cleanup",
        status=(
            StructuralCheckStatus.SATISFIED
            if evidence.complete
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=None if evidence.complete else "trace_scope_cleanup_incomplete",
        detail=None if evidence.complete else evidence.detail,
    )


def _finite_values_check(
    evidence: FiniteValueEvidence | None,
    *,
    expected_edge_weight_count: int | None,
) -> StructuralCheck:
    if evidence is None:
        return _missing_check("finite_values", "missing_finite_value_evidence")
    coverage_matches = (
        expected_edge_weight_count is None
        or evidence.edge_weights_checked == expected_edge_weight_count
    )
    satisfied = evidence.satisfied and coverage_matches
    return StructuralCheck(
        check_id="finite_values",
        status=(
            StructuralCheckStatus.SATISFIED
            if satisfied
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=None if satisfied else "finite_value_check_failed",
        detail=(
            None
            if satisfied
            else "complete="
            f"{evidence.complete}; nonfinite_value_count={evidence.nonfinite_value_count}; "
            f"edge_weights_checked={evidence.edge_weights_checked}; "
            f"expected_edge_weight_count={expected_edge_weight_count}"
        ),
    )


def _signed_accounting_check(
    evidence: SignedAccountingEvidence | None,
    *,
    expected_bucket_count: int | None,
) -> StructuralCheck:
    if evidence is None:
        return _missing_check("signed_accounting", "missing_signed_accounting_evidence")
    coverage_matches = (
        expected_bucket_count is None
        or evidence.buckets_checked == expected_bucket_count
    )
    satisfied = evidence.satisfied and coverage_matches
    return StructuralCheck(
        check_id="signed_accounting",
        status=(
            StructuralCheckStatus.SATISFIED
            if satisfied
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=None if satisfied else "signed_accounting_check_failed",
        detail=(
            None
            if satisfied
            else "complete="
            f"{evidence.complete}; signed_weight_buckets="
            f"{evidence.signed_weight_buckets}/{evidence.buckets_checked}; "
            f"expected_bucket_count={expected_bucket_count}"
        ),
    )


def _row_denominator_check(
    evidence: RowDenominatorEvidence | None,
) -> StructuralCheck:
    if evidence is None:
        return _missing_check("row_denominator", "missing_row_denominator_evidence")
    return StructuralCheck(
        check_id="row_denominator",
        status=(
            StructuralCheckStatus.SATISFIED
            if evidence.satisfied
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=None if evidence.satisfied else "row_denominator_check_failed",
        detail=(
            None
            if evidence.satisfied
            else "complete="
            f"{evidence.complete}; policy_id={evidence.policy_id!r}; "
            f"violation_count={evidence.violation_count}"
        ),
    )


def _frontier_sequence_check(
    evidence: FrontierRefreshSequenceEvidence | None,
) -> StructuralCheck:
    if evidence is None:
        return _missing_check(
            "frontier_refresh_sequence",
            "missing_frontier_refresh_sequence_evidence",
        )
    return StructuralCheck(
        check_id="frontier_refresh_sequence",
        status=(
            StructuralCheckStatus.SATISFIED
            if evidence.satisfied
            else StructuralCheckStatus.VIOLATED
        ),
        reason_code=(
            None if evidence.satisfied else "frontier_refresh_sequence_check_failed"
        ),
        detail=(
            None
            if evidence.satisfied
            else "complete="
            f"{evidence.complete}; monotonic={evidence.monotonic}; "
            f"duplicate_event_count={evidence.duplicate_event_count}"
        ),
    )


def _missing_check(check_id: str, reason_code: str) -> StructuralCheck:
    return StructuralCheck(
        check_id=check_id,
        status=StructuralCheckStatus.VIOLATED,
        reason_code=reason_code,
    )


def _mechanism_detail(requirement: MechanismRequirementObservation) -> str:
    value = f"expected={requirement.expected!r}; observed={requirement.observed!r}"
    return value if requirement.detail is None else f"{value}; {requirement.detail}"


def _evidence_identity(
    graph: TypedCompactGraph, *, artifact_sha256: str
) -> GraphEvidenceIdentity:
    return GraphEvidenceIdentity(
        artifact_sha256=artifact_sha256,
        graph_fingerprint=graph.graph_fingerprint,
        target_fingerprint=graph.target_fingerprint,
        provider_fingerprint=graph.provider_fingerprint,
        trace_fingerprint=graph.trace_fingerprint,
        step_fingerprint=graph.step_fingerprint,
        schema_version=graph.schema_version,
        compact_save_format=graph.compact_save_format,
        step_idx=graph.step_idx,
        retention_policy_id=graph.retention_policy_id,
        retention_policy_fingerprint=graph.retention_policy_fingerprint,
    )


def _identity_mismatches(
    evidence: GraphEvidenceIdentity, expected: ExpectedGraphIdentity
) -> list[str]:
    mismatches: list[str] = []
    for field_name in (
        "step_idx",
        "graph_fingerprint",
        "target_fingerprint",
        "provider_fingerprint",
        "trace_fingerprint",
        "step_fingerprint",
    ):
        expected_value = getattr(expected, field_name)
        if expected_value is None:
            continue
        observed_value = getattr(evidence, field_name)
        if observed_value != expected_value:
            mismatches.append(
                f"{field_name}: expected {expected_value!r}, observed {observed_value!r}"
            )
    return mismatches


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
