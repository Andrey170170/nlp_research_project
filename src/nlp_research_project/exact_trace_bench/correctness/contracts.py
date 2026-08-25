"""Typed contracts for the exact-trace correctness gate.

The three axes deliberately remain independent.  This module does not expose
an aggregate correctness verdict or promotion decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Self


CORRECTNESS_REPORT_SCHEMA_VERSION = 1
CORRECTNESS_REPORT_FORMAT = "exact_trace_correctness_report_v1"

MetricValue = bool | int | float | str | None


class StructuralVerdict(str, Enum):
    CONFORMANT = "conformant"
    INVALID = "invalid"


class StructuralAssessmentScope(str, Enum):
    ARTIFACT_ONLY = "artifact_only"
    COMPLETED_TRACE = "completed_trace"


class StructuralCheckStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"


class NumericalStabilityStatus(str, Enum):
    EXACT_STABLE = "exact_stable"
    ALIAS_STABLE = "alias_stable"
    BOUNDED = "bounded"
    DIVERGENT = "divergent"
    UNKNOWN = "unknown"


class BehavioralFaithfulnessStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class BehavioralSufficiency(str, Enum):
    """Step 1b does not make a sufficiency claim."""

    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GraphEvidenceIdentity:
    """Portable identity of the strictly reopened graph evidence."""

    artifact_sha256: str
    graph_fingerprint: str
    target_fingerprint: str
    provider_fingerprint: str
    trace_fingerprint: str
    step_fingerprint: str
    schema_version: int
    compact_save_format: str
    step_idx: int
    retention_policy_id: str
    retention_policy_fingerprint: str

    def __post_init__(self) -> None:
        _require_fingerprint(self.artifact_sha256, field_name="artifact_sha256")
        for name in (
            "graph_fingerprint",
            "target_fingerprint",
            "provider_fingerprint",
            "trace_fingerprint",
            "step_fingerprint",
            "retention_policy_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), field_name=name)
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if self.step_idx < 0:
            raise ValueError("step_idx must be non-negative")
        if not self.compact_save_format:
            raise ValueError("compact_save_format is required")
        if not self.retention_policy_id:
            raise ValueError("retention_policy_id is required")

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "graph_fingerprint": self.graph_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "provider_fingerprint": self.provider_fingerprint,
            "trace_fingerprint": self.trace_fingerprint,
            "step_fingerprint": self.step_fingerprint,
            "schema_version": self.schema_version,
            "compact_save_format": self.compact_save_format,
            "step_idx": self.step_idx,
            "retention_policy_id": self.retention_policy_id,
            "retention_policy_fingerprint": self.retention_policy_fingerprint,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "artifact_sha256",
            "graph_fingerprint",
            "target_fingerprint",
            "provider_fingerprint",
            "trace_fingerprint",
            "step_fingerprint",
            "schema_version",
            "compact_save_format",
            "step_idx",
            "retention_policy_id",
            "retention_policy_fingerprint",
        )
        _require_exact_fields(value, fields, label="structural.evidence")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class ExpectedGraphIdentity:
    """Optional external pins used to bind a graph to its requested trace."""

    step_idx: int | None = None
    graph_fingerprint: str | None = None
    target_fingerprint: str | None = None
    provider_fingerprint: str | None = None
    trace_fingerprint: str | None = None
    step_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.step_idx is not None and self.step_idx < 0:
            raise ValueError("step_idx must be non-negative")
        for name in (
            "graph_fingerprint",
            "target_fingerprint",
            "provider_fingerprint",
            "trace_fingerprint",
            "step_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_fingerprint(value, field_name=name)

    @property
    def has_pins(self) -> bool:
        return any(value is not None for value in self.__dict__.values())


@dataclass(frozen=True)
class TerminalSuccessEvidence:
    terminal_state: str
    successful: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.terminal_state:
            raise ValueError("terminal_state is required")
        if self.detail == "":
            raise ValueError("detail cannot be empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "terminal_state": self.terminal_state,
            "successful": self.successful,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("terminal_state", "successful", "detail")
        _require_exact_fields(value, fields, label="trace_evidence.terminal_success")
        return cls(
            terminal_state=_require_str(
                value["terminal_state"], label="terminal_state"
            ),
            successful=_require_bool(value["successful"], label="successful"),
            detail=_optional_str(value["detail"], label="detail"),
        )


@dataclass(frozen=True)
class TraceScopeCleanupEvidence:
    """Cleanup owned by one trace, excluding any outer reusable session."""

    complete: bool
    resources_released: int
    detail: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            self.resources_released, field_name="resources_released"
        )
        if self.detail == "":
            raise ValueError("detail cannot be empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": "trace",
            "complete": self.complete,
            "resources_released": self.resources_released,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("scope", "complete", "resources_released", "detail")
        _require_exact_fields(value, fields, label="trace_evidence.trace_scope_cleanup")
        if value["scope"] != "trace":
            raise ValueError("trace-scope cleanup evidence must have scope='trace'")
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            resources_released=_require_nonnegative_int(
                value["resources_released"], field_name="resources_released"
            ),
            detail=_optional_str(value["detail"], label="detail"),
        )


@dataclass(frozen=True)
class EnvelopeCleanupEvidence:
    """Cleanup of an outer reusable window/session, never a per-token claim."""

    complete: bool
    envelope_id: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.envelope_id:
            raise ValueError("envelope_id is required")
        if self.detail == "":
            raise ValueError("detail cannot be empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": "outer_window_session_envelope",
            "complete": self.complete,
            "envelope_id": self.envelope_id,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("scope", "complete", "envelope_id", "detail")
        _require_exact_fields(value, fields, label="envelope_evidence.cleanup")
        if value["scope"] != "outer_window_session_envelope":
            raise ValueError(
                "envelope cleanup evidence must have "
                "scope='outer_window_session_envelope'"
            )
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            envelope_id=_require_str(value["envelope_id"], label="envelope_id"),
            detail=_optional_str(value["detail"], label="detail"),
        )


@dataclass(frozen=True)
class FiniteValueEvidence:
    complete: bool
    tensor_values_checked: int
    edge_weights_checked: int
    nonfinite_value_count: int

    def __post_init__(self) -> None:
        for name in (
            "tensor_values_checked",
            "edge_weights_checked",
            "nonfinite_value_count",
        ):
            _require_nonnegative_int(getattr(self, name), field_name=name)

    @property
    def satisfied(self) -> bool:
        return self.complete and self.nonfinite_value_count == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "tensor_values_checked": self.tensor_values_checked,
            "edge_weights_checked": self.edge_weights_checked,
            "nonfinite_value_count": self.nonfinite_value_count,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "complete",
            "tensor_values_checked",
            "edge_weights_checked",
            "nonfinite_value_count",
        )
        _require_exact_fields(value, fields, label="trace_evidence.finite_values")
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            tensor_values_checked=_require_nonnegative_int(
                value["tensor_values_checked"], field_name="tensor_values_checked"
            ),
            edge_weights_checked=_require_nonnegative_int(
                value["edge_weights_checked"], field_name="edge_weights_checked"
            ),
            nonfinite_value_count=_require_nonnegative_int(
                value["nonfinite_value_count"], field_name="nonfinite_value_count"
            ),
        )


@dataclass(frozen=True)
class SignedAccountingEvidence:
    complete: bool
    buckets_checked: int
    signed_weight_buckets: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.buckets_checked, field_name="buckets_checked")
        _require_nonnegative_int(
            self.signed_weight_buckets, field_name="signed_weight_buckets"
        )
        if self.buckets_checked == 0:
            raise ValueError("signed accounting must cover at least one bucket")
        if self.signed_weight_buckets > self.buckets_checked:
            raise ValueError("signed_weight_buckets cannot exceed buckets_checked")

    @property
    def satisfied(self) -> bool:
        return self.complete and self.signed_weight_buckets == self.buckets_checked

    def to_json(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "buckets_checked": self.buckets_checked,
            "signed_weight_buckets": self.signed_weight_buckets,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("complete", "buckets_checked", "signed_weight_buckets")
        _require_exact_fields(value, fields, label="trace_evidence.signed_accounting")
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            buckets_checked=_require_nonnegative_int(
                value["buckets_checked"], field_name="buckets_checked"
            ),
            signed_weight_buckets=_require_nonnegative_int(
                value["signed_weight_buckets"], field_name="signed_weight_buckets"
            ),
        )


@dataclass(frozen=True)
class RowDenominatorEvidence:
    complete: bool
    policy_id: str
    rows_checked: int
    violation_count: int

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("row-denominator policy_id is required")
        _require_nonnegative_int(self.rows_checked, field_name="rows_checked")
        _require_nonnegative_int(self.violation_count, field_name="violation_count")

    @property
    def satisfied(self) -> bool:
        return self.complete and self.violation_count == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "policy_id": self.policy_id,
            "rows_checked": self.rows_checked,
            "violation_count": self.violation_count,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("complete", "policy_id", "rows_checked", "violation_count")
        _require_exact_fields(value, fields, label="trace_evidence.row_denominator")
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            policy_id=_require_str(value["policy_id"], label="policy_id"),
            rows_checked=_require_nonnegative_int(
                value["rows_checked"], field_name="rows_checked"
            ),
            violation_count=_require_nonnegative_int(
                value["violation_count"], field_name="violation_count"
            ),
        )


@dataclass(frozen=True)
class FrontierRefreshSequenceEvidence:
    complete: bool
    events_checked: int
    monotonic: bool
    duplicate_event_count: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.events_checked, field_name="events_checked")
        _require_nonnegative_int(
            self.duplicate_event_count, field_name="duplicate_event_count"
        )

    @property
    def satisfied(self) -> bool:
        return self.complete and self.monotonic and self.duplicate_event_count == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "events_checked": self.events_checked,
            "monotonic": self.monotonic,
            "duplicate_event_count": self.duplicate_event_count,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "complete",
            "events_checked",
            "monotonic",
            "duplicate_event_count",
        )
        _require_exact_fields(
            value, fields, label="trace_evidence.frontier_refresh_sequence"
        )
        return cls(
            complete=_require_bool(value["complete"], label="complete"),
            events_checked=_require_nonnegative_int(
                value["events_checked"], field_name="events_checked"
            ),
            monotonic=_require_bool(value["monotonic"], label="monotonic"),
            duplicate_event_count=_require_nonnegative_int(
                value["duplicate_event_count"], field_name="duplicate_event_count"
            ),
        )


@dataclass(frozen=True)
class MechanismRequirementObservation:
    requirement_id: str
    expected: str
    observed: str | None
    satisfied: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.requirement_id:
            raise ValueError("mechanism requirement_id is required")
        if not self.expected:
            raise ValueError("mechanism expected value is required")
        if self.observed == "":
            raise ValueError("mechanism observed value cannot be empty")
        if self.detail == "":
            raise ValueError("detail cannot be empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "expected": self.expected,
            "observed": self.observed,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("requirement_id", "expected", "observed", "satisfied", "detail")
        _require_exact_fields(
            value, fields, label="trace_evidence.mechanism_requirement"
        )
        return cls(
            requirement_id=_require_str(
                value["requirement_id"], label="requirement_id"
            ),
            expected=_require_str(value["expected"], label="expected"),
            observed=_optional_str(value["observed"], label="observed"),
            satisfied=_require_bool(value["satisfied"], label="satisfied"),
            detail=_optional_str(value["detail"], label="detail"),
        )


@dataclass(frozen=True)
class CompletedTraceEvidence:
    """Trace-local evidence; missing fields remain representable and fail closed."""

    terminal_success: TerminalSuccessEvidence | None = None
    trace_scope_cleanup: TraceScopeCleanupEvidence | None = None
    finite_values: FiniteValueEvidence | None = None
    signed_accounting: SignedAccountingEvidence | None = None
    row_denominator: RowDenominatorEvidence | None = None
    frontier_refresh_sequence: FrontierRefreshSequenceEvidence | None = None
    mechanism_requirements: tuple[MechanismRequirementObservation, ...] | None = None

    def __post_init__(self) -> None:
        if self.mechanism_requirements is not None:
            requirements = tuple(self.mechanism_requirements)
            object.__setattr__(self, "mechanism_requirements", requirements)
            names = tuple(item.requirement_id for item in requirements)
            if len(set(names)) != len(names):
                raise ValueError("mechanism requirement_id values must be unique")

    @property
    def satisfied(self) -> bool:
        scalar_evidence = (
            self.terminal_success,
            self.trace_scope_cleanup,
            self.finite_values,
            self.signed_accounting,
            self.row_denominator,
            self.frontier_refresh_sequence,
        )
        if any(item is None for item in scalar_evidence):
            return False
        if self.mechanism_requirements is None:
            return False
        assert self.terminal_success is not None
        assert self.trace_scope_cleanup is not None
        assert self.finite_values is not None
        assert self.signed_accounting is not None
        assert self.row_denominator is not None
        assert self.frontier_refresh_sequence is not None
        return (
            self.terminal_success.successful
            and self.trace_scope_cleanup.complete
            and self.finite_values.satisfied
            and self.signed_accounting.satisfied
            and self.row_denominator.satisfied
            and self.frontier_refresh_sequence.satisfied
            and all(item.satisfied for item in self.mechanism_requirements)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "terminal_success": _optional_evidence_json(self.terminal_success),
            "trace_scope_cleanup": _optional_evidence_json(self.trace_scope_cleanup),
            "finite_values": _optional_evidence_json(self.finite_values),
            "signed_accounting": _optional_evidence_json(self.signed_accounting),
            "row_denominator": _optional_evidence_json(self.row_denominator),
            "frontier_refresh_sequence": _optional_evidence_json(
                self.frontier_refresh_sequence
            ),
            "mechanism_requirements": (
                None
                if self.mechanism_requirements is None
                else [item.to_json() for item in self.mechanism_requirements]
            ),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "terminal_success",
            "trace_scope_cleanup",
            "finite_values",
            "signed_accounting",
            "row_denominator",
            "frontier_refresh_sequence",
            "mechanism_requirements",
        )
        _require_exact_fields(value, fields, label="trace_evidence")
        requirements = value["mechanism_requirements"]
        return cls(
            terminal_success=_optional_evidence(
                value["terminal_success"],
                TerminalSuccessEvidence,
                label="trace_evidence.terminal_success",
            ),
            trace_scope_cleanup=_optional_evidence(
                value["trace_scope_cleanup"],
                TraceScopeCleanupEvidence,
                label="trace_evidence.trace_scope_cleanup",
            ),
            finite_values=_optional_evidence(
                value["finite_values"],
                FiniteValueEvidence,
                label="trace_evidence.finite_values",
            ),
            signed_accounting=_optional_evidence(
                value["signed_accounting"],
                SignedAccountingEvidence,
                label="trace_evidence.signed_accounting",
            ),
            row_denominator=_optional_evidence(
                value["row_denominator"],
                RowDenominatorEvidence,
                label="trace_evidence.row_denominator",
            ),
            frontier_refresh_sequence=_optional_evidence(
                value["frontier_refresh_sequence"],
                FrontierRefreshSequenceEvidence,
                label="trace_evidence.frontier_refresh_sequence",
            ),
            mechanism_requirements=(
                None
                if requirements is None
                else tuple(
                    MechanismRequirementObservation.from_json(
                        _require_mapping(item, label="mechanism_requirement")
                    )
                    for item in _require_sequence(
                        requirements, label="mechanism_requirements"
                    )
                )
            ),
        )


@dataclass(frozen=True)
class StructuralCheck:
    check_id: str
    status: StructuralCheckStatus
    reason_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.check_id:
            raise ValueError("structural check_id is required")
        if self.status is StructuralCheckStatus.VIOLATED and not self.reason_code:
            raise ValueError("a violated structural check requires a reason_code")
        if (
            self.status is StructuralCheckStatus.SATISFIED
            and self.reason_code is not None
        ):
            raise ValueError("a satisfied structural check cannot have a reason_code")
        if self.reason_code == "":
            raise ValueError("reason_code cannot be empty")
        if self.detail == "":
            raise ValueError("detail cannot be empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("check_id", "status", "reason_code", "detail")
        _require_exact_fields(value, fields, label="structural.check")
        return cls(
            check_id=_require_str(value["check_id"], label="check_id"),
            status=StructuralCheckStatus(value["status"]),
            reason_code=_optional_str(value["reason_code"], label="reason_code"),
            detail=_optional_str(value["detail"], label="detail"),
        )


@dataclass(frozen=True)
class StructuralConformanceReport:
    verdict: StructuralVerdict
    checks: tuple[StructuralCheck, ...]
    evidence: GraphEvidenceIdentity | None
    scope: StructuralAssessmentScope = StructuralAssessmentScope.ARTIFACT_ONLY
    trace_evidence: CompletedTraceEvidence | None = None
    envelope_cleanup: EnvelopeCleanupEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.checks:
            raise ValueError("structural report requires at least one check")
        check_ids = tuple(check.check_id for check in self.checks)
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("structural check_id values must be unique")
        violated = any(
            check.status is StructuralCheckStatus.VIOLATED for check in self.checks
        )
        expected = (
            StructuralVerdict.INVALID if violated else StructuralVerdict.CONFORMANT
        )
        if self.verdict is not expected:
            raise ValueError("structural verdict does not agree with typed checks")
        if self.verdict is StructuralVerdict.CONFORMANT and self.evidence is None:
            raise ValueError("conformant structural report requires evidence identity")
        if self.scope is StructuralAssessmentScope.ARTIFACT_ONLY:
            if self.trace_evidence is not None:
                raise ValueError(
                    "artifact-only report cannot contain completed-trace evidence"
                )
            if self.envelope_cleanup is not None:
                raise ValueError(
                    "artifact-only report cannot contain envelope cleanup evidence"
                )
        elif self.trace_evidence is None:
            raise ValueError("completed-trace report requires trace_evidence")
        elif (
            self.verdict is StructuralVerdict.CONFORMANT
            and not self.trace_evidence.satisfied
        ):
            raise ValueError(
                "conformant completed-trace report requires complete, satisfied "
                "trace evidence"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "verdict": self.verdict.value,
            "checks": [check.to_json() for check in self.checks],
            "evidence": None if self.evidence is None else self.evidence.to_json(),
            "trace_evidence": (
                None if self.trace_evidence is None else self.trace_evidence.to_json()
            ),
            "envelope_cleanup": (
                None
                if self.envelope_cleanup is None
                else self.envelope_cleanup.to_json()
            ),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "scope",
            "verdict",
            "checks",
            "evidence",
            "trace_evidence",
            "envelope_cleanup",
        )
        _require_exact_fields(value, fields, label="structural")
        checks = _require_sequence(value["checks"], label="structural.checks")
        evidence = value["evidence"]
        trace_evidence = value["trace_evidence"]
        envelope_cleanup = value["envelope_cleanup"]
        return cls(
            scope=StructuralAssessmentScope(value["scope"]),
            verdict=StructuralVerdict(value["verdict"]),
            checks=tuple(
                StructuralCheck.from_json(
                    _require_mapping(item, label="structural.check")
                )
                for item in checks
            ),
            evidence=(
                None
                if evidence is None
                else GraphEvidenceIdentity.from_json(
                    _require_mapping(evidence, label="structural.evidence")
                )
            ),
            trace_evidence=(
                None
                if trace_evidence is None
                else CompletedTraceEvidence.from_json(
                    _require_mapping(trace_evidence, label="trace_evidence")
                )
            ),
            envelope_cleanup=(
                None
                if envelope_cleanup is None
                else EnvelopeCleanupEvidence.from_json(
                    _require_mapping(envelope_cleanup, label="envelope_cleanup")
                )
            ),
        )


@dataclass(frozen=True)
class NumericalScopeReport:
    scope_id: str
    status: NumericalStabilityStatus
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise ValueError("numerical scope_id is required")
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))

    def to_json(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = ("scope_id", "status", "metrics", "reason_codes")
        _require_exact_fields(value, fields, label="numerical_stability.scope")
        return cls(
            scope_id=_require_str(value["scope_id"], label="scope_id"),
            status=NumericalStabilityStatus(value["status"]),
            metrics=_require_mapping(value["metrics"], label="metrics"),
            reason_codes=tuple(
                _require_str(item, label="reason_code")
                for item in _require_sequence(
                    value["reason_codes"], label="reason_codes"
                )
            ),
        )


@dataclass(frozen=True)
class NumericalStabilityReport:
    scopes: tuple[NumericalScopeReport, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", tuple(self.scopes))
        if not self.scopes:
            raise ValueError("numerical stability report requires at least one scope")
        scope_ids = tuple(scope.scope_id for scope in self.scopes)
        if len(set(scope_ids)) != len(scope_ids):
            raise ValueError("numerical scope_id values must be unique")

    def to_json(self) -> dict[str, Any]:
        return {"scopes": [scope.to_json() for scope in self.scopes]}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        _require_exact_fields(value, ("scopes",), label="numerical_stability")
        return cls(
            scopes=tuple(
                NumericalScopeReport.from_json(
                    _require_mapping(item, label="numerical_stability.scope")
                )
                for item in _require_sequence(
                    value["scopes"], label="numerical_stability.scopes"
                )
            )
        )


@dataclass(frozen=True)
class BehavioralFaithfulnessReport:
    status: BehavioralFaithfulnessStatus
    sufficiency: BehavioralSufficiency = BehavioralSufficiency.UNKNOWN
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    evidence_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.sufficiency is not BehavioralSufficiency.UNKNOWN:
            raise ValueError("behavioral sufficiency must remain unknown in schema v1")
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.evidence_fingerprint is not None:
            _require_fingerprint(
                self.evidence_fingerprint, field_name="evidence_fingerprint"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sufficiency": self.sufficiency.value,
            "metrics": dict(self.metrics),
            "reason_codes": list(self.reason_codes),
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "status",
            "sufficiency",
            "metrics",
            "reason_codes",
            "evidence_fingerprint",
        )
        _require_exact_fields(value, fields, label="behavioral_faithfulness")
        return cls(
            status=BehavioralFaithfulnessStatus(value["status"]),
            sufficiency=BehavioralSufficiency(value["sufficiency"]),
            metrics=_require_mapping(value["metrics"], label="metrics"),
            reason_codes=tuple(
                _require_str(item, label="reason_code")
                for item in _require_sequence(
                    value["reason_codes"], label="reason_codes"
                )
            ),
            evidence_fingerprint=_optional_str(
                value["evidence_fingerprint"], label="evidence_fingerprint"
            ),
        )


@dataclass(frozen=True)
class CorrectnessReport:
    structural: StructuralConformanceReport
    numerical_stability: NumericalStabilityReport
    behavioral_faithfulness: BehavioralFaithfulnessReport
    schema_version: int = CORRECTNESS_REPORT_SCHEMA_VERSION
    report_format: str = CORRECTNESS_REPORT_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != CORRECTNESS_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported correctness report schema_version: {self.schema_version}"
            )
        if self.report_format != CORRECTNESS_REPORT_FORMAT:
            raise ValueError(
                f"unsupported correctness report format: {self.report_format!r}"
            )

    @property
    def report_fingerprint(self) -> str:
        return fingerprint_json(self._content_json())

    def _content_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_format": self.report_format,
            "structural": self.structural.to_json(),
            "numerical_stability": self.numerical_stability.to_json(),
            "behavioral_faithfulness": self.behavioral_faithfulness.to_json(),
        }

    def to_json(self) -> dict[str, Any]:
        payload = self._content_json()
        payload["report_fingerprint"] = self.report_fingerprint
        return payload

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "schema_version",
            "report_format",
            "structural",
            "numerical_stability",
            "behavioral_faithfulness",
            "report_fingerprint",
        )
        _require_exact_fields(value, fields, label="correctness report")
        report = cls(
            schema_version=value["schema_version"],
            report_format=value["report_format"],
            structural=StructuralConformanceReport.from_json(
                _require_mapping(value["structural"], label="structural")
            ),
            numerical_stability=NumericalStabilityReport.from_json(
                _require_mapping(
                    value["numerical_stability"], label="numerical_stability"
                )
            ),
            behavioral_faithfulness=BehavioralFaithfulnessReport.from_json(
                _require_mapping(
                    value["behavioral_faithfulness"],
                    label="behavioral_faithfulness",
                )
            ),
        )
        stored = _require_str(value["report_fingerprint"], label="report_fingerprint")
        _require_fingerprint(stored, field_name="report_fingerprint")
        if stored != report.report_fingerprint:
            raise ValueError("correctness report fingerprint mismatch")
        return report


def fingerprint_json(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _freeze_metrics(value: Mapping[str, MetricValue]) -> Mapping[str, MetricValue]:
    if not isinstance(value, Mapping):
        raise TypeError("metrics must be a mapping")
    normalized: dict[str, MetricValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric names must be non-empty strings")
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"metric {key!r} has unsupported value type")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"metric {key!r} must be finite")
        normalized[key] = item
    return MappingProxyType(normalized)


def _reason_codes(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("reason codes must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("reason codes must be unique")
    return result


def _require_fingerprint(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{field_name} must be a sha256 fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a sha256 fingerprint") from exc


def _require_exact_fields(
    value: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    expected = set(fields)
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match schema: missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be an array")
    return value


def _require_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _optional_str(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, label=label)


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _require_nonnegative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _optional_evidence_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value.to_json()
    if not isinstance(payload, dict):
        raise TypeError("evidence to_json() must return an object")
    return payload


def _optional_evidence(value: Any, evidence_type: type[Any], *, label: str) -> Any:
    if value is None:
        return None
    return evidence_type.from_json(_require_mapping(value, label=label))
