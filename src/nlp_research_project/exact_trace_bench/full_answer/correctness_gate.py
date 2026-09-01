"""Post-session correctness evidence production for full-answer traces."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from circuit_tracer.verification import (
    BehavioralProbePolicy,
    NNSightInterventionRuntime,
    OrderingAdmissionMode,
    verify_behavior,
)

from ..correctness.behavioral import (
    MAX_UNSELECTED_CONTROLS,
    BehavioralProbeMode,
    BehavioralRequestStatus,
    LiveBehavioralCandidate,
    LiveProviderDecoderSource,
    QualifiedAliasPair,
    map_sibling_behavioral_report,
    persist_sibling_behavioral_report,
    prepare_live_behavioral_request,
)
from ..correctness.calibration import (
    FrozenCorrectnessCalibration,
    calibration_declaration_from_graph_knobs,
    load_declared_correctness_calibration,
)
from ..correctness.contracts import (
    BehavioralFaithfulnessReport,
    BehavioralFaithfulnessStatus,
    CompletedTraceEvidence,
    CorrectnessReport,
    EnvelopeCleanupEvidence,
    ExpectedGraphIdentity,
    FiniteValueEvidence,
    FrontierRefreshSequenceEvidence,
    fingerprint_json,
    MechanismRequirementObservation,
    NumericalScopeReport,
    NumericalStabilityReport,
    NumericalStabilityStatus,
    RowDenominatorEvidence,
    SignedAccountingEvidence,
    StructuralVerdict,
    TerminalSuccessEvidence,
    TraceScopeCleanupEvidence,
)
from ..correctness.frontier import FeatureKey
from ..correctness.frontier_artifact import (
    build_bounded_frontier_artifact,
    save_bounded_frontier_artifact,
)
from ..correctness.numerical import (
    DeclaredNumericalEvaluation,
    declaration_from_graph_knobs,
    evaluate_declared_numerical_stability,
    file_sha256,
)
from ..correctness.ordering_admission import (
    OrderingAdmissionError,
    admitted_nnsight_ordering,
)
from ..correctness.persistence import save_correctness_report
from ..correctness.structural import evaluate_structural_conformance
from ..io_utils import write_json
from ..trace_runtime.artifacts import load_and_validate_capture_artifact
from ..typed_compact_graph import load_typed_compact_graph
from .execution_control import execution_resolution

_ROW_DENOMINATOR_POLICY_ID = "canonical_scaled_l1_row_evidence_sha256_v2"
_REQUIRED_NUMERICAL_STATUSES_BY_POLICY = {
    "behavioral_closure_v1": frozenset(
        {
            NumericalStabilityStatus.EXACT_STABLE,
            NumericalStabilityStatus.ALIAS_STABLE,
            NumericalStabilityStatus.BOUNDED,
        }
    ),
}


@dataclass(frozen=True)
class CorrectnessGateOutcome:
    trace_payload: dict[str, Any]
    required_policy_satisfied: bool


class _BehavioralPreparationDeadlineExhausted(RuntimeError):
    """Internal control flow after a fail-closed preparation budget refusal."""


def _behavioral_probe_policy(
    *,
    policy_id: str,
    calibration: FrozenCorrectnessCalibration | None,
) -> BehavioralProbePolicy:
    kwargs = (
        {}
        if calibration is None
        else calibration.behavioral_probe_policy_kwargs()
    )
    return BehavioralProbePolicy(policy_id=policy_id, **kwargs)


def _behavioral_calibration_admitted(
    *,
    mode: BehavioralProbeMode,
    calibration: FrozenCorrectnessCalibration | None,
) -> bool:
    return mode is not BehavioralProbeMode.REQUIRED or calibration is not None


def _behavioral_alias_handoff(
    *,
    mode: BehavioralProbeMode,
    calibration: FrozenCorrectnessCalibration | None,
    numerical_aliases: tuple[QualifiedAliasPair, ...],
    compact_result: Mapping[str, Any],
    required_alias: bool = True,
) -> tuple[QualifiedAliasPair, ...]:
    if calibration is not None:
        return numerical_aliases if required_alias else ()
    if mode is BehavioralProbeMode.SMOKE:
        return _qualified_alias_pairs(compact_result)
    return ()


def _verify_behavior_with_ordering_admission(
    *,
    request: Any,
    model: Any,
    mode: BehavioralProbeMode,
    graph_knobs: Mapping[str, Any],
    transcoder_metadata: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Run one behavioral request with required-only, model-local admission."""

    if mode is BehavioralProbeMode.SMOKE:
        return (
            verify_behavior(
                request,
                NNSightInterventionRuntime(
                    model,
                    ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
                ),
            ),
            None,
        )
    identity = getattr(request, "identity", None)
    execution_fingerprint = getattr(identity, "execution_fingerprint", None)
    if not isinstance(execution_fingerprint, str) or not execution_fingerprint:
        raise OrderingAdmissionError(
            "required behavioral request lacks execution_fingerprint"
        )
    with admitted_nnsight_ordering(
        model=model,
        graph_knobs=graph_knobs,
        transcoder_metadata=transcoder_metadata,
        execution_fingerprint=execution_fingerprint,
    ) as admission_evidence:
        sibling = verify_behavior(
            request,
            NNSightInterventionRuntime(
                model,
                ordering_admission_mode=OrderingAdmissionMode.QUALIFIED,
            ),
        )
    return sibling, admission_evidence


def run_full_answer_correctness_gate(
    *,
    token_dir: Path,
    graph_path: Path,
    compact_result: Mapping[str, Any],
    graph_identity: ExpectedGraphIdentity,
    trace_result: Any,
    spec: Mapping[str, Any],
    model: Any,
    transcoder_metadata: Mapping[str, Any],
    envelope_cleanup: EnvelopeCleanupEvidence | None,
) -> CorrectnessGateOutcome:
    """Persist independent correctness axes after the trace session is closed."""

    knobs = _mapping(spec.get("graph_knobs"), "graph_knobs")
    mode = BehavioralProbeMode(str(knobs["correctness_probe_mode"]))
    policy_id = str(knobs["correctness_policy_id"])
    if mode is BehavioralProbeMode.OFF:
        return CorrectnessGateOutcome(
            trace_payload={
                "mode": mode.value,
                "policy_id": policy_id,
                "status": "not_applicable",
                "required_policy_satisfied": False,
                "axes": {
                    "structural": "not_applicable",
                    "numerical": "not_applicable",
                    "behavioral": "not_applicable",
                },
                "artifacts": {},
                "errors": [],
            },
            required_policy_satisfied=False,
        )

    errors: list[dict[str, str]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    calibration: FrozenCorrectnessCalibration | None = None
    try:
        calibration_declaration = calibration_declaration_from_graph_knobs(knobs)
        if calibration_declaration is None:
            artifacts["calibration"] = {"status": "not_declared", "path": None}
        else:
            calibration = load_declared_correctness_calibration(
                calibration_declaration,
                graph_knobs=knobs,
            )
            artifacts["calibration"] = {
                "status": "validated",
                "path": str(calibration.manifest_path),
                "calibration_id": calibration.calibration_id,
                "calibration_fingerprint": calibration.calibration_fingerprint,
                "required_numerical_scopes": list(
                    calibration.required_numerical_scopes
                ),
            }
    except (OSError, TypeError, ValueError) as error:
        _record_error(errors, "correctness_calibration", error)
        artifacts["calibration"] = {"status": "unavailable", "path": None}
    reopened_graph = None
    try:
        reopened_graph = load_typed_compact_graph(
            graph_path,
            expected_step_idx=int(spec["generated_index"]),
        )
    except (EOFError, OSError, TypeError, ValueError) as error:
        _record_error(errors, "strict_typed_graph_reopen", error)
    trace_evidence = _completed_trace_evidence(
        trace_result,
        graph=reopened_graph,
        compact_result=compact_result,
        spec=spec,
        expected_telemetry_path=token_dir / "telemetry_live.jsonl",
        selected_config=knobs,
        errors=errors,
    )
    structural = evaluate_structural_conformance(
        graph_path,
        expected=graph_identity,
        trace_evidence=trace_evidence,
        envelope_cleanup=envelope_cleanup,
    )
    structural_path = token_dir / "correctness_structural.json"
    write_json(structural_path, structural.to_json())
    artifacts["structural"] = {"status": "persisted", "path": str(structural_path)}

    numerical = NumericalStabilityReport(
        scopes=(
            NumericalScopeReport(
                scope_id="typed_graph",
                status=NumericalStabilityStatus.UNKNOWN,
                metrics={"reference_count": 0, "repeat_count": 0},
                reason_codes=("reference_or_repeat_absent",),
            ),
        )
    )
    behavioral = BehavioralFaithfulnessReport(
        status=BehavioralFaithfulnessStatus.UNKNOWN,
        metrics={"policy_id": policy_id},
        reason_codes=("behavioral_probe_unavailable",),
    )
    alias_comparator_status: str | None = None
    behavioral_alias_selection_count = 0
    numerical_aliases: tuple[QualifiedAliasPair, ...] = ()

    frontier = None
    try:
        if reopened_graph is None:
            raise ValueError("strictly reopened typed graph is unavailable")
        descriptors = load_and_validate_capture_artifact(
            "feature_semantic_descriptors",
            token_dir / "feature_semantic_descriptors.npz",
        )
        decoder_fingerprint = _decoder_fingerprint(transcoder_metadata)
        frontier = build_bounded_frontier_artifact(
            graph=reopened_graph,
            compact_result=compact_result,
            feature_semantic_descriptors=descriptors,
            decoder_fingerprint=decoder_fingerprint,
            max_near_cutoff=MAX_UNSELECTED_CONTROLS,
        )
        frontier_path = token_dir / "correctness_frontier.json"
        save_bounded_frontier_artifact(frontier, frontier_path)
        artifacts["frontier"] = {
            "status": "persisted",
            "path": str(frontier_path),
            "content_fingerprint": frontier.content_fingerprint,
        }
    except Exception as error:
        _record_error(errors, "frontier", error)
        artifacts["frontier"] = {"status": "unavailable", "path": None}

    try:
        numerical_result = _run_declared_numerical_comparison(
            knobs=knobs,
            graph_path=graph_path,
            token_dir=token_dir,
            frontier_available=frontier is not None,
            calibration=calibration,
        )
        if numerical_result is not None:
            numerical_evaluation, numerical_artifact = numerical_result
            numerical = numerical_evaluation.report
            numerical_aliases = numerical_evaluation.qualified_alias_pairs
            artifacts["numerical_details"] = numerical_artifact
    except Exception as error:
        _record_error(errors, "declared_numerical_comparison", error)
        numerical = NumericalStabilityReport(
            scopes=(
                NumericalScopeReport(
                    scope_id="typed_graph",
                    status=NumericalStabilityStatus.UNKNOWN,
                    metrics={"reference_count": 0, "repeat_count": 0},
                    reason_codes=("declared_numerical_comparison_failed",),
                ),
            )
        )
        artifacts["numerical_details"] = {"status": "unavailable", "path": None}

    behavioral_calibration_admitted = _behavioral_calibration_admitted(
        mode=mode,
        calibration=calibration,
    )
    required_alias_scope_ids = _required_alias_scope_ids(
        scopes=numerical.scopes,
        required_scope_ids=(
            None if calibration is None else calibration.required_numerical_scopes
        ),
    )
    required_alias_handoff_admitted = not required_alias_scope_ids or (
        calibration is not None
        and _qualified_alias_handoff_satisfied(
            policy_id=policy_id,
            calibration_fingerprint=calibration.calibration_fingerprint,
            qualified_aliases=numerical_aliases,
        )
    )
    if (
        frontier is not None
        and (envelope_cleanup is None or envelope_cleanup.complete)
        and behavioral_calibration_admitted
        and required_alias_handoff_admitted
    ):
        try:
            graph = load_typed_compact_graph(
                graph_path,
                expected_step_idx=int(spec["generated_index"]),
            )
            qualified_aliases = _behavioral_alias_handoff(
                mode=mode,
                calibration=calibration,
                numerical_aliases=numerical_aliases,
                compact_result=compact_result,
                required_alias=bool(required_alias_scope_ids),
            )
            probe_policy = _behavioral_probe_policy(
                policy_id=policy_id,
                calibration=calibration,
            )
            candidate = LiveBehavioralCandidate(
                graph_path=graph_path,
                compact_result=compact_result,
                frontier=frontier.evidence,
                trace_id=str(spec["trace_id"]),
                trace_fingerprint=graph.trace_fingerprint,
                target_fingerprint=graph.target_fingerprint,
                step_fingerprint=graph.step_fingerprint,
                semantic_fingerprint=str(
                    getattr(trace_result, "semantic_fingerprint")
                ),
                execution_fingerprint=str(
                    getattr(trace_result, "execution_fingerprint")
                ),
                provider_fingerprint=graph.provider_fingerprint,
                decoder_fingerprint=frontier.evidence.decoder_fingerprint,
                prompt_token_ids=tuple(int(value) for value in graph.token_ids.tolist()),
                target_position=int(spec["target_position"]),
                target_token_id=int(spec["target_token_id"]),
                policy_mode=mode,
                policy=probe_policy,
                qualified_aliases=qualified_aliases,
                decoder_source=(
                    LiveProviderDecoderSource(
                        model,
                        decoder_fingerprint=frontier.evidence.decoder_fingerprint,
                    )
                    if qualified_aliases
                    else None
                ),
            )
            preparation_started = time.perf_counter()
            prepared = prepare_live_behavioral_request(candidate)
            preparation_seconds = time.perf_counter() - preparation_started
            runtime_budget_seconds = max(
                0.0,
                probe_policy.max_seconds - preparation_seconds,
            )
            budget_evidence = {
                "end_to_end_budget_seconds": probe_policy.max_seconds,
                "preparation_seconds": preparation_seconds,
                "runtime_budget_seconds": runtime_budget_seconds,
                "budget_contract": "cooperative_prepare_plus_runtime_v1",
            }
            behavioral = prepared.assessment
            if prepared.status is BehavioralRequestStatus.READY:
                assert prepared.request is not None
                if runtime_budget_seconds <= probe_policy.cleanup_reserve_seconds:
                    behavioral = BehavioralFaithfulnessReport(
                        status=BehavioralFaithfulnessStatus.UNKNOWN,
                        metrics={"policy_id": policy_id, **budget_evidence},
                        reason_codes=("behavioral_preparation_deadline_exhausted",),
                    )
                    artifacts["behavioral_sibling"] = {
                        "status": "preparation_deadline_exhausted",
                        "path": None,
                        **budget_evidence,
                    }
                    raise _BehavioralPreparationDeadlineExhausted
                bounded_request = replace(
                    prepared.request,
                    policy=replace(
                        prepared.request.policy,
                        max_seconds=runtime_budget_seconds,
                    ),
                )
                sibling, ordering_admission_evidence = (
                    _verify_behavior_with_ordering_admission(
                        request=bounded_request,
                        model=model,
                        mode=mode,
                        graph_knobs=knobs,
                        transcoder_metadata=transcoder_metadata,
                    )
                )
                artifacts["ordering_admission"] = (
                    {
                        "status": "not_applicable",
                        "reason": "candidate_smoke_uses_unqualified_lane",
                    }
                    if ordering_admission_evidence is None
                    else {
                        "status": "validated",
                        "evidence": ordering_admission_evidence,
                    }
                )
                ordering_admission_metrics = (
                    {
                        "ordering_admission_status": "not_applicable",
                        "ordering_admission_evidence_fingerprint": None,
                        "ordering_admission_behavioral_execution_fingerprint": None,
                    }
                    if ordering_admission_evidence is None
                    else {
                        "ordering_admission_status": "validated",
                        "ordering_admission_evidence_fingerprint": fingerprint_json(
                            ordering_admission_evidence
                        ),
                        "ordering_admission_behavioral_execution_fingerprint": (
                            ordering_admission_evidence[
                                "behavioral_execution_fingerprint"
                            ]
                        ),
                    }
                )
                sibling_path = token_dir / "correctness_behavioral_sibling.json"
                persist_sibling_behavioral_report(sibling, sibling_path)
                artifacts["behavioral_sibling"] = {
                    "status": "persisted",
                    "path": str(sibling_path),
                    "evidence_fingerprint": sibling.evidence_fingerprint,
                    "alias_comparator_status": sibling.alias_comparator_status.value,
                    "alias_selection_count": len(sibling.alias_selections),
                    "ordering_admission": (
                        "candidate_smoke"
                        if mode is BehavioralProbeMode.SMOKE
                        else "qualified_only"
                    ),
                    **budget_evidence,
                }
                alias_comparator_status = sibling.alias_comparator_status.value
                behavioral_alias_selection_count = len(sibling.alias_selections)
                mapped = map_sibling_behavioral_report(sibling)
                behavioral = replace(
                    mapped,
                    metrics={
                        **mapped.metrics,
                        **budget_evidence,
                        **ordering_admission_metrics,
                    },
                )
            else:
                artifacts["behavioral_sibling"] = {
                    "status": prepared.status.value,
                    "path": None,
                    **budget_evidence,
                }
        except _BehavioralPreparationDeadlineExhausted:
            pass
        except OrderingAdmissionError as error:
            _record_error(errors, "ordering_admission", error)
            artifacts["ordering_admission"] = {
                "status": "refused",
                "evidence": None,
            }
            artifacts["behavioral_sibling"] = {
                "status": "unavailable",
                "path": None,
            }
            behavioral = BehavioralFaithfulnessReport(
                status=BehavioralFaithfulnessStatus.UNKNOWN,
                metrics={
                    "policy_id": policy_id,
                    "ordering_admission_status": "refused",
                    "ordering_admission_evidence_fingerprint": None,
                },
                reason_codes=("ordering_qualification_unavailable",),
            )
        except Exception as error:
            _record_error(errors, "behavioral", error)
            artifacts["behavioral_sibling"] = {
                "status": "unavailable",
                "path": None,
            }
            behavioral = BehavioralFaithfulnessReport(
                status=BehavioralFaithfulnessStatus.UNKNOWN,
                metrics={"policy_id": policy_id},
                reason_codes=("behavioral_probe_failed",),
            )
    else:
        if not behavioral_calibration_admitted:
            reason = "frozen_correctness_calibration_unavailable"
        elif not required_alias_handoff_admitted:
            reason = "required_alias_handoff_unavailable"
        elif envelope_cleanup is not None and not envelope_cleanup.complete:
            reason = "outer_session_cleanup_incomplete"
        else:
            reason = "frontier_evidence_unavailable"
        artifacts["behavioral_sibling"] = {"status": "unavailable", "path": None}
        behavioral = BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.UNKNOWN,
            metrics={"policy_id": policy_id},
            reason_codes=(reason,),
        )

    report = CorrectnessReport(
        structural=structural,
        numerical_stability=numerical,
        behavioral_faithfulness=behavioral,
    )
    report_path = token_dir / "correctness_report.json"
    save_correctness_report(report, report_path)
    artifacts["project_report"] = {
        "status": "persisted",
        "path": str(report_path),
        "report_fingerprint": report.report_fingerprint,
        "calibration_id": (
            None if calibration is None else calibration.calibration_id
        ),
        "calibration_fingerprint": (
            None if calibration is None else calibration.calibration_fingerprint
        ),
    }
    required_policy_satisfied = (
        calibration is not None
        and structural.verdict is StructuralVerdict.CONFORMANT
        and _required_numerical_policy_satisfied(
            policy_id=policy_id,
            scopes=numerical.scopes,
            required_scope_ids=calibration.required_numerical_scopes,
            calibration_fingerprint=calibration.calibration_fingerprint,
        )
        and behavioral.status is BehavioralFaithfulnessStatus.SUPPORTED
        and _required_alias_policy_satisfied(
            policy_id=policy_id,
            scopes=numerical.scopes,
            required_scope_ids=calibration.required_numerical_scopes,
            calibration_id=calibration.calibration_id,
            calibration_fingerprint=calibration.calibration_fingerprint,
            qualified_aliases=numerical_aliases,
            behavioral=behavioral,
            alias_comparator_status=alias_comparator_status,
            behavioral_alias_selection_count=behavioral_alias_selection_count,
        )
    )
    return CorrectnessGateOutcome(
        trace_payload={
            "mode": mode.value,
            "policy_id": policy_id,
            "calibration_id": (
                None if calibration is None else calibration.calibration_id
            ),
            "calibration_fingerprint": (
                None if calibration is None else calibration.calibration_fingerprint
            ),
            "status": "evidence_recorded",
            "required_policy_satisfied": required_policy_satisfied,
            "axes": {
                "structural": structural.verdict.value,
                "numerical": [scope.status.value for scope in numerical.scopes],
                "behavioral": behavioral.status.value,
            },
            "artifacts": artifacts,
            "errors": errors,
        },
        required_policy_satisfied=required_policy_satisfied,
    )


def _run_declared_numerical_comparison(
    *,
    knobs: Mapping[str, Any],
    graph_path: Path,
    token_dir: Path,
    frontier_available: bool = False,
    calibration: FrozenCorrectnessCalibration | None = None,
) -> tuple[DeclaredNumericalEvaluation, dict[str, Any]] | None:
    declaration = (
        calibration.numerical_reference_declaration
        if calibration is not None
        else declaration_from_graph_knobs(knobs)
    )
    if declaration is None:
        return None
    evaluation_kwargs: dict[str, Any] = {}
    if calibration is not None:
        evaluation_kwargs["calibration"] = calibration
    evaluation = evaluate_declared_numerical_stability(
        candidate_graph_path=graph_path,
        declaration=declaration,
        candidate_frontier_path=(
            token_dir / "correctness_frontier.json" if frontier_available else None
        ),
        candidate_descriptor_path=(
            token_dir / "feature_semantic_descriptors.npz"
            if frontier_available
            else None
        ),
        **evaluation_kwargs,
    )
    numerical_path = token_dir / "correctness_numerical_details.json"
    write_json(numerical_path, dict(evaluation.details))
    artifact = {
        "status": "persisted",
        "path": str(numerical_path),
        "artifact_sha256": file_sha256(numerical_path),
        "manifest_sha256": declaration["manifest_sha256"],
    }
    if calibration is not None:
        artifact.update(
            {
                "calibration_id": calibration.calibration_id,
                "calibration_fingerprint": calibration.calibration_fingerprint,
            }
        )
    return evaluation, artifact


def _completed_trace_evidence(
    trace_result: Any,
    *,
    graph: Any | None,
    compact_result: Mapping[str, Any],
    spec: Mapping[str, Any],
    expected_telemetry_path: Path,
    selected_config: Mapping[str, Any],
    errors: list[dict[str, str]],
) -> CompletedTraceEvidence:
    status = getattr(trace_result, "status", None)
    status_value = getattr(status, "value", status)
    terminal_done, frontier_refresh_sequence = _closed_telemetry_evidence(
        trace_result,
        spec=spec,
        expected_path=expected_telemetry_path,
        expected_refresh_count=_expected_refresh_count(
            compact_result,
            errors=errors,
        ),
        errors=errors,
    )
    finite_values = None
    signed_accounting = None
    if graph is not None:
        finite_values = FiniteValueEvidence(
            complete=True,
            tensor_values_checked=int(graph.edge_count),
            edge_weights_checked=int(graph.edge_count),
            nonfinite_value_count=0,
        )
        signed_buckets = sum(
            bool(graph.bucket_metadata[name].get("weights_signed"))
            for name in graph.bucket_names
        )
        signed_accounting = SignedAccountingEvidence(
            complete=True,
            buckets_checked=len(graph.bucket_names),
            signed_weight_buckets=signed_buckets,
        )
    row_denominator = _row_denominator_evidence_from_compact_result(
        compact_result,
        errors=errors,
    )
    return CompletedTraceEvidence(
        terminal_success=TerminalSuccessEvidence(
            terminal_state=str(status_value or "unknown"),
            successful=status_value == "succeeded" and terminal_done,
            detail=(
                "Canonical succeeded TraceResult and terminal attribute.done observed."
                if status_value == "succeeded" and terminal_done
                else "Canonical succeeded TraceResult or terminal attribute.done is missing."
            ),
        ),
        trace_scope_cleanup=TraceScopeCleanupEvidence(
            complete=status_value == "succeeded" and terminal_done,
            resources_released=0,
            detail=(
                "Backend trace scope returned only after canonical cleanup; "
                "individual release count is not emitted."
            ),
        ),
        finite_values=finite_values,
        signed_accounting=signed_accounting,
        row_denominator=row_denominator,
        frontier_refresh_sequence=frontier_refresh_sequence,
        mechanism_requirements=_mechanism_requirements(
            trace_result,
            selected_config=selected_config,
        ),
    )


def _row_denominator_evidence_from_compact_result(
    compact_result: Mapping[str, Any],
    *,
    errors: list[dict[str, str]],
) -> RowDenominatorEvidence | None:
    payload = compact_result.get("correctness_row_denominator_evidence")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        _record_error(
            errors,
            "correctness_row_denominator_evidence",
            TypeError("correctness_row_denominator_evidence must be an object"),
        )
        return None
    try:
        expected_fields = {
            "complete",
            "policy_id",
            "rows_checked",
            "violation_count",
            "authoritative_batch_count",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "correctness row-denominator evidence fields do not match v2 schema"
            )
        batch_count = payload["authoritative_batch_count"]
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count < 0
        ):
            raise ValueError("authoritative_batch_count must be a non-negative integer")
        evidence = RowDenominatorEvidence.from_json(
            {
                key: payload[key]
                for key in expected_fields
                if key != "authoritative_batch_count"
            }
        )
        if evidence.policy_id != _ROW_DENOMINATOR_POLICY_ID:
            raise ValueError(
                "unsupported correctness row-denominator policy_id: "
                f"{evidence.policy_id!r}"
            )
        if evidence.complete and batch_count == 0:
            raise ValueError("complete row-denominator evidence requires authoritative batches")
    except (KeyError, TypeError, ValueError) as error:
        _record_error(errors, "correctness_row_denominator_evidence", error)
        return None
    return evidence


def _closed_telemetry_evidence(
    trace_result: Any,
    *,
    spec: Mapping[str, Any],
    expected_path: Path,
    expected_refresh_count: int | None,
    errors: list[dict[str, str]],
) -> tuple[bool, FrontierRefreshSequenceEvidence | None]:
    """Stream the closed, identity-bound sink; never trust its capped mirror."""

    summary = getattr(trace_result, "telemetry_summary", None)
    if not isinstance(summary, Mapping):
        _record_error(
            errors,
            "closed_telemetry_sink",
            TypeError("TraceResult.telemetry_summary must be an object"),
        )
        return False, None
    if summary.get("sink_status") != "closed":
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError("telemetry sink_status must be 'closed'"),
        )
        return False, None
    sink_path = summary.get("sink_path")
    if not isinstance(sink_path, str) or not sink_path:
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError("telemetry sink_path is required"),
        )
        return False, None
    try:
        resolved_sink = Path(sink_path).resolve(strict=True)
        resolved_expected = expected_path.resolve(strict=True)
    except OSError as error:
        _record_error(errors, "closed_telemetry_sink", error)
        return False, None
    if resolved_sink != resolved_expected:
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError(
                "telemetry sink_path does not match this token's canonical sink"
            ),
        )
        return False, None

    expected_identity = {
        "trace_id": str(spec["trace_id"]),
        "trajectory_id": str(spec["trajectory_id"]),
        "generated_index": int(spec["generated_index"]),
        "target_position": int(spec["target_position"]),
        "target_token_id": int(spec["target_token_id"]),
    }
    row_count = 0
    terminal_count = 0
    terminal_event_index: int | None = None
    refresh_count = 0
    next_refresh_index = 0
    refresh_sequence_error: str | None = None
    try:
        with resolved_sink.open("r", encoding="utf-8") as handle:
            for row_count, line in enumerate(handle, start=1):
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError(f"telemetry line {row_count} must be an object")
                for name, expected in expected_identity.items():
                    if record.get(name) != expected:
                        raise ValueError(
                            f"telemetry line {row_count} has mismatched {name}"
                        )
                if record.get("event_index") != row_count - 1:
                    raise ValueError(
                        f"telemetry line {row_count} has non-contiguous event_index"
                    )
                if record.get("sequence") != row_count:
                    raise ValueError(
                        f"telemetry line {row_count} has non-contiguous sequence"
                    )
                attrs = record.get("attrs")
                if record.get("name") == "attribute.done":
                    if not isinstance(attrs, Mapping) or attrs.get("status") != "succeeded":
                        raise ValueError(
                            "terminal attribute.done does not record status='succeeded'"
                        )
                    terminal_count += 1
                    terminal_event_index = row_count - 1
                if record.get("name") != "phase4.refresh":
                    continue
                value = attrs.get("refresh_index") if isinstance(attrs, Mapping) else None
                if not isinstance(value, int) or isinstance(value, bool):
                    refresh_sequence_error = (
                        f"telemetry line {row_count} phase4.refresh lacks refresh_index"
                    )
                    continue
                refresh_count += 1
                if value != next_refresh_index and refresh_sequence_error is None:
                    refresh_sequence_error = (
                        "phase4.refresh indices must start at zero and be contiguous; "
                        f"expected {next_refresh_index}, observed {value}"
                    )
                next_refresh_index += 1
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        _record_error(errors, "closed_telemetry_sink", error)
        return False, None

    if row_count != summary.get("sink_event_count") or row_count != summary.get(
        "event_count"
    ):
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError("telemetry sink row count does not match terminal summary"),
        )
        return False, None
    sink_error_count = summary.get("sink_error_count")
    if (
        not isinstance(sink_error_count, int)
        or isinstance(sink_error_count, bool)
        or sink_error_count != 0
    ):
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError("telemetry sink recorded write errors"),
        )
        return False, None
    if terminal_count != 1 or terminal_event_index != row_count - 1:
        _record_error(
            errors,
            "closed_telemetry_sink",
            ValueError("telemetry sink must end with exactly one attribute.done event"),
        )
        return False, None
    if expected_refresh_count is None:
        _record_error(
            errors,
            "frontier_refresh_sequence",
            ValueError("explicit phase4_refresh_count evidence is required"),
        )
        return True, None
    if refresh_count != expected_refresh_count:
        refresh_sequence_error = (
            "phase4.refresh event count does not match explicit "
            f"phase4_refresh_count: {refresh_count} != {expected_refresh_count}"
        )
    if refresh_sequence_error is not None:
        _record_error(
            errors,
            "frontier_refresh_sequence",
            ValueError(refresh_sequence_error),
        )
        return True, None
    return True, FrontierRefreshSequenceEvidence(
        complete=True,
        events_checked=refresh_count,
        monotonic=True,
        duplicate_event_count=0,
    )


def _expected_refresh_count(
    compact_result: Mapping[str, Any],
    *,
    errors: list[dict[str, str]],
) -> int | None:
    value = compact_result.get("phase4_refresh_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        if value is not None:
            _record_error(
                errors,
                "frontier_refresh_sequence",
                ValueError("phase4_refresh_count must be a non-negative int"),
            )
        return None
    return value


def _required_numerical_policy_satisfied(
    *,
    policy_id: str,
    scopes: tuple[NumericalScopeReport, ...],
    required_scope_ids: tuple[str, ...] | None = None,
    calibration_fingerprint: str | None = None,
) -> bool:
    allowed = _REQUIRED_NUMERICAL_STATUSES_BY_POLICY.get(policy_id)
    if allowed is None or not scopes:
        return False
    scopes_by_id = {scope.scope_id: scope for scope in scopes}
    if len(scopes_by_id) != len(scopes):
        return False
    admitted_scopes = scopes
    if required_scope_ids is not None:
        if not required_scope_ids or len(set(required_scope_ids)) != len(
            required_scope_ids
        ):
            return False
        if any(scope_id not in scopes_by_id for scope_id in required_scope_ids):
            return False
        admitted_scopes = tuple(
            scopes_by_id[scope_id] for scope_id in required_scope_ids
        )
    for scope in admitted_scopes:
        if scope.status not in allowed:
            return False
        if scope.status in {
            NumericalStabilityStatus.ALIAS_STABLE,
            NumericalStabilityStatus.BOUNDED,
        }:
            classification_policy_id = scope.metrics.get("classification_policy_id")
            scope_calibration_fingerprint = scope.metrics.get(
                "calibration_fingerprint"
            )
            if (
                not isinstance(classification_policy_id, str)
                or not classification_policy_id
                or (
                    scope.status is NumericalStabilityStatus.ALIAS_STABLE
                    and classification_policy_id != policy_id
                )
                or (
                    calibration_fingerprint is not None
                    and classification_policy_id != policy_id
                )
            ):
                return False
            if not _is_sha256_fingerprint(scope_calibration_fingerprint):
                return False
            if (
                calibration_fingerprint is not None
                and scope_calibration_fingerprint != calibration_fingerprint
            ):
                return False
    return True


def _required_alias_scope_ids(
    *,
    scopes: tuple[NumericalScopeReport, ...],
    required_scope_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if required_scope_ids is None:
        return tuple(
            scope.scope_id
            for scope in scopes
            if scope.status is NumericalStabilityStatus.ALIAS_STABLE
        )
    required = set(required_scope_ids)
    return tuple(
        scope.scope_id
        for scope in scopes
        if scope.scope_id in required
        and scope.status is NumericalStabilityStatus.ALIAS_STABLE
    )


def _qualified_alias_handoff_satisfied(
    *,
    policy_id: str,
    calibration_fingerprint: str,
    qualified_aliases: tuple[QualifiedAliasPair, ...],
) -> bool:
    return bool(qualified_aliases) and all(
        alias.selection_policy_id == policy_id
        and alias.calibration_fingerprint == calibration_fingerprint
        for alias in qualified_aliases
    )


def _required_alias_policy_satisfied(
    *,
    policy_id: str,
    scopes: tuple[NumericalScopeReport, ...],
    required_scope_ids: tuple[str, ...],
    calibration_id: str,
    calibration_fingerprint: str,
    qualified_aliases: tuple[QualifiedAliasPair, ...],
    behavioral: BehavioralFaithfulnessReport,
    alias_comparator_status: str | None,
    behavioral_alias_selection_count: int,
) -> bool:
    """Bind required numerical alias churn to its executed behavioral comparator."""

    if not _required_alias_scope_ids(
        scopes=scopes,
        required_scope_ids=required_scope_ids,
    ):
        return True
    if not calibration_id or not _is_sha256_fingerprint(calibration_fingerprint):
        return False
    if not _qualified_alias_handoff_satisfied(
        policy_id=policy_id,
        calibration_fingerprint=calibration_fingerprint,
        qualified_aliases=qualified_aliases,
    ):
        return False
    return (
        behavioral.status is BehavioralFaithfulnessStatus.SUPPORTED
        and behavioral.metrics.get("policy_id") == policy_id
        and behavioral.metrics.get("calibration_id") == calibration_id
        and behavioral.metrics.get("evidence_completeness") == "complete"
        and behavioral.metrics.get("runtime_status") == "complete"
        and alias_comparator_status == "complete"
        and behavioral_alias_selection_count >= 1
    )


def _is_sha256_fingerprint(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _mechanism_requirements(
    trace_result: Any,
    *,
    selected_config: Mapping[str, Any],
) -> tuple[MechanismRequirementObservation, ...]:
    descriptor = getattr(trace_result, "effective_execution", None)
    if descriptor is None:
        effective = None
    elif isinstance(descriptor, Mapping):
        effective = descriptor
    else:
        effective = descriptor.to_dict()
    resolution = execution_resolution(
        selected_config=selected_config,
        effective_execution=effective,
    )
    required = {"backward_engine_mode"}
    if selected_config.get("feature_row_influence_requirement") == "required":
        required.add("feature_row_influence_mode")
    fields = resolution.get("fields")
    if not isinstance(fields, list):
        fields = []
    by_name = {
        str(row.get("field")): row for row in fields if isinstance(row, Mapping)
    }
    observations = []
    for name in sorted(required):
        row = by_name.get(name, {})
        requested = row.get("requested", selected_config.get(name))
        observed = row.get("effective")
        observations.append(
            MechanismRequirementObservation(
                requirement_id=name,
                expected=str(requested),
                observed=None if observed is None else str(observed),
                satisfied=requested is not None and requested == observed,
                detail=(
                    None
                    if requested is not None and requested == observed
                    else "Requested/effective execution resolution did not match."
                ),
            )
        )
    return tuple(observations)


def _decoder_fingerprint(transcoder_metadata: Mapping[str, Any]) -> str:
    detected = _mapping(transcoder_metadata.get("detected"), "transcoder.detected")
    provider = detected.get("provider_fingerprint")
    if not isinstance(provider, Mapping):
        raise ValueError("loaded provider fingerprint is unavailable")
    payload = json.dumps(
        provider,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _qualified_alias_pairs(
    compact_result: Mapping[str, Any],
) -> tuple[QualifiedAliasPair, ...]:
    """Read only explicit, comparison-owned alias evidence; never infer from sketches."""

    payload = compact_result.get("correctness_qualified_alias_pairs")
    if payload is None:
        return ()
    if not isinstance(payload, (list, tuple)):
        raise TypeError("correctness_qualified_alias_pairs must be an array")
    if len(payload) > 2:
        raise ValueError("correctness_qualified_alias_pairs exceeds the bound of two")
    result = []
    for index, raw in enumerate(payload):
        row = _mapping(raw, f"correctness_qualified_alias_pairs[{index}]")
        result.append(
            QualifiedAliasPair(
                source=_alias_feature_key(row.get("source"), f"alias[{index}].source"),
                substitute=_alias_feature_key(
                    row.get("substitute"), f"alias[{index}].substitute"
                ),
                selection_policy_id=str(row.get("selection_policy_id", "")),
                calibration_fingerprint=str(row.get("calibration_fingerprint", "")),
                comparison_evidence_fingerprint=str(
                    row.get("comparison_evidence_fingerprint", "")
                ),
                baseline_graph_fingerprint=str(
                    row.get("baseline_graph_fingerprint", "")
                ),
                candidate_graph_fingerprint=str(
                    row.get("candidate_graph_fingerprint", "")
                ),
                qualified_decoder_cosine=_finite_float(
                    row.get("qualified_decoder_cosine"),
                    f"alias[{index}].qualified_decoder_cosine",
                ),
            )
        )
    return tuple(result)


def _alias_feature_key(value: Any, label: str) -> FeatureKey:
    row = _mapping(value, label)
    coordinates = []
    for name in ("layer", "position", "feature_id"):
        coordinate = row.get(name)
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise TypeError(f"{label}.{name} must be an integer")
        coordinates.append(coordinate)
    return FeatureKey(*coordinates)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _record_error(
    errors: list[dict[str, str]],
    stage: str,
    error: BaseException,
) -> None:
    errors.append(
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "detail": str(error),
        }
    )
