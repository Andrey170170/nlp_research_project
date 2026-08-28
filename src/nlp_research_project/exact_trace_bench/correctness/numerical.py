"""Declared, provenance-bound numerical comparisons for correctness evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..graph_compare import compare_compact_paths
from ..trace_runtime.artifacts import load_and_validate_capture_artifact
from ..typed_compact_graph import load_typed_compact_graph
from .contracts import (
    NumericalScopeReport,
    NumericalStabilityReport,
    NumericalStabilityStatus,
)
from .frontier import FeatureKey, FrontierEvidence, FrontierRecord
from .frontier_artifact import load_bounded_frontier_artifact
from .stability import (
    AliasMatchPolicy,
    BoundedFrontierDeltaPolicy,
    DecoderVectorSource,
    FrozenFrontierClassificationPolicy,
    FrontierStabilityReport,
    classify_frontier_stability,
    compare_frontiers,
)

if TYPE_CHECKING:
    from .behavioral import QualifiedAliasPair
    from .calibration import FrozenCorrectnessCalibration


NUMERICAL_REFERENCE_MANIFEST_SCHEMA_VERSION = 1
NUMERICAL_REFERENCE_MANIFEST_FORMAT = "exact_trace_numerical_reference_manifest_v1"
NUMERICAL_REFERENCE_ROLES = ("repeat", "canonical")
NUMERICAL_MANIFEST_PATH_KNOB = "correctness_numerical_manifest_path"
NUMERICAL_MANIFEST_SHA256_KNOB = "correctness_numerical_manifest_sha256"

_IDENTITY_FIELDS = (
    "artifact_sha256",
    "schema_version",
    "compact_save_format",
    "step_idx",
    "content_fingerprint",
    "graph_fingerprint",
    "target_fingerprint",
    "provider_fingerprint",
    "trace_fingerprint",
    "step_fingerprint",
    "retention_policy_id",
    "retention_policy_fingerprint",
)
_REFERENCE_BASE_FIELDS = {
    "reference_id",
    "role",
    "graph_path",
    "graph_sha256",
    "identity",
}
_REFERENCE_FRONTIER_FIELDS = {
    "frontier_path",
    "frontier_sha256",
    "descriptor_path",
    "descriptor_sha256",
}
_RAW_FRONTIER_PAIR_LIMIT = 8


@dataclass(frozen=True)
class DeclaredNumericalEvaluation:
    report: NumericalStabilityReport
    details: Mapping[str, Any]
    qualified_alias_pairs: tuple[QualifiedAliasPair, ...] = ()


@dataclass(frozen=True)
class _MappingDecoderSource(DecoderVectorSource):
    vectors: Mapping[FeatureKey, np.ndarray]

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray:
        try:
            return self.vectors[feature]
        except KeyError as exc:
            raise ValueError(f"decoder descriptor absent for {feature!r}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_reference_identity_receipt(graph_path: Path) -> dict[str, Any]:
    """Strictly reopen one typed graph and return its immutable identity receipt."""

    path = Path(graph_path).resolve()
    graph = load_typed_compact_graph(path)
    return {
        "artifact_sha256": file_sha256(path),
        "schema_version": int(graph.schema_version),
        "compact_save_format": str(graph.compact_save_format),
        "step_idx": int(graph.step_idx),
        "content_fingerprint": str(graph.content_fingerprint),
        "graph_fingerprint": str(graph.graph_fingerprint),
        "target_fingerprint": str(graph.target_fingerprint),
        "provider_fingerprint": str(graph.provider_fingerprint),
        "trace_fingerprint": str(graph.trace_fingerprint),
        "step_fingerprint": str(graph.step_fingerprint),
        "retention_policy_id": str(graph.retention_policy_id),
        "retention_policy_fingerprint": str(graph.retention_policy_fingerprint),
    }


def prepare_numerical_manifest_declaration(manifest_path: Path) -> dict[str, Any]:
    """Hash-check a manifest and all referenced graphs for trace-spec capture."""

    path = Path(manifest_path).resolve()
    manifest_sha256 = file_sha256(path)
    _load_and_validate_manifest(path, expected_sha256=manifest_sha256)
    return {
        "schema_version": NUMERICAL_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "format": NUMERICAL_REFERENCE_MANIFEST_FORMAT,
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
    }


def declaration_from_graph_knobs(knobs: Mapping[str, Any]) -> dict[str, Any] | None:
    path = knobs.get(NUMERICAL_MANIFEST_PATH_KNOB)
    sha256 = knobs.get(NUMERICAL_MANIFEST_SHA256_KNOB)
    if path is None and sha256 is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError(f"graph_knobs.{NUMERICAL_MANIFEST_PATH_KNOB} is required")
    _require_sha256(sha256, label=f"graph_knobs.{NUMERICAL_MANIFEST_SHA256_KNOB}")
    return {
        "schema_version": NUMERICAL_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "format": NUMERICAL_REFERENCE_MANIFEST_FORMAT,
        "manifest_path": path,
        "manifest_sha256": sha256,
    }


def evaluate_declared_numerical_stability(
    *,
    candidate_graph_path: Path,
    declaration: Mapping[str, Any],
    candidate_frontier_path: Path | None = None,
    candidate_descriptor_path: Path | None = None,
    calibration: FrozenCorrectnessCalibration | None = None,
) -> DeclaredNumericalEvaluation:
    """Revalidate declared inputs and compare each role under frozen thresholds."""

    _validate_declaration(declaration)
    manifest_path = Path(str(declaration["manifest_path"]))
    manifest = _load_and_validate_manifest(
        manifest_path,
        expected_sha256=str(declaration["manifest_sha256"]),
    )
    candidate_path = Path(candidate_graph_path).resolve()
    candidate_sha256 = file_sha256(candidate_path)
    references = {str(row["role"]): row for row in manifest["references"]}
    scopes: list[NumericalScopeReport] = []
    comparison_details: list[dict[str, Any]] = []
    frontier_details: list[dict[str, Any]] = []
    qualified_alias_pairs: list[QualifiedAliasPair] = []
    for role in NUMERICAL_REFERENCE_ROLES:
        reference = references.get(role)
        scope_id = f"typed_graph.{role}"
        if reference is None:
            scopes.append(
                NumericalScopeReport(
                    scope_id=scope_id,
                    status=NumericalStabilityStatus.UNKNOWN,
                    metrics={
                        "reference_count": 0,
                        "repeat_count": 0,
                        "reference_role": role,
                    },
                    reason_codes=(f"declared_{role}_reference_absent",),
                )
            )
            scopes.append(
                NumericalScopeReport(
                    scope_id=f"feature_frontier.{role}",
                    status=NumericalStabilityStatus.UNKNOWN,
                    metrics={"reference_count": 0, "reference_role": role},
                    reason_codes=(f"declared_{role}_reference_absent",),
                )
            )
            continue
        reference_path = Path(str(reference["graph_path"]))
        comparison = compare_compact_paths(reference_path, candidate_path)
        scope_compatible, compatibility = _comparison_scope_compatibility(
            reference_path=reference_path,
            candidate_path=candidate_path,
            comparison=comparison,
        )
        exact = scope_compatible and _comparison_is_exact(comparison)
        metrics = _scalar_comparison_metrics(
            comparison,
            role=role,
            reference=reference,
            exact=exact,
        )
        metrics.update(compatibility)
        status, reason_codes = _classify_typed_graph_comparison(
            comparison,
            scope_compatible=scope_compatible,
            exact=exact,
            calibration=calibration,
        )
        _annotate_calibration_metrics(metrics, calibration=calibration)
        scopes.append(
            NumericalScopeReport(
                scope_id=scope_id,
                status=status,
                metrics=metrics,
                reason_codes=reason_codes,
            )
        )
        comparison_details.append(
            {
                "reference_id": reference["reference_id"],
                "role": role,
                "graph_path": reference["graph_path"],
                "graph_sha256": reference["graph_sha256"],
                "identity": reference["identity"],
                "scope_compatible": scope_compatible,
                "compatibility": compatibility,
                "comparison_exact": exact,
                "comparison": comparison,
            }
        )
        frontier_scope, frontier_detail, frontier_aliases = _evaluate_declared_frontier(
            role=role,
            reference=reference,
            typed_graph_exact=exact,
            candidate_graph_fingerprint=str(
                build_reference_identity_receipt(candidate_path)["graph_fingerprint"]
            ),
            candidate_frontier_path=candidate_frontier_path,
            candidate_descriptor_path=candidate_descriptor_path,
            calibration=calibration,
        )
        scopes.append(frontier_scope)
        if frontier_detail is not None:
            frontier_details.append(frontier_detail)
        if role == "repeat":
            qualified_alias_pairs.extend(frontier_aliases)
    details = {
        "schema_version": 1,
        "format": "exact_trace_numerical_comparison_details_v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": declaration["manifest_sha256"],
        "candidate_graph_path": str(candidate_path),
        "candidate_graph_sha256": candidate_sha256,
        "comparisons": comparison_details,
        "frontier_comparisons": frontier_details,
    }
    return DeclaredNumericalEvaluation(
        report=NumericalStabilityReport(scopes=tuple(scopes)),
        details=details,
        qualified_alias_pairs=tuple(qualified_alias_pairs),
    )


def _load_and_validate_manifest(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="manifest_sha256")
    actual_manifest_sha256 = file_sha256(path)
    if actual_manifest_sha256 != expected_sha256:
        raise ValueError(
            "numerical reference manifest sha256 mismatch: "
            f"expected {expected_sha256}, observed {actual_manifest_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "format",
        "references",
    }:
        raise ValueError("numerical reference manifest has unexpected fields")
    if payload["schema_version"] != NUMERICAL_REFERENCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported numerical reference manifest schema_version")
    if payload["format"] != NUMERICAL_REFERENCE_MANIFEST_FORMAT:
        raise ValueError("unsupported numerical reference manifest format")
    references = payload["references"]
    if not isinstance(references, list) or not references:
        raise ValueError("numerical reference manifest requires references")
    seen_roles: set[str] = set()
    seen_ids: set[str] = set()
    for index, raw_value in enumerate(references):
        fields = set(raw_value) if isinstance(raw_value, dict) else set()
        if not isinstance(raw_value, dict) or fields not in {
            frozenset(_REFERENCE_BASE_FIELDS),
            frozenset(_REFERENCE_BASE_FIELDS | _REFERENCE_FRONTIER_FIELDS),
        }:
            raise ValueError(f"numerical reference manifest references[{index}] invalid")
        raw = cast(Mapping[str, Any], raw_value)
        reference_id = raw["reference_id"]
        role = raw["role"]
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError("numerical reference_id is required")
        if reference_id in seen_ids:
            raise ValueError(f"duplicate numerical reference_id: {reference_id!r}")
        if role not in NUMERICAL_REFERENCE_ROLES:
            raise ValueError(f"unsupported numerical reference role: {role!r}")
        if role in seen_roles:
            raise ValueError(f"duplicate numerical reference role: {role!r}")
        graph_path = Path(str(raw["graph_path"]))
        if not graph_path.is_absolute():
            raise ValueError("numerical reference graph_path must be absolute")
        expected_graph_sha256 = raw["graph_sha256"]
        _require_sha256(expected_graph_sha256, label="graph_sha256")
        actual_graph_sha256 = file_sha256(graph_path)
        if actual_graph_sha256 != expected_graph_sha256:
            raise ValueError(
                f"numerical reference artifact sha256 mismatch for {reference_id!r}: "
                f"expected {expected_graph_sha256}, observed {actual_graph_sha256}"
            )
        identity_value = raw["identity"]
        if not isinstance(identity_value, dict) or set(identity_value) != set(
            _IDENTITY_FIELDS
        ):
            raise ValueError(f"numerical reference identity invalid for {reference_id!r}")
        identity = cast(Mapping[str, Any], identity_value)
        observed_identity = build_reference_identity_receipt(graph_path)
        if identity != observed_identity:
            raise ValueError(
                f"numerical reference identity mismatch for {reference_id!r}"
            )
        if identity["artifact_sha256"] != expected_graph_sha256:
            raise ValueError(
                f"numerical reference receipt hash mismatch for {reference_id!r}"
            )
        if _REFERENCE_FRONTIER_FIELDS <= fields:
            _validate_reference_frontier_artifacts(
                raw,
                reference_id=reference_id,
                graph_fingerprint=str(identity["graph_fingerprint"]),
            )
        seen_ids.add(reference_id)
        seen_roles.add(str(role))
    return payload


def _validate_declaration(declaration: Mapping[str, Any]) -> None:
    if set(declaration) != {
        "schema_version",
        "format",
        "manifest_path",
        "manifest_sha256",
    }:
        raise ValueError("numerical manifest declaration has unexpected fields")
    if declaration["schema_version"] != NUMERICAL_REFERENCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("numerical manifest declaration schema_version mismatch")
    if declaration["format"] != NUMERICAL_REFERENCE_MANIFEST_FORMAT:
        raise ValueError("numerical manifest declaration format mismatch")
    manifest_path = declaration["manifest_path"]
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
        raise ValueError("numerical manifest declaration path must be absolute")
    _require_sha256(declaration["manifest_sha256"], label="manifest_sha256")


def _validate_reference_frontier_artifacts(
    reference: Mapping[str, Any],
    *,
    reference_id: str,
    graph_fingerprint: str,
) -> None:
    for kind in ("frontier", "descriptor"):
        path = Path(str(reference[f"{kind}_path"]))
        if not path.is_absolute():
            raise ValueError(f"numerical reference {kind}_path must be absolute")
        expected = reference[f"{kind}_sha256"]
        _require_sha256(expected, label=f"{kind}_sha256")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"numerical reference {kind} sha256 mismatch for {reference_id!r}: "
                f"expected {expected}, observed {observed}"
            )
    frontier = load_bounded_frontier_artifact(Path(str(reference["frontier_path"])))
    descriptor = load_and_validate_capture_artifact(
        "feature_semantic_descriptors",
        Path(str(reference["descriptor_path"])),
    )
    if frontier.evidence.graph_fingerprint != graph_fingerprint:
        raise ValueError("numerical reference frontier graph fingerprint mismatch")
    _validate_frontier_descriptor_identity(frontier, descriptor)


def _evaluate_declared_frontier(
    *,
    role: str,
    reference: Mapping[str, Any],
    typed_graph_exact: bool,
    candidate_graph_fingerprint: str,
    candidate_frontier_path: Path | None,
    candidate_descriptor_path: Path | None,
    calibration: FrozenCorrectnessCalibration | None,
) -> tuple[
    NumericalScopeReport,
    dict[str, Any] | None,
    tuple[QualifiedAliasPair, ...],
]:
    scope_id = f"feature_frontier.{role}"
    if not _REFERENCE_FRONTIER_FIELDS <= set(reference):
        return (
            NumericalScopeReport(
                scope_id=scope_id,
                status=NumericalStabilityStatus.UNKNOWN,
                metrics={"reference_count": 1, "reference_role": role},
                reason_codes=(f"declared_{role}_frontier_absent",),
            ),
            None,
            (),
        )
    if candidate_frontier_path is None or candidate_descriptor_path is None:
        return (
            NumericalScopeReport(
                scope_id=scope_id,
                status=NumericalStabilityStatus.UNKNOWN,
                metrics={"reference_count": 1, "reference_role": role},
                reason_codes=("candidate_frontier_unavailable",),
            ),
            None,
            (),
        )
    try:
        baseline = load_bounded_frontier_artifact(Path(str(reference["frontier_path"])))
        candidate = load_bounded_frontier_artifact(candidate_frontier_path)
        baseline_descriptor = load_and_validate_capture_artifact(
            "feature_semantic_descriptors", Path(str(reference["descriptor_path"]))
        )
        candidate_descriptor = load_and_validate_capture_artifact(
            "feature_semantic_descriptors", candidate_descriptor_path
        )
        _validate_frontier_descriptor_identity(baseline, baseline_descriptor)
        _validate_frontier_descriptor_identity(candidate, candidate_descriptor)
        if candidate.evidence.graph_fingerprint != candidate_graph_fingerprint:
            raise ValueError("candidate frontier graph fingerprint mismatch")
        raw_metrics, detail, frontier_exact = _raw_frontier_comparison(
            baseline.evidence,
            candidate.evidence,
            baseline_descriptor=baseline_descriptor,
            candidate_descriptor=candidate_descriptor,
        )
        if calibration is not None:
            stability = compare_frontiers(
                baseline.evidence,
                candidate.evidence,
                decoder_source=_combined_descriptor_source(
                    baseline_descriptor,
                    candidate_descriptor,
                ),
                policy=_alias_match_policy(
                    baseline.evidence,
                    candidate.evidence,
                    calibration=calibration,
                ),
            )
    except (OSError, TypeError, ValueError) as error:
        return (
            NumericalScopeReport(
                scope_id=scope_id,
                status=NumericalStabilityStatus.UNKNOWN,
                metrics={"reference_count": 1, "reference_role": role},
                reason_codes=("frontier_evidence_incompatible",),
            ),
            {"role": role, "status": "incompatible", "detail": str(error)},
            (),
        )
    exact = bool(typed_graph_exact and frontier_exact)
    raw_metrics.update(
        {
            "reference_count": 1,
            "reference_role": role,
            "comparison_exact": exact,
        }
    )
    if calibration is None:
        return (
            NumericalScopeReport(
                scope_id=scope_id,
                status=(
                    NumericalStabilityStatus.EXACT_STABLE
                    if exact
                    else NumericalStabilityStatus.UNKNOWN
                ),
                metrics=raw_metrics,
                reason_codes=(
                    () if exact else ("frozen_frontier_calibration_absent",)
                ),
            ),
            {"role": role, "status": "compared", **detail},
            (),
        )

    pass_policy = _frontier_classification_policy(
        calibration=calibration,
        review=False,
    )
    classified = classify_frontier_stability(
        stability,
        frozen_policy=pass_policy,
        exact_graph_or_scope_equal=exact,
    )
    if classified.status is NumericalStabilityStatus.DIVERGENT:
        review_classified = classify_frontier_stability(
            stability,
            frozen_policy=_frontier_classification_policy(
                calibration=calibration,
                review=True,
            ),
            exact_graph_or_scope_equal=exact,
        )
        if review_classified.status in {
            NumericalStabilityStatus.BOUNDED,
            NumericalStabilityStatus.ALIAS_STABLE,
        }:
            classified = NumericalScopeReport(
                scope_id=classified.scope_id,
                status=NumericalStabilityStatus.REVIEW,
                metrics=review_classified.metrics,
                reason_codes=("frontier_metrics_in_review_band",),
            )
    metrics = {**classified.metrics, **raw_metrics}
    _annotate_calibration_metrics(metrics, calibration=calibration)
    metrics["alias_behavioral_effect_check"] = "deferred_to_behavioral_axis"
    qualified_aliases = (
        _qualified_alias_pairs_from_stability(
            stability,
            candidate_frontier=candidate.evidence,
            calibration=calibration,
        )
        if classified.status is NumericalStabilityStatus.ALIAS_STABLE
        else ()
    )
    return (
        NumericalScopeReport(
            scope_id=scope_id,
            status=classified.status,
            metrics=metrics,
            reason_codes=classified.reason_codes,
        ),
        {
            "role": role,
            "status": "compared",
            **detail,
            "classification": {
                "status": classified.status.value,
                "reason_codes": list(classified.reason_codes),
                "metrics": dict(classified.metrics),
            },
        },
        qualified_aliases,
    )


def _validate_frontier_descriptor_identity(
    frontier: Any,
    descriptor: Mapping[str, Any],
) -> None:
    if not frontier.descriptor_is_decoder_evidence:
        raise ValueError("frontier does not contain decoder evidence")
    identities = {
        "decoder_source_fingerprint": frontier.decoder_source_fingerprint,
        "decoder_evidence_fingerprint": frontier.decoder_evidence_fingerprint,
        "projection_fingerprint": frontier.projection_fingerprint,
    }
    for field, expected in identities.items():
        if expected != _scalar_text(descriptor.get(field), field):
            raise ValueError(f"frontier {field} does not match descriptor artifact")


def _combined_descriptor_source(
    baseline_descriptor: Mapping[str, Any],
    candidate_descriptor: Mapping[str, Any],
) -> DecoderVectorSource:
    baseline_vectors, baseline_identity = _descriptor_vectors(baseline_descriptor)
    candidate_vectors, candidate_identity = _descriptor_vectors(candidate_descriptor)
    if baseline_identity != candidate_identity:
        raise ValueError("frontier descriptor projection identities differ")
    # One provider/projection identity denotes one decoder source. Shared keys
    # need only one copy; raw comparison separately retains whether the captured
    # arrays were byte-for-byte equal.
    return _MappingDecoderSource({**baseline_vectors, **candidate_vectors})


def _alias_match_policy(
    baseline: FrontierEvidence,
    candidate: FrontierEvidence,
    *,
    calibration: FrozenCorrectnessCalibration,
) -> AliasMatchPolicy:
    frontier = calibration.numerical.frontier
    influence_total = max(
        sum(abs(record.influence) for record in baseline.selected),
        sum(abs(record.influence) for record in candidate.selected),
    )
    return AliasMatchPolicy(
        min_decoder_cosine=frontier.alias_min_decoder_cosine,
        min_activation_scaled_cosine=frontier.alias_min_decoder_cosine,
        min_feature_neighborhood_cosine=-1.0,
        min_logit_neighborhood_cosine=-1.0,
        signed_effect_atol=float("inf"),
        signed_effect_rtol=0.0,
        high_influence_mass=(
            frontier.unmatched_influence_mass_fraction.review_max
            * influence_total
        ),
        high_edge_mass=float("inf"),
    )


def _frontier_classification_policy(
    *,
    calibration: FrozenCorrectnessCalibration,
    review: bool,
) -> FrozenFrontierClassificationPolicy:
    frontier = calibration.numerical.frontier
    selected_min = (
        frontier.selected_recovery.review_min
        if review
        else frontier.selected_recovery.pass_min
    )
    unmatched_max = (
        frontier.unmatched_influence_mass_fraction.review_max
        if review
        else frontier.unmatched_influence_mass_fraction.pass_max
    )
    bounded = frontier.bounded_deltas
    return FrozenFrontierClassificationPolicy(
        policy_id=calibration.policy_id,
        calibration_fingerprint=calibration.calibration_fingerprint,
        min_baseline_selected_recovery=selected_min,
        min_candidate_selected_recovery=selected_min,
        min_baseline_influence_recovery=max(0.0, 1.0 - unmatched_max),
        min_candidate_influence_recovery=max(0.0, 1.0 - unmatched_max),
        min_baseline_edge_mass_recovery=0.0,
        min_candidate_edge_mass_recovery=0.0,
        max_unmatched_high_mass_churn=0,
        bounded_deltas=BoundedFrontierDeltaPolicy(
            max_cutoff_score_abs_delta=bounded.max_cutoff_score_abs_delta,
            max_relative_cutoff_gap_abs_delta=(
                bounded.max_relative_cutoff_gap_abs_delta
            ),
            max_cutoff_tie_count_abs_delta=(
                bounded.max_cutoff_tie_count_abs_delta
            ),
            max_record_scalar_abs_delta=bounded.max_record_scalar_abs_delta,
            max_typed_edge_weight_abs_delta=(
                bounded.max_typed_edge_weight_abs_delta
            ),
        ),
    )


def _qualified_alias_pairs_from_stability(
    report: FrontierStabilityReport,
    *,
    candidate_frontier: FrontierEvidence,
    calibration: FrozenCorrectnessCalibration,
) -> tuple[QualifiedAliasPair, ...]:
    """Project admitted repeat churn into the behavioral alias handoff.

    Behavioral substitution runs on the candidate graph, so the candidate's
    newly selected identity is the source. The baseline identity is usable as
    its substitute only when that identity remains in the candidate frontier as
    an unselected record.
    """

    from .behavioral import MAX_QUALIFIED_ALIASES, QualifiedAliasPair

    candidate_records = {record.key: record for record in candidate_frontier.records}
    exited = set(report.exact_exited_selected)
    entered = set(report.exact_entered_selected)
    eligible = [
        match
        for match in report.matches
        if not match.exact_identity
        and match.baseline in exited
        and match.candidate in entered
        and match.baseline in candidate_records
        and not candidate_records[match.baseline].selected
        and candidate_records.get(match.candidate) is not None
        and candidate_records[match.candidate].selected
    ]
    eligible.sort(
        key=lambda match: (
            -abs(candidate_records[match.candidate].influence),
            match.candidate,
            match.baseline,
        )
    )
    admitted = eligible[:MAX_QUALIFIED_ALIASES]
    if not admitted:
        return ()
    comparison_payload = {
        "schema_version": 1,
        "format": "exact_trace_qualified_alias_comparison_v1",
        "selection_policy_id": calibration.policy_id,
        "calibration_fingerprint": calibration.calibration_fingerprint,
        "baseline_graph_fingerprint": report.baseline_graph_fingerprint,
        "candidate_graph_fingerprint": report.candidate_graph_fingerprint,
        "provider_fingerprint": report.provider_fingerprint,
        "decoder_fingerprint": report.decoder_fingerprint,
        "pairs": [
            {
                "source": _feature_key_json(match.candidate),
                "substitute": _feature_key_json(match.baseline),
                "qualified_decoder_cosine": match.decoder_cosine,
            }
            for match in admitted
        ],
    }
    encoded = json.dumps(
        comparison_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    comparison_fingerprint = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return tuple(
        QualifiedAliasPair(
            source=match.candidate,
            substitute=match.baseline,
            selection_policy_id=calibration.policy_id,
            calibration_fingerprint=calibration.calibration_fingerprint,
            comparison_evidence_fingerprint=comparison_fingerprint,
            baseline_graph_fingerprint=report.baseline_graph_fingerprint,
            candidate_graph_fingerprint=report.candidate_graph_fingerprint,
            qualified_decoder_cosine=match.decoder_cosine,
        )
        for match in admitted
    )
def _raw_frontier_comparison(
    baseline: FrontierEvidence,
    candidate: FrontierEvidence,
    *,
    baseline_descriptor: Mapping[str, Any],
    candidate_descriptor: Mapping[str, Any],
) -> tuple[dict[str, bool | int | float | str | None], dict[str, Any], bool]:
    if baseline.provider_fingerprint != candidate.provider_fingerprint:
        raise ValueError("frontier provider fingerprints differ")
    if baseline.decoder_fingerprint != candidate.decoder_fingerprint:
        raise ValueError("frontier decoder fingerprints differ")
    baseline_vectors, baseline_identity = _descriptor_vectors(baseline_descriptor)
    candidate_vectors, candidate_identity = _descriptor_vectors(candidate_descriptor)
    if baseline_identity != candidate_identity:
        raise ValueError("frontier descriptor projection identities differ")

    baseline_selected = {record.key: record for record in baseline.selected}
    candidate_selected = {record.key: record for record in candidate.selected}
    exited = _bounded_churn_records(
        tuple(
            record
            for key, record in baseline_selected.items()
            if key not in candidate_selected
        )
    )
    entered = _bounded_churn_records(
        tuple(
            record
            for key, record in candidate_selected.items()
            if key not in baseline_selected
        )
    )
    candidates: list[dict[str, Any]] = []
    for left in exited:
        for right in entered:
            if left.key.layer != right.key.layer or left.key.position != right.key.position:
                continue
            left_vector = baseline_vectors.get(left.key)
            right_vector = candidate_vectors.get(right.key)
            if left_vector is None or right_vector is None:
                continue
            candidates.append(
                {
                    "baseline": left,
                    "candidate": right,
                    "decoder_cosine": _dense_cosine(left_vector, right_vector),
                    "activation_scaled_cosine": _dense_cosine(
                        left.activation * left_vector,
                        right.activation * right_vector,
                    ),
                    "signed_effect_delta": abs(
                        left.signed_target_effect - right.signed_target_effect
                    ),
                    "feature_neighborhood_cosine": _sparse_cosine(
                        {edge.source: edge.weight for edge in left.feature_from_feature},
                        {edge.source: edge.weight for edge in right.feature_from_feature},
                    ),
                    "logit_neighborhood_cosine": _sparse_cosine(
                        {
                            edge.logit_token_id: edge.weight
                            for edge in left.logit_from_feature
                        },
                        {
                            edge.logit_token_id: edge.weight
                            for edge in right.logit_from_feature
                        },
                    ),
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["decoder_cosine"]),
            cast(FrontierRecord, row["baseline"]).key,
            cast(FrontierRecord, row["candidate"]).key,
        )
    )
    used_left: set[FeatureKey] = set()
    used_right: set[FeatureKey] = set()
    pairs: list[dict[str, Any]] = []
    for row in candidates:
        left = cast(FrontierRecord, row["baseline"])
        right = cast(FrontierRecord, row["candidate"])
        if left.key in used_left or right.key in used_right:
            continue
        used_left.add(left.key)
        used_right.add(right.key)
        pairs.append(
            {
                "baseline": _feature_key_json(left.key),
                "candidate": _feature_key_json(right.key),
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"baseline", "candidate"}
                },
            }
        )
    frontier_records_exact = bool(
        baseline.records == candidate.records
        and baseline.cutoff_score == candidate.cutoff_score
        and baseline.relative_cutoff_gap == candidate.relative_cutoff_gap
        and baseline.cutoff_tie_count == candidate.cutoff_tie_count
        and baseline.near_cutoff_count == candidate.near_cutoff_count
    )
    decoder_evidence_exact = bool(
        baseline_vectors.keys() == candidate_vectors.keys()
        and all(
            np.array_equal(baseline_vectors[key], candidate_vectors[key])
            for key in baseline_vectors
        )
    )
    frontier_exact = frontier_records_exact and decoder_evidence_exact
    cosines = [float(row["decoder_cosine"]) for row in pairs]
    metrics: dict[str, bool | int | float | str | None] = {
        "frontier_evidence_exact": frontier_exact,
        "frontier_records_exact": frontier_records_exact,
        "decoder_evidence_exact": decoder_evidence_exact,
        "exact_shared_selected_count": len(
            baseline_selected.keys() & candidate_selected.keys()
        ),
        "exact_exited_selected_count": len(baseline_selected.keys() - candidate_selected.keys()),
        "exact_entered_selected_count": len(candidate_selected.keys() - baseline_selected.keys()),
        "raw_decoder_pair_limit": _RAW_FRONTIER_PAIR_LIMIT,
        "raw_decoder_pair_count": len(pairs),
        "raw_decoder_cosine_max": max(cosines) if cosines else None,
        "raw_decoder_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "decoder_source_fingerprint": baseline_identity[0],
        "projection_fingerprint": baseline_identity[1],
    }
    return metrics, {"metrics": metrics, "raw_one_to_one_pairs": pairs}, frontier_exact


def _descriptor_vectors(
    descriptor: Mapping[str, Any],
) -> tuple[dict[FeatureKey, np.ndarray], tuple[str, str]]:
    features = np.asarray(descriptor["candidate_features"], dtype=np.int64)
    sketch = np.asarray(descriptor["semantic_sketch"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 3 or sketch.ndim != 2:
        raise ValueError("decoder descriptor arrays have invalid shapes")
    if features.shape[0] != sketch.shape[0] or not np.isfinite(sketch).all():
        raise ValueError("decoder descriptor arrays do not align")
    vectors = {
        FeatureKey(*(int(value) for value in feature)): sketch[index]
        for index, feature in enumerate(features)
    }
    if len(vectors) != len(features):
        raise ValueError("decoder descriptor feature identities are not unique")
    return vectors, (
        _scalar_text(descriptor.get("decoder_source_fingerprint"), "decoder_source_fingerprint"),
        _scalar_text(descriptor.get("projection_fingerprint"), "projection_fingerprint"),
    )


def _bounded_churn_records(
    records: tuple[FrontierRecord, ...],
) -> tuple[FrontierRecord, ...]:
    return tuple(
        sorted(records, key=lambda record: (-abs(record.influence), record.key))[
            :_RAW_FRONTIER_PAIR_LIMIT
        ]
    )


def _dense_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("decoder descriptor dimensions differ")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _sparse_cosine(
    left: Mapping[object, float], right: Mapping[object, float]
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    left_values = np.asarray([left.get(key, 0.0) for key in keys], dtype=np.float64)
    right_values = np.asarray([right.get(key, 0.0) for key in keys], dtype=np.float64)
    return _dense_cosine(left_values, right_values)


def _feature_key_json(key: FeatureKey) -> dict[str, int]:
    return {"layer": key.layer, "position": key.position, "feature_id": key.feature_id}


def _scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must be scalar")
    result = str(array.item())
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _comparison_is_exact(comparison: Mapping[str, Any]) -> bool:
    typed = comparison.get("typed_bucket_comparison")
    return bool(
        isinstance(typed, Mapping)
        and typed.get("classification") == "strict_exact"
        and comparison.get("feature_jaccard") == 1.0
        and comparison.get("target_token_match") == 1.0
    )


def _comparison_scope_compatibility(
    *,
    reference_path: Path,
    candidate_path: Path,
    comparison: Mapping[str, Any],
) -> tuple[bool, dict[str, bool]]:
    """Require the same scientific object while allowing execution-form drift."""

    reference = load_typed_compact_graph(reference_path)
    candidate = load_typed_compact_graph(candidate_path)
    fingerprints = comparison.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        fingerprints = {}
    compatibility = {
        "retention_policy_compatible": bool(comparison.get("policy_compatible")),
        "provider_fingerprint_match": bool(
            fingerprints.get("provider_fingerprint_match")
        ),
        "target_fingerprint_match": bool(
            fingerprints.get("target_fingerprint_match")
        ),
        "step_fingerprint_match": bool(fingerprints.get("step_fingerprint_match")),
        "ordered_feature_ids_match": bool(
            np.array_equal(reference.feature_ids, candidate.feature_ids)
        ),
        "step_index_match": int(reference.step_idx) == int(candidate.step_idx),
        "schema_version_match": int(reference.schema_version)
        == int(candidate.schema_version),
        "compact_save_format_match": str(reference.compact_save_format)
        == str(candidate.compact_save_format),
    }
    return all(compatibility.values()), compatibility


def _classify_typed_graph_comparison(
    comparison: Mapping[str, Any],
    *,
    scope_compatible: bool,
    exact: bool,
    calibration: FrozenCorrectnessCalibration | None,
) -> tuple[NumericalStabilityStatus, tuple[str, ...]]:
    if exact:
        return NumericalStabilityStatus.EXACT_STABLE, ()
    if calibration is None:
        reason = (
            "comparison_scope_incompatible"
            if not scope_compatible
            else "frozen_numerical_policy_absent"
        )
        return NumericalStabilityStatus.UNKNOWN, (reason,)
    if not scope_compatible:
        return NumericalStabilityStatus.DIVERGENT, (
            "comparison_scope_incompatible",
        )

    typed = comparison.get("typed_bucket_comparison")
    buckets = typed.get("buckets") if isinstance(typed, Mapping) else None
    if not isinstance(buckets, Mapping):
        return NumericalStabilityStatus.UNKNOWN, (
            "typed_bucket_metrics_unavailable",
        )

    exact_bucket_failures = tuple(
        name
        for name in calibration.numerical.exact_buckets
        if not (
            isinstance(buckets.get(name), Mapping)
            and buckets[name].get("exact") is True
        )
    )
    if exact_bucket_failures:
        return NumericalStabilityStatus.DIVERGENT, (
            "required_exact_bucket_drift",
        )

    pass_results: list[bool] = []
    review_results: list[bool] = []
    calibrated_values: list[object] = [comparison.get("feature_jaccard")]
    for bucket_name in calibration.numerical.buckets:
        raw_bucket = buckets.get(bucket_name)
        if isinstance(raw_bucket, Mapping) and raw_bucket.get("exact") is not True:
            calibrated_values.extend(
                raw_bucket.get(name)
                for name in (
                    "normalized_l1_deviation",
                    "weighted_jaccard",
                    "shared_sign_agreement",
                )
            )
    if any(_metric_is_nonfinite(value) for value in calibrated_values):
        return NumericalStabilityStatus.DIVERGENT, (
            "nonfinite_numerical_metric",
        )
    feature_jaccard = _finite_metric(comparison.get("feature_jaccard"))
    if feature_jaccard is None:
        return NumericalStabilityStatus.UNKNOWN, (
            "feature_jaccard_unavailable",
        )
    feature_band = calibration.numerical.feature_jaccard
    pass_results.append(feature_jaccard >= feature_band.pass_min)
    review_results.append(feature_jaccard >= feature_band.review_min)

    for bucket_name, bucket_policy in calibration.numerical.buckets.items():
        raw_bucket = buckets.get(bucket_name)
        if not isinstance(raw_bucket, Mapping):
            return NumericalStabilityStatus.UNKNOWN, (
                "calibrated_bucket_metrics_unavailable",
            )
        if raw_bucket.get("exact") is True:
            # Exact empty buckets can have undefined overlap/sign metrics while
            # still providing stronger evidence than every calibrated bound.
            pass_results.extend((True, True, True))
            review_results.extend((True, True, True))
            continue
        normalized_l1 = _finite_metric(raw_bucket.get("normalized_l1_deviation"))
        weighted_jaccard = _finite_metric(raw_bucket.get("weighted_jaccard"))
        shared_sign = _finite_metric(raw_bucket.get("shared_sign_agreement"))
        if None in (normalized_l1, weighted_jaccard, shared_sign):
            return NumericalStabilityStatus.UNKNOWN, (
                "calibrated_bucket_metrics_unavailable",
            )
        assert normalized_l1 is not None
        assert weighted_jaccard is not None
        assert shared_sign is not None
        pass_results.extend(
            (
                normalized_l1
                <= bucket_policy.normalized_l1_deviation.pass_max,
                weighted_jaccard >= bucket_policy.weighted_jaccard.pass_min,
                shared_sign >= bucket_policy.shared_sign_agreement.pass_min,
            )
        )
        review_results.extend(
            (
                normalized_l1
                <= bucket_policy.normalized_l1_deviation.review_max,
                weighted_jaccard >= bucket_policy.weighted_jaccard.review_min,
                shared_sign >= bucket_policy.shared_sign_agreement.review_min,
            )
        )

    if all(pass_results):
        return NumericalStabilityStatus.BOUNDED, ()
    if all(review_results):
        return NumericalStabilityStatus.REVIEW, (
            "numerical_metrics_in_review_band",
        )
    return NumericalStabilityStatus.DIVERGENT, (
        "numerical_metrics_exceed_review_bounds",
    )


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _metric_is_nonfinite(value: object) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and not np.isfinite(float(value))
    )


def _annotate_calibration_metrics(
    metrics: dict[str, bool | int | float | str | None],
    *,
    calibration: FrozenCorrectnessCalibration | None,
) -> None:
    if calibration is None:
        return
    metrics.update(
        {
            "classification_policy_id": calibration.policy_id,
            "calibration_fingerprint": calibration.calibration_fingerprint,
        }
    )


def _scalar_comparison_metrics(
    comparison: Mapping[str, Any],
    *,
    role: str,
    reference: Mapping[str, Any],
    exact: bool,
) -> dict[str, bool | int | float | str | None]:
    typed = comparison["typed_bucket_comparison"]
    assert isinstance(typed, Mapping)
    non_exact = typed.get("non_exact_buckets", [])
    metrics: dict[str, bool | int | float | str | None] = {
        "reference_count": 1,
        "repeat_count": 1 if role == "repeat" else 0,
        "reference_role": role,
        "reference_id": str(reference["reference_id"]),
        "reference_artifact_sha256": str(reference["graph_sha256"]),
        "comparison_exact": exact,
        "policy_compatible": bool(comparison["policy_compatible"]),
        "feature_jaccard": comparison.get("feature_jaccard"),
        "target_token_match": comparison.get("target_token_match"),
        "non_exact_bucket_count": len(non_exact) if isinstance(non_exact, list) else 0,
    }
    for key, value in comparison.items():
        if key.startswith("bucket_") and isinstance(value, (bool, int, float, str)):
            metrics[key] = value
    return metrics


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a sha256 fingerprint")
