"""Deterministic, one-to-one stability comparison at a feature frontier."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contracts import (
    NumericalScopeReport,
    NumericalStabilityStatus,
)
from .frontier import FeatureKey, FrontierEvidence, FrontierRecord


class DecoderVectorSource(Protocol):
    """Narrow provider seam used only to generate alias candidates."""

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray:
        """Return one finite, one-dimensional decoder direction."""


@dataclass(frozen=True)
class AliasMatchPolicy:
    min_decoder_cosine: float
    min_activation_scaled_cosine: float
    min_feature_neighborhood_cosine: float
    min_logit_neighborhood_cosine: float
    signed_effect_atol: float
    signed_effect_rtol: float
    high_influence_mass: float
    high_edge_mass: float
    require_same_position: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_decoder_cosine",
            "min_activation_scaled_cosine",
            "min_feature_neighborhood_cosine",
            "min_logit_neighborhood_cosine",
        ):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in (
            "signed_effect_atol",
            "signed_effect_rtol",
            "high_influence_mass",
            "high_edge_mass",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class FrontierMatch:
    baseline: FeatureKey
    candidate: FeatureKey
    exact_identity: bool
    decoder_cosine: float
    activation_scaled_cosine: float
    signed_effect_delta: float
    feature_neighborhood_cosine: float
    logit_neighborhood_cosine: float


@dataclass(frozen=True)
class RecoveryMeasure:
    baseline_total: float
    candidate_total: float
    matched_baseline: float
    matched_candidate: float

    @property
    def baseline_fraction(self) -> float | None:
        return _fraction(self.matched_baseline, self.baseline_total)

    @property
    def candidate_fraction(self) -> float | None:
        return _fraction(self.matched_candidate, self.candidate_total)


@dataclass(frozen=True)
class HighMassChurn:
    side: str
    feature: FeatureKey
    influence_mass: float
    edge_mass: float
    exceeds_influence_threshold: bool
    exceeds_edge_threshold: bool


@dataclass(frozen=True)
class FrontierStabilityReport:
    baseline_graph_fingerprint: str
    candidate_graph_fingerprint: str
    provider_fingerprint: str
    decoder_fingerprint: str
    baseline_cutoff_score: float
    candidate_cutoff_score: float
    baseline_relative_cutoff_gap: float
    candidate_relative_cutoff_gap: float
    baseline_cutoff_tie_count: int
    candidate_cutoff_tie_count: int
    baseline_near_cutoff_count: int
    candidate_near_cutoff_count: int
    frontier_evidence_exact: bool
    numeric_deltas: FrontierNumericDeltas
    exact_unchanged_selected: tuple[FeatureKey, ...]
    exact_exited_selected: tuple[FeatureKey, ...]
    exact_entered_selected: tuple[FeatureKey, ...]
    matches: tuple[FrontierMatch, ...]
    selected: RecoveryMeasure
    influence: RecoveryMeasure
    edge_mass: RecoveryMeasure
    unmatched_high_mass_churn: tuple[HighMassChurn, ...]


@dataclass(frozen=True)
class FrontierNumericDeltas:
    structurally_comparable: bool
    cutoff_score_abs_delta: float
    relative_cutoff_gap_abs_delta: float
    cutoff_tie_count_abs_delta: int
    max_record_scalar_abs_delta: float | None
    max_typed_edge_weight_abs_delta: float | None


@dataclass(frozen=True)
class BoundedFrontierDeltaPolicy:
    """Frozen numeric bounds for same-identity, non-exact frontier evidence."""

    max_cutoff_score_abs_delta: float
    max_relative_cutoff_gap_abs_delta: float
    max_cutoff_tie_count_abs_delta: int
    max_record_scalar_abs_delta: float
    max_typed_edge_weight_abs_delta: float

    def __post_init__(self) -> None:
        for name in (
            "max_cutoff_score_abs_delta",
            "max_relative_cutoff_gap_abs_delta",
            "max_record_scalar_abs_delta",
            "max_typed_edge_weight_abs_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_cutoff_tie_count_abs_delta < 0:
            raise ValueError("max_cutoff_tie_count_abs_delta must be non-negative")


@dataclass(frozen=True)
class _AdmittedCandidate:
    baseline: FrontierRecord
    candidate: FrontierRecord
    decoder_cosine: float
    activation_scaled_cosine: float
    signed_effect_delta: float
    feature_neighborhood_cosine: float
    logit_neighborhood_cosine: float


@dataclass(frozen=True)
class FrozenFrontierClassificationPolicy:
    """Versioned thresholds fitted outside the comparison implementation."""

    policy_id: str
    calibration_fingerprint: str
    min_baseline_selected_recovery: float
    min_candidate_selected_recovery: float
    min_baseline_influence_recovery: float
    min_candidate_influence_recovery: float
    min_baseline_edge_mass_recovery: float
    min_candidate_edge_mass_recovery: float
    max_unmatched_high_mass_churn: int
    bounded_deltas: BoundedFrontierDeltaPolicy | None = None

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("classification policy_id is required")
        _require_sha256_fingerprint(
            "calibration_fingerprint", self.calibration_fingerprint
        )
        for name in (
            "min_baseline_selected_recovery",
            "min_candidate_selected_recovery",
            "min_baseline_influence_recovery",
            "min_candidate_influence_recovery",
            "min_baseline_edge_mass_recovery",
            "min_candidate_edge_mass_recovery",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_unmatched_high_mass_churn < 0:
            raise ValueError("max_unmatched_high_mass_churn must be non-negative")


def compare_frontiers(
    baseline: FrontierEvidence,
    candidate: FrontierEvidence,
    *,
    decoder_source: DecoderVectorSource,
    policy: AliasMatchPolicy,
) -> FrontierStabilityReport:
    """Compare two bounded frontiers without hiding exact selection churn.

    Exact identities are paired first.  Remaining alias candidates must have
    compatible layer/position locality, pass the decoder-cosine generator, and
    then independently pass direction, signed-effect, and both typed
    neighborhood gates.  A canonical greedy order makes the admitted matching
    deterministic and one-to-one; no record can recover more than one peer.
    """

    if baseline.provider_fingerprint != candidate.provider_fingerprint:
        raise ValueError("frontier provider fingerprints must match")
    if baseline.decoder_fingerprint != candidate.decoder_fingerprint:
        raise ValueError("frontier decoder fingerprints must match")

    baseline_by_key = _records_by_key(baseline)
    candidate_by_key = _records_by_key(candidate)
    numeric_deltas = _frontier_numeric_deltas(
        baseline,
        candidate,
        baseline_by_key=baseline_by_key,
        candidate_by_key=candidate_by_key,
    )
    baseline_selected = frozenset(record.key for record in baseline.selected)
    candidate_selected = frozenset(record.key for record in candidate.selected)

    exact_unchanged = tuple(sorted(baseline_selected & candidate_selected))
    exact_exited = tuple(sorted(baseline_selected - candidate_selected))
    exact_entered = tuple(sorted(candidate_selected - baseline_selected))

    shared_keys = sorted(
        key
        for key in baseline_by_key.keys() & candidate_by_key.keys()
        if baseline_by_key[key].selected == candidate_by_key[key].selected
    )
    matched_baseline = set(shared_keys)
    matched_candidate = set(shared_keys)
    matches = [
        _exact_match(
            baseline_by_key[key],
            candidate_by_key[key],
            decoder_source=decoder_source,
        )
        for key in shared_keys
    ]

    admitted: list[_AdmittedCandidate] = []
    for baseline_record in baseline.records:
        if baseline_record.key in matched_baseline:
            continue
        for candidate_record in candidate.records:
            if candidate_record.key in matched_candidate:
                continue
            if baseline_record.key == candidate_record.key:
                # A selection-boundary flip is exact churn, not an alias.  Keep
                # both records available for a possible selected-to-selected
                # cross-identity recovery instead of consuming them here.
                continue
            pair = _admit_alias_candidate(
                baseline_record,
                candidate_record,
                decoder_source=decoder_source,
                policy=policy,
            )
            if pair is not None:
                admitted.append(pair)

    for pair in sorted(admitted, key=_candidate_order):
        if pair.baseline.key in matched_baseline:
            continue
        if pair.candidate.key in matched_candidate:
            continue
        matched_baseline.add(pair.baseline.key)
        matched_candidate.add(pair.candidate.key)
        matches.append(_to_match(pair))

    matches.sort(key=lambda match: (match.baseline, match.candidate))
    matched_selected_pairs = tuple(
        match
        for match in matches
        if baseline_by_key[match.baseline].selected
        and candidate_by_key[match.candidate].selected
    )
    recovered_baseline = tuple(
        baseline_by_key[match.baseline] for match in matched_selected_pairs
    )
    recovered_candidate = tuple(
        candidate_by_key[match.candidate] for match in matched_selected_pairs
    )

    unmatched_baseline_selected = tuple(
        record
        for record in baseline.selected
        if record.key not in {item.key for item in recovered_baseline}
    )
    unmatched_candidate_selected = tuple(
        record
        for record in candidate.selected
        if record.key not in {item.key for item in recovered_candidate}
    )

    return FrontierStabilityReport(
        baseline_graph_fingerprint=baseline.graph_fingerprint,
        candidate_graph_fingerprint=candidate.graph_fingerprint,
        provider_fingerprint=baseline.provider_fingerprint,
        decoder_fingerprint=baseline.decoder_fingerprint,
        baseline_cutoff_score=baseline.cutoff_score,
        candidate_cutoff_score=candidate.cutoff_score,
        baseline_relative_cutoff_gap=baseline.relative_cutoff_gap,
        candidate_relative_cutoff_gap=candidate.relative_cutoff_gap,
        baseline_cutoff_tie_count=baseline.cutoff_tie_count,
        candidate_cutoff_tie_count=candidate.cutoff_tie_count,
        baseline_near_cutoff_count=baseline.near_cutoff_count,
        candidate_near_cutoff_count=candidate.near_cutoff_count,
        frontier_evidence_exact=(
            baseline.records == candidate.records
            and baseline.cutoff_score == candidate.cutoff_score
            and baseline.relative_cutoff_gap == candidate.relative_cutoff_gap
            and baseline.cutoff_tie_count == candidate.cutoff_tie_count
            and baseline.near_cutoff_count == candidate.near_cutoff_count
        ),
        numeric_deltas=numeric_deltas,
        exact_unchanged_selected=exact_unchanged,
        exact_exited_selected=exact_exited,
        exact_entered_selected=exact_entered,
        matches=tuple(matches),
        selected=RecoveryMeasure(
            baseline_total=float(len(baseline.selected)),
            candidate_total=float(len(candidate.selected)),
            matched_baseline=float(len(recovered_baseline)),
            matched_candidate=float(len(recovered_candidate)),
        ),
        influence=_recovery_measure(
            baseline.selected,
            candidate.selected,
            recovered_baseline,
            recovered_candidate,
            value=lambda record: abs(record.influence),
        ),
        edge_mass=_recovery_measure(
            baseline.selected,
            candidate.selected,
            recovered_baseline,
            recovered_candidate,
            value=lambda record: record.edge_mass,
        ),
        unmatched_high_mass_churn=_unmatched_high_mass_churn(
            unmatched_baseline_selected,
            unmatched_candidate_selected,
            policy,
        ),
    )


def classify_frontier_stability(
    report: FrontierStabilityReport,
    *,
    frozen_policy: FrozenFrontierClassificationPolicy | None,
    exact_graph_or_scope_equal: bool | None = None,
) -> NumericalScopeReport:
    """Classify frontier evidence without inventing calibration thresholds."""

    if frozen_policy is None:
        return NumericalScopeReport(
            scope_id="feature_frontier",
            status=NumericalStabilityStatus.UNKNOWN,
            metrics={
                **_classification_metrics(report),
                "exact_graph_or_scope_equal": exact_graph_or_scope_equal,
            },
            reason_codes=("missing_frozen_frontier_calibration",),
        )

    metrics = {
        **_classification_metrics(report),
        "classification_policy_id": frozen_policy.policy_id,
        "calibration_fingerprint": frozen_policy.calibration_fingerprint,
        "exact_graph_or_scope_equal": exact_graph_or_scope_equal,
    }
    if report.frontier_evidence_exact and exact_graph_or_scope_equal is True:
        return NumericalScopeReport(
            scope_id="feature_frontier",
            status=NumericalStabilityStatus.EXACT_STABLE,
            metrics=metrics,
        )

    has_selected_identity_churn = bool(
        report.exact_exited_selected or report.exact_entered_selected
    )
    if not has_selected_identity_churn:
        if frozen_policy.bounded_deltas is None:
            return NumericalScopeReport(
                scope_id="feature_frontier",
                status=NumericalStabilityStatus.UNKNOWN,
                metrics=metrics,
                reason_codes=("bounded_frontier_thresholds_missing",),
            )
        if _within_bounded_delta_policy(
            report.numeric_deltas,
            frozen_policy.bounded_deltas,
        ):
            return NumericalScopeReport(
                scope_id="feature_frontier",
                status=NumericalStabilityStatus.BOUNDED,
                metrics=metrics,
            )
        return NumericalScopeReport(
            scope_id="feature_frontier",
            status=NumericalStabilityStatus.DIVERGENT,
            metrics=metrics,
            reason_codes=("frontier_numeric_deltas_exceed_frozen_bounds",),
        )

    threshold_results = (
        _meets_recovery(
            report.selected.baseline_fraction,
            frozen_policy.min_baseline_selected_recovery,
        ),
        _meets_recovery(
            report.selected.candidate_fraction,
            frozen_policy.min_candidate_selected_recovery,
        ),
        _meets_recovery(
            report.influence.baseline_fraction,
            frozen_policy.min_baseline_influence_recovery,
        ),
        _meets_recovery(
            report.influence.candidate_fraction,
            frozen_policy.min_candidate_influence_recovery,
        ),
        _meets_recovery(
            report.edge_mass.baseline_fraction,
            frozen_policy.min_baseline_edge_mass_recovery,
        ),
        _meets_recovery(
            report.edge_mass.candidate_fraction,
            frozen_policy.min_candidate_edge_mass_recovery,
        ),
    )
    churn_within_limit = (
        len(report.unmatched_high_mass_churn)
        <= frozen_policy.max_unmatched_high_mass_churn
    )
    if all(threshold_results) and churn_within_limit:
        return NumericalScopeReport(
            scope_id="feature_frontier",
            status=NumericalStabilityStatus.ALIAS_STABLE,
            metrics=metrics,
        )

    reasons: list[str] = []
    if not all(threshold_results):
        reasons.append("frontier_recovery_below_frozen_threshold")
    if not churn_within_limit:
        reasons.append("unmatched_high_mass_churn_exceeds_frozen_limit")
    return NumericalScopeReport(
        scope_id="feature_frontier",
        status=NumericalStabilityStatus.DIVERGENT,
        metrics=metrics,
        reason_codes=tuple(reasons),
    )


def _classification_metrics(
    report: FrontierStabilityReport,
) -> dict[str, bool | int | float | str | None]:
    return {
        "provider_fingerprint": report.provider_fingerprint,
        "decoder_fingerprint": report.decoder_fingerprint,
        "baseline_cutoff_score": report.baseline_cutoff_score,
        "candidate_cutoff_score": report.candidate_cutoff_score,
        "baseline_relative_cutoff_gap": report.baseline_relative_cutoff_gap,
        "candidate_relative_cutoff_gap": report.candidate_relative_cutoff_gap,
        "baseline_cutoff_tie_count": report.baseline_cutoff_tie_count,
        "candidate_cutoff_tie_count": report.candidate_cutoff_tie_count,
        "baseline_near_cutoff_count": report.baseline_near_cutoff_count,
        "candidate_near_cutoff_count": report.candidate_near_cutoff_count,
        "exact_exited_selected_count": len(report.exact_exited_selected),
        "exact_entered_selected_count": len(report.exact_entered_selected),
        "selected_baseline_recovery": report.selected.baseline_fraction,
        "selected_candidate_recovery": report.selected.candidate_fraction,
        "influence_baseline_recovery": report.influence.baseline_fraction,
        "influence_candidate_recovery": report.influence.candidate_fraction,
        "edge_mass_baseline_recovery": report.edge_mass.baseline_fraction,
        "edge_mass_candidate_recovery": report.edge_mass.candidate_fraction,
        "unmatched_high_mass_churn_count": len(report.unmatched_high_mass_churn),
        "frontier_evidence_exact": report.frontier_evidence_exact,
        "frontier_structurally_comparable": (
            report.numeric_deltas.structurally_comparable
        ),
        "cutoff_score_abs_delta": report.numeric_deltas.cutoff_score_abs_delta,
        "relative_cutoff_gap_abs_delta": (
            report.numeric_deltas.relative_cutoff_gap_abs_delta
        ),
        "cutoff_tie_count_abs_delta": (
            report.numeric_deltas.cutoff_tie_count_abs_delta
        ),
        "max_record_scalar_abs_delta": (
            report.numeric_deltas.max_record_scalar_abs_delta
        ),
        "max_typed_edge_weight_abs_delta": (
            report.numeric_deltas.max_typed_edge_weight_abs_delta
        ),
    }


def _meets_recovery(observed: float | None, minimum: float) -> bool:
    # A zero floor explicitly disables that recovery dimension. Empty evidence
    # then has no mass to recover and must not turn an otherwise qualified
    # comparison into an unknown/failure.
    return minimum == 0.0 if observed is None else observed >= minimum


def _require_sha256_fingerprint(label: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256 fingerprint") from exc


def _frontier_numeric_deltas(
    baseline: FrontierEvidence,
    candidate: FrontierEvidence,
    *,
    baseline_by_key: Mapping[FeatureKey, FrontierRecord],
    candidate_by_key: Mapping[FeatureKey, FrontierRecord],
) -> FrontierNumericDeltas:
    structurally_comparable = _frontier_structures_match(
        baseline_by_key,
        candidate_by_key,
    )
    scalar_delta: float | None = None
    edge_delta: float | None = None
    if structurally_comparable:
        scalar_deltas: list[float] = []
        edge_deltas: list[float] = []
        for key in sorted(baseline_by_key):
            left = baseline_by_key[key]
            right = candidate_by_key[key]
            scalar_deltas.extend(
                abs(left_value - right_value)
                for left_value, right_value in (
                    (left.selection_score, right.selection_score),
                    (left.cutoff_distance, right.cutoff_distance),
                    (left.activation, right.activation),
                    (left.signed_target_effect, right.signed_target_effect),
                    (left.influence, right.influence),
                )
            )
            left_feature_edges = {
                edge.source: edge.weight for edge in left.feature_from_feature
            }
            right_feature_edges = {
                edge.source: edge.weight for edge in right.feature_from_feature
            }
            edge_deltas.extend(
                abs(left_feature_edges[source] - right_feature_edges[source])
                for source in sorted(left_feature_edges)
            )
            left_logit_edges = {
                edge.logit_token_id: edge.weight for edge in left.logit_from_feature
            }
            right_logit_edges = {
                edge.logit_token_id: edge.weight for edge in right.logit_from_feature
            }
            edge_deltas.extend(
                abs(left_logit_edges[token_id] - right_logit_edges[token_id])
                for token_id in sorted(left_logit_edges)
            )
        scalar_delta = max(scalar_deltas, default=0.0)
        edge_delta = max(edge_deltas, default=0.0)
    return FrontierNumericDeltas(
        structurally_comparable=structurally_comparable,
        cutoff_score_abs_delta=abs(baseline.cutoff_score - candidate.cutoff_score),
        relative_cutoff_gap_abs_delta=abs(
            baseline.relative_cutoff_gap - candidate.relative_cutoff_gap
        ),
        cutoff_tie_count_abs_delta=abs(
            baseline.cutoff_tie_count - candidate.cutoff_tie_count
        ),
        max_record_scalar_abs_delta=scalar_delta,
        max_typed_edge_weight_abs_delta=edge_delta,
    )


def _frontier_structures_match(
    baseline: Mapping[FeatureKey, FrontierRecord],
    candidate: Mapping[FeatureKey, FrontierRecord],
) -> bool:
    if baseline.keys() != candidate.keys():
        return False
    for key in baseline:
        left = baseline[key]
        right = candidate[key]
        if left.selected != right.selected or left.rank != right.rank:
            return False
        if {edge.source for edge in left.feature_from_feature} != {
            edge.source for edge in right.feature_from_feature
        }:
            return False
        if {edge.logit_token_id for edge in left.logit_from_feature} != {
            edge.logit_token_id for edge in right.logit_from_feature
        }:
            return False
    return True


def _within_bounded_delta_policy(
    observed: FrontierNumericDeltas,
    policy: BoundedFrontierDeltaPolicy,
) -> bool:
    if not observed.structurally_comparable:
        return False
    if observed.max_record_scalar_abs_delta is None:
        return False
    if observed.max_typed_edge_weight_abs_delta is None:
        return False
    return (
        observed.cutoff_score_abs_delta <= policy.max_cutoff_score_abs_delta
        and observed.relative_cutoff_gap_abs_delta
        <= policy.max_relative_cutoff_gap_abs_delta
        and observed.cutoff_tie_count_abs_delta
        <= policy.max_cutoff_tie_count_abs_delta
        and observed.max_record_scalar_abs_delta
        <= policy.max_record_scalar_abs_delta
        and observed.max_typed_edge_weight_abs_delta
        <= policy.max_typed_edge_weight_abs_delta
    )


def _admit_alias_candidate(
    baseline: FrontierRecord,
    candidate: FrontierRecord,
    *,
    decoder_source: DecoderVectorSource,
    policy: AliasMatchPolicy,
) -> _AdmittedCandidate | None:
    if baseline.key.layer != candidate.key.layer:
        return None
    if policy.require_same_position and baseline.key.position != candidate.key.position:
        return None

    baseline_decoder = _decoder_vector(decoder_source, baseline.key)
    candidate_decoder = _decoder_vector(decoder_source, candidate.key)
    decoder_cosine = _dense_cosine(baseline_decoder, candidate_decoder)
    if decoder_cosine < policy.min_decoder_cosine:
        return None

    scaled_cosine = _dense_cosine(
        baseline.activation * baseline_decoder,
        candidate.activation * candidate_decoder,
    )
    if scaled_cosine < policy.min_activation_scaled_cosine:
        return None

    signed_effect_delta = abs(
        baseline.signed_target_effect - candidate.signed_target_effect
    )
    if not _signed_effect_compatible(
        baseline.signed_target_effect,
        candidate.signed_target_effect,
        atol=policy.signed_effect_atol,
        rtol=policy.signed_effect_rtol,
    ):
        return None

    feature_cosine = _sparse_cosine(
        {edge.source: edge.weight for edge in baseline.feature_from_feature},
        {edge.source: edge.weight for edge in candidate.feature_from_feature},
    )
    if feature_cosine < policy.min_feature_neighborhood_cosine:
        return None
    logit_cosine = _sparse_cosine(
        {edge.logit_token_id: edge.weight for edge in baseline.logit_from_feature},
        {edge.logit_token_id: edge.weight for edge in candidate.logit_from_feature},
    )
    if logit_cosine < policy.min_logit_neighborhood_cosine:
        return None

    return _AdmittedCandidate(
        baseline=baseline,
        candidate=candidate,
        decoder_cosine=decoder_cosine,
        activation_scaled_cosine=scaled_cosine,
        signed_effect_delta=signed_effect_delta,
        feature_neighborhood_cosine=feature_cosine,
        logit_neighborhood_cosine=logit_cosine,
    )


def _candidate_order(
    pair: _AdmittedCandidate,
) -> tuple[int, int, float, float, float, float, float, FeatureKey, FeatureKey]:
    selection_priority = -int(pair.baseline.selected and pair.candidate.selected)
    boundary_priority = -int(pair.baseline.selected or pair.candidate.selected)
    return (
        selection_priority,
        boundary_priority,
        -pair.decoder_cosine,
        -pair.activation_scaled_cosine,
        pair.signed_effect_delta,
        -pair.feature_neighborhood_cosine,
        -pair.logit_neighborhood_cosine,
        pair.baseline.key,
        pair.candidate.key,
    )


def _exact_match(
    baseline: FrontierRecord,
    candidate: FrontierRecord,
    *,
    decoder_source: DecoderVectorSource,
) -> FrontierMatch:
    del decoder_source
    return FrontierMatch(
        baseline=baseline.key,
        candidate=candidate.key,
        exact_identity=True,
        decoder_cosine=1.0,
        activation_scaled_cosine=_scalar_direction_cosine(
            baseline.activation,
            candidate.activation,
        ),
        signed_effect_delta=abs(
            baseline.signed_target_effect - candidate.signed_target_effect
        ),
        feature_neighborhood_cosine=_sparse_cosine(
            {edge.source: edge.weight for edge in baseline.feature_from_feature},
            {edge.source: edge.weight for edge in candidate.feature_from_feature},
        ),
        logit_neighborhood_cosine=_sparse_cosine(
            {edge.logit_token_id: edge.weight for edge in baseline.logit_from_feature},
            {edge.logit_token_id: edge.weight for edge in candidate.logit_from_feature},
        ),
    )


def _to_match(pair: _AdmittedCandidate) -> FrontierMatch:
    return FrontierMatch(
        baseline=pair.baseline.key,
        candidate=pair.candidate.key,
        exact_identity=False,
        decoder_cosine=pair.decoder_cosine,
        activation_scaled_cosine=pair.activation_scaled_cosine,
        signed_effect_delta=pair.signed_effect_delta,
        feature_neighborhood_cosine=pair.feature_neighborhood_cosine,
        logit_neighborhood_cosine=pair.logit_neighborhood_cosine,
    )


def _records_by_key(evidence: FrontierEvidence) -> dict[FeatureKey, FrontierRecord]:
    return {record.key: record for record in evidence.records}


def _decoder_vector(source: DecoderVectorSource, feature: FeatureKey) -> np.ndarray:
    vector = np.asarray(source.decoder_vector(feature), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"decoder vector for {feature!r} must be non-empty and 1D")
    if not np.isfinite(vector).all():
        raise ValueError(f"decoder vector for {feature!r} must be finite")
    return vector


def _dense_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("decoder vector shapes must match")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _sparse_cosine(
    left: Mapping[object, float],
    right: Mapping[object, float],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    shared = left.keys() & right.keys()
    dot = math.fsum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(math.fsum(value * value for value in left.values()))
    right_norm = math.sqrt(math.fsum(value * value for value in right.values()))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _signed_effect_compatible(
    left: float,
    right: float,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if math.isclose(left, 0.0, abs_tol=atol) and math.isclose(
        right, 0.0, abs_tol=atol
    ):
        return True
    if left * right <= 0.0:
        return False
    return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)


def _recovery_measure(
    baseline: tuple[FrontierRecord, ...],
    candidate: tuple[FrontierRecord, ...],
    recovered_baseline: tuple[FrontierRecord, ...],
    recovered_candidate: tuple[FrontierRecord, ...],
    *,
    value: Callable[[FrontierRecord], float],
) -> RecoveryMeasure:
    # The callable is kept internal so the public interface remains expressed
    # in scientific measures rather than a generic metric hook.
    return RecoveryMeasure(
        baseline_total=math.fsum(value(record) for record in baseline),
        candidate_total=math.fsum(value(record) for record in candidate),
        matched_baseline=math.fsum(value(record) for record in recovered_baseline),
        matched_candidate=math.fsum(value(record) for record in recovered_candidate),
    )


def _high_mass_churn(
    side: str,
    record: FrontierRecord,
    policy: AliasMatchPolicy,
) -> HighMassChurn:
    influence_mass = abs(record.influence)
    edge_mass = record.edge_mass
    return HighMassChurn(
        side=side,
        feature=record.key,
        influence_mass=influence_mass,
        edge_mass=edge_mass,
        exceeds_influence_threshold=influence_mass >= policy.high_influence_mass,
        exceeds_edge_threshold=edge_mass >= policy.high_edge_mass,
    )


def _unmatched_high_mass_churn(
    baseline: tuple[FrontierRecord, ...],
    candidate: tuple[FrontierRecord, ...],
    policy: AliasMatchPolicy,
) -> tuple[HighMassChurn, ...]:
    churn = (
        *(_high_mass_churn("exited", record, policy) for record in baseline),
        *(_high_mass_churn("entered", record, policy) for record in candidate),
    )
    return tuple(
        sorted(
            (
                item
                for item in churn
                if item.exceeds_influence_threshold or item.exceeds_edge_threshold
            ),
            key=lambda item: (item.side, item.feature),
        )
    )


def _scalar_direction_cosine(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 1.0
    if left == 0.0 or right == 0.0:
        return 0.0
    return 1.0 if left * right > 0.0 else -1.0


def _fraction(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator
