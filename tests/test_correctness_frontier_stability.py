from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.correctness.contracts import (
    NumericalStabilityStatus,
)
from nlp_research_project.exact_trace_bench.correctness.frontier import (
    FeatureFeatureEdge,
    FeatureKey,
    FrontierBounds,
    FrontierRecord,
    LogitFeatureEdge,
    build_frontier_evidence,
)
from nlp_research_project.exact_trace_bench.correctness.stability import (
    AliasMatchPolicy,
    BoundedFrontierDeltaPolicy,
    FrozenFrontierClassificationPolicy,
    classify_frontier_stability,
    compare_frontiers,
)

COMMON_SOURCE = FeatureKey(0, 0, 99)


class FakeDecoderSource:
    def __init__(self, vectors: dict[FeatureKey, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray:
        return np.asarray(self._vectors[feature], dtype=np.float64)


def _record(
    feature_id: int,
    *,
    selected: bool,
    rank: int,
    activation: float = 2.0,
    effect: float = 3.0,
    influence: float = 4.0,
    feature_weight: float = 2.0,
    logit_weight: float = 3.0,
) -> FrontierRecord:
    return FrontierRecord(
        key=FeatureKey(1, 2, feature_id),
        selected=selected,
        rank=rank,
        selection_score=10.0 - rank,
        cutoff_distance=-0.1 if selected else 0.1,
        activation=activation,
        signed_target_effect=effect,
        influence=influence,
        feature_from_feature=(FeatureFeatureEdge(COMMON_SOURCE, feature_weight),),
        logit_from_feature=(LogitFeatureEdge(7, logit_weight),),
    )


def _evidence(
    fingerprint: str,
    *,
    selected: tuple[FrontierRecord, ...],
    near: tuple[FrontierRecord, ...] = (),
):
    return build_frontier_evidence(
        graph_fingerprint=_fingerprint(fingerprint),
        provider_fingerprint=_fingerprint("provider"),
        decoder_fingerprint=_fingerprint("decoder"),
        cutoff_score=0.75,
        relative_cutoff_gap=0.01,
        cutoff_tie_count=1,
        near_cutoff_count=len(near),
        selected=selected,
        near_cutoff=near,
        bounds=FrontierBounds(max_selected=8, max_near_cutoff=8),
    )


def _policy() -> AliasMatchPolicy:
    return AliasMatchPolicy(
        min_decoder_cosine=0.9,
        min_activation_scaled_cosine=0.9,
        min_feature_neighborhood_cosine=0.9,
        min_logit_neighborhood_cosine=0.9,
        signed_effect_atol=0.01,
        signed_effect_rtol=0.05,
        high_influence_mass=3.5,
        high_edge_mass=4.5,
    )


def _classification_policy(
    *,
    bounded_deltas: BoundedFrontierDeltaPolicy | None = None,
) -> FrozenFrontierClassificationPolicy:
    return FrozenFrontierClassificationPolicy(
        policy_id="frontier_alias_calibration_v1",
        calibration_fingerprint=_fingerprint("frozen-calibration"),
        min_baseline_selected_recovery=0.9,
        min_candidate_selected_recovery=0.9,
        min_baseline_influence_recovery=0.9,
        min_candidate_influence_recovery=0.9,
        min_baseline_edge_mass_recovery=0.9,
        min_candidate_edge_mass_recovery=0.9,
        max_unmatched_high_mass_churn=0,
        bounded_deltas=bounded_deltas,
    )


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def test_frontier_is_canonical_and_fails_closed_at_bounds() -> None:
    first = _record(1, selected=True, rank=0)
    second = _record(2, selected=True, rank=1)

    evidence = build_frontier_evidence(
        graph_fingerprint=_fingerprint("graph-a"),
        provider_fingerprint=_fingerprint("provider"),
        decoder_fingerprint=_fingerprint("decoder"),
        cutoff_score=0.75,
        relative_cutoff_gap=0.01,
        cutoff_tie_count=2,
        near_cutoff_count=0,
        selected=(second, first),
        near_cutoff=(),
        bounds=FrontierBounds(max_selected=2, max_near_cutoff=0),
    )

    assert tuple(record.key for record in evidence.selected) == (
        first.key,
        second.key,
    )
    assert evidence.provider_fingerprint == _fingerprint("provider")
    assert evidence.decoder_fingerprint == _fingerprint("decoder")
    assert evidence.cutoff_score == pytest.approx(0.75)
    assert evidence.relative_cutoff_gap == pytest.approx(0.01)
    assert evidence.cutoff_tie_count == 2
    assert evidence.near_cutoff_count == 0
    with pytest.raises(ValueError, match="selected frontier exceeds"):
        build_frontier_evidence(
            graph_fingerprint=_fingerprint("graph-a"),
            provider_fingerprint=_fingerprint("provider"),
            decoder_fingerprint=_fingerprint("decoder"),
            cutoff_score=0.75,
            relative_cutoff_gap=0.01,
            cutoff_tie_count=2,
            near_cutoff_count=0,
            selected=(first, second),
            near_cutoff=(),
            bounds=FrontierBounds(max_selected=1, max_near_cutoff=0),
        )


def test_frontier_rejects_inconsistent_cutoff_evidence() -> None:
    near = _record(2, selected=False, rank=1)

    with pytest.raises(ValueError, match="near_cutoff_count must equal"):
        build_frontier_evidence(
            graph_fingerprint=_fingerprint("graph-a"),
            provider_fingerprint=_fingerprint("provider"),
            decoder_fingerprint=_fingerprint("decoder"),
            cutoff_score=0.75,
            relative_cutoff_gap=0.01,
            cutoff_tie_count=1,
            near_cutoff_count=0,
            selected=(_record(1, selected=True, rank=0),),
            near_cutoff=(near,),
            bounds=FrontierBounds(max_selected=2, max_near_cutoff=2),
        )
    with pytest.raises(ValueError, match="relative_cutoff_gap must be non-negative"):
        build_frontier_evidence(
            graph_fingerprint=_fingerprint("graph-a"),
            provider_fingerprint=_fingerprint("provider"),
            decoder_fingerprint=_fingerprint("decoder"),
            cutoff_score=0.75,
            relative_cutoff_gap=-0.01,
            cutoff_tie_count=1,
            near_cutoff_count=1,
            selected=(_record(1, selected=True, rank=0),),
            near_cutoff=(near,),
            bounds=FrontierBounds(max_selected=2, max_near_cutoff=2),
        )


def test_alias_swap_recovers_mass_without_erasing_exact_churn() -> None:
    exited = _record(10, selected=True, rank=0)
    entered_near = _record(20, selected=False, rank=1)
    entered = replace(entered_near, selected=True, rank=0, cutoff_distance=-0.1)
    exited_near = replace(exited, selected=False, rank=1, cutoff_distance=0.1)
    baseline = _evidence("baseline", selected=(exited,), near=(entered_near,))
    candidate = _evidence("candidate", selected=(entered,), near=(exited_near,))
    decoders = FakeDecoderSource(
        {
            exited.key: (1.0, 0.0),
            entered.key: (0.99, 0.01),
        }
    )

    report = compare_frontiers(
        baseline,
        candidate,
        decoder_source=decoders,
        policy=_policy(),
    )

    assert report.exact_exited_selected == (exited.key,)
    assert report.exact_entered_selected == (entered.key,)
    assert {
        (item.baseline, item.candidate) for item in report.matches
    } == {
        (exited.key, entered.key),
        (entered.key, exited.key),
    }
    assert all(item.exact_identity is False for item in report.matches)
    assert report.selected.baseline_fraction == pytest.approx(1.0)
    assert report.influence.matched_baseline == pytest.approx(4.0)
    assert report.edge_mass.matched_baseline == pytest.approx(5.0)
    assert report.unmatched_high_mass_churn == ()

    uncalibrated = classify_frontier_stability(report, frozen_policy=None)
    assert uncalibrated.status is NumericalStabilityStatus.UNKNOWN
    assert uncalibrated.reason_codes == ("missing_frozen_frontier_calibration",)

    calibrated = classify_frontier_stability(
        report,
        frozen_policy=_classification_policy(),
        exact_graph_or_scope_equal=False,
    )
    assert calibrated.status is NumericalStabilityStatus.ALIAS_STABLE
    assert calibrated.metrics["classification_policy_id"] == (
        "frontier_alias_calibration_v1"
    )
    assert calibrated.metrics["baseline_relative_cutoff_gap"] == pytest.approx(0.01)


def test_same_selected_ids_do_not_imply_exact_numeric_stability() -> None:
    baseline_record = _record(10, selected=True, rank=0)
    exact_candidate = _record(10, selected=True, rank=0)
    drifted_candidate = replace(
        exact_candidate,
        activation=exact_candidate.activation + 0.001,
        influence=exact_candidate.influence + 0.001,
        feature_from_feature=(FeatureFeatureEdge(COMMON_SOURCE, 2.001),),
    )
    baseline = _evidence("baseline", selected=(baseline_record,))
    exact = compare_frontiers(
        baseline,
        _evidence("exact-candidate", selected=(exact_candidate,)),
        decoder_source=FakeDecoderSource({}),
        policy=_policy(),
    )
    drifted = compare_frontiers(
        baseline,
        _evidence("drifted-candidate", selected=(drifted_candidate,)),
        decoder_source=FakeDecoderSource({}),
        policy=_policy(),
    )

    missing_scope_pin = classify_frontier_stability(
        exact,
        frozen_policy=_classification_policy(),
    )
    assert missing_scope_pin.status is NumericalStabilityStatus.UNKNOWN
    exact_scope = classify_frontier_stability(
        exact,
        frozen_policy=_classification_policy(),
        exact_graph_or_scope_equal=True,
    )
    assert exact_scope.status is NumericalStabilityStatus.EXACT_STABLE

    unbounded_drift = classify_frontier_stability(
        drifted,
        frozen_policy=_classification_policy(),
        exact_graph_or_scope_equal=False,
    )
    assert unbounded_drift.status is NumericalStabilityStatus.UNKNOWN
    assert unbounded_drift.reason_codes == ("bounded_frontier_thresholds_missing",)

    admitted_drift = classify_frontier_stability(
        drifted,
        frozen_policy=_classification_policy(
            bounded_deltas=BoundedFrontierDeltaPolicy(
                max_cutoff_score_abs_delta=0.0,
                max_relative_cutoff_gap_abs_delta=0.0,
                max_cutoff_tie_count_abs_delta=0,
                max_record_scalar_abs_delta=0.01,
                max_typed_edge_weight_abs_delta=0.01,
            )
        ),
        exact_graph_or_scope_equal=False,
    )
    assert admitted_drift.status is NumericalStabilityStatus.BOUNDED

    rejected_drift = classify_frontier_stability(
        drifted,
        frozen_policy=_classification_policy(
            bounded_deltas=BoundedFrontierDeltaPolicy(
                max_cutoff_score_abs_delta=0.0,
                max_relative_cutoff_gap_abs_delta=0.0,
                max_cutoff_tie_count_abs_delta=0,
                max_record_scalar_abs_delta=0.0,
                max_typed_edge_weight_abs_delta=0.0,
            )
        ),
        exact_graph_or_scope_equal=False,
    )
    assert rejected_drift.status is NumericalStabilityStatus.DIVERGENT


def test_alias_matching_is_deterministic_and_one_to_one() -> None:
    left_first = _record(10, selected=True, rank=0)
    left_second = _record(11, selected=True, rank=1)
    right_shared = _record(20, selected=True, rank=0)
    right_unmatched = _record(21, selected=True, rank=1)
    baseline = _evidence("baseline", selected=(left_second, left_first))
    candidate = _evidence("candidate", selected=(right_unmatched, right_shared))
    decoders = FakeDecoderSource(
        {
            left_first.key: (1.0, 0.0),
            left_second.key: (1.0, 0.0),
            right_shared.key: (1.0, 0.0),
            right_unmatched.key: (0.0, 1.0),
        }
    )

    first = compare_frontiers(
        baseline,
        candidate,
        decoder_source=decoders,
        policy=_policy(),
    )
    second = compare_frontiers(
        baseline,
        candidate,
        decoder_source=decoders,
        policy=_policy(),
    )

    assert first == second
    assert [(item.baseline, item.candidate) for item in first.matches] == [
        (left_first.key, right_shared.key)
    ]
    assert len({item.baseline for item in first.matches}) == len(first.matches)
    assert len({item.candidate for item in first.matches}) == len(first.matches)
    assert first.selected.matched_baseline == 1.0
    assert {item.feature for item in first.unmatched_high_mass_churn} == {
        left_second.key,
        right_unmatched.key,
    }


@pytest.mark.parametrize(
    "candidate",
    [
        _record(20, selected=True, rank=0, effect=-3.0),
        _record(20, selected=True, rank=0, feature_weight=-2.0),
        _record(20, selected=True, rank=0, logit_weight=-3.0),
        _record(20, selected=True, rank=0, activation=-2.0),
    ],
)
def test_decoder_similarity_cannot_bypass_scientific_gates(
    candidate: FrontierRecord,
) -> None:
    baseline_record = _record(10, selected=True, rank=0)
    baseline = _evidence("baseline", selected=(baseline_record,))
    candidate_evidence = _evidence("candidate", selected=(candidate,))
    decoders = FakeDecoderSource(
        {
            baseline_record.key: (1.0, 0.0),
            candidate.key: (1.0, 0.0),
        }
    )

    report = compare_frontiers(
        baseline,
        candidate_evidence,
        decoder_source=decoders,
        policy=_policy(),
    )

    assert report.matches == ()
    assert report.selected.matched_baseline == 0.0
    assert len(report.unmatched_high_mass_churn) == 2
