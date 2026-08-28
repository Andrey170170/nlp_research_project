from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.correctness.numerical import (
    _alias_match_policy,
    _combined_descriptor_source,
    _frontier_classification_policy,
    _qualified_alias_pairs_from_stability,
    _raw_frontier_comparison,
    build_reference_identity_receipt,
    evaluate_declared_numerical_stability,
    prepare_numerical_manifest_declaration,
)
from nlp_research_project.exact_trace_bench.correctness.stability import (
    classify_frontier_stability,
    compare_frontiers,
)
from nlp_research_project.exact_trace_bench.correctness.frontier import (
    FeatureKey,
    FrontierBounds,
    FrontierRecord,
    build_frontier_evidence,
)
from nlp_research_project.exact_trace_bench.correctness.contracts import (
    NumericalStabilityStatus,
)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_manifest(path: Path, *, references: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "exact_trace_numerical_reference_manifest_v1",
                "references": references,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _reference(path: Path, *, role: str) -> dict[str, object]:
    return {
        "reference_id": f"{role}-fixture",
        "role": role,
        "graph_path": str(path.resolve()),
        "graph_sha256": _sha256(path),
        "identity": build_reference_identity_receipt(path),
    }


def _record(feature_id: int, *, selected: bool, rank: int) -> FrontierRecord:
    return FrontierRecord(
        key=FeatureKey(1, 2, feature_id),
        selected=selected,
        rank=rank,
        selection_score=1.0 - rank * 0.1,
        cutoff_distance=-0.1 if selected else 0.1,
        activation=2.0,
        signed_target_effect=3.0,
        influence=4.0,
    )


def _frontier(selected_id: int, near_id: int, graph: str):
    return build_frontier_evidence(
        graph_fingerprint="sha256:" + graph * 64,
        provider_fingerprint="sha256:" + "a" * 64,
        decoder_fingerprint="sha256:" + "b" * 64,
        cutoff_score=0.8,
        relative_cutoff_gap=0.01,
        cutoff_tie_count=1,
        near_cutoff_count=1,
        selected=(_record(selected_id, selected=True, rank=0),),
        near_cutoff=(_record(near_id, selected=False, rank=1),),
        bounds=FrontierBounds(max_selected=1, max_near_cutoff=1),
    )


def _descriptor(first: int, second: int, vectors: list[list[float]]):
    return {
        "candidate_features": np.asarray(
            [[1, 2, first], [1, 2, second]], dtype=np.int64
        ),
        "semantic_sketch": np.asarray(vectors, dtype=np.float32),
        "decoder_source_fingerprint": np.asarray("sha256:" + "b" * 64),
        "projection_fingerprint": np.asarray("sha256:" + "c" * 64),
    }


def _frozen_calibration():
    lower = lambda pass_min, review_min: SimpleNamespace(  # noqa: E731
        pass_min=pass_min, review_min=review_min
    )
    upper = lambda pass_max, review_max: SimpleNamespace(  # noqa: E731
        pass_max=pass_max, review_max=review_max
    )
    bucket = lambda l1_pass, l1_review: SimpleNamespace(  # noqa: E731
        normalized_l1_deviation=upper(l1_pass, l1_review),
        weighted_jaccard=lower(0.97, 0.95),
        shared_sign_agreement=lower(0.995, 0.99),
    )
    bounded = SimpleNamespace(
        max_cutoff_score_abs_delta=1e-5,
        max_relative_cutoff_gap_abs_delta=1e-5,
        max_cutoff_tie_count_abs_delta=0,
        max_record_scalar_abs_delta=1e-5,
        max_typed_edge_weight_abs_delta=1e-5,
    )
    return SimpleNamespace(
        policy_id="correctness_calibration_v1",
        calibration_fingerprint="sha256:" + "f" * 64,
        numerical=SimpleNamespace(
            feature_jaccard=lower(0.995, 0.99),
            buckets={
                "feature<-error": bucket(0.03, 0.05),
                "logit<-error": bucket(0.01, 0.02),
            },
            exact_buckets=(
                "feature<-feature",
                "feature<-token",
                "logit<-feature",
                "logit<-token",
            ),
            frontier=SimpleNamespace(
                selected_recovery=lower(0.995, 0.99),
                unmatched_influence_mass_fraction=upper(0.005, 0.01),
                alias_min_decoder_cosine=0.95,
                bounded_deltas=bounded,
            ),
        ),
    )


def test_declared_numerical_manifest_exact_repeat_and_canonical(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    repeat = tmp_path / "repeat.npz"
    canonical = tmp_path / "canonical.npz"
    for path in (candidate, repeat, canonical):
        write_typed_graph(path)
    manifest = tmp_path / "numerical.json"
    _write_manifest(
        manifest,
        references=[
            _reference(repeat, role="repeat"),
            _reference(canonical, role="canonical"),
        ],
    )

    declaration = prepare_numerical_manifest_declaration(manifest)
    result = evaluate_declared_numerical_stability(
        candidate_graph_path=candidate,
        declaration=declaration,
    )

    assert [scope.scope_id for scope in result.report.scopes] == [
        "typed_graph.repeat",
        "feature_frontier.repeat",
        "typed_graph.canonical",
        "feature_frontier.canonical",
    ]
    assert all(
        scope.status is NumericalStabilityStatus.EXACT_STABLE
        for scope in result.report.scopes
        if scope.scope_id.startswith("typed_graph.")
    )
    assert all(
        scope.status is NumericalStabilityStatus.UNKNOWN
        for scope in result.report.scopes
        if scope.scope_id.startswith("feature_frontier.")
    )
    assert result.report.scopes[0].metrics["comparison_exact"] is True
    assert result.details["manifest_sha256"] == declaration["manifest_sha256"]


def test_declared_numerical_manifest_nonexact_is_unknown_without_policy(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    canonical = tmp_path / "canonical.npz"
    write_typed_graph(candidate)
    write_typed_graph(
        canonical,
        bucket_values={"feature<-error": [[1.0, 0.0], [0.0, -2.0]]},
    )
    manifest = tmp_path / "numerical.json"
    _write_manifest(manifest, references=[_reference(canonical, role="canonical")])

    result = evaluate_declared_numerical_stability(
        candidate_graph_path=candidate,
        declaration=prepare_numerical_manifest_declaration(manifest),
    )

    repeat, repeat_frontier, canonical_scope, canonical_frontier = (
        result.report.scopes
    )
    assert repeat.status is NumericalStabilityStatus.UNKNOWN
    assert repeat.reason_codes == ("declared_repeat_reference_absent",)
    assert repeat_frontier.status is NumericalStabilityStatus.UNKNOWN
    assert canonical_scope.status is NumericalStabilityStatus.UNKNOWN
    assert canonical_scope.reason_codes == ("frozen_numerical_policy_absent",)
    assert canonical_scope.metrics["comparison_exact"] is False
    assert canonical_frontier.reason_codes == ("declared_canonical_frontier_absent",)


@pytest.mark.parametrize(
    ("changed_value", "expected_status", "expected_reason"),
    [
        (1.08, NumericalStabilityStatus.BOUNDED, ()),
        (
            1.16,
            NumericalStabilityStatus.REVIEW,
            ("numerical_metrics_in_review_band",),
        ),
        (
            2.0,
            NumericalStabilityStatus.DIVERGENT,
            ("numerical_metrics_exceed_review_bounds",),
        ),
    ],
)
def test_frozen_calibration_classifies_typed_graph_bands(
    tmp_path: Path,
    changed_value: float,
    expected_status: NumericalStabilityStatus,
    expected_reason: tuple[str, ...],
) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    reference_graph = tmp_path / "reference.npz"
    write_typed_graph(reference_graph)
    write_typed_graph(
        candidate,
        bucket_values={
            "feature<-error": [[changed_value, 1.0], [1.0, 1.0]],
        },
    )
    manifest = tmp_path / "numerical.json"
    _write_manifest(
        manifest,
        references=[_reference(reference_graph, role="repeat")],
    )

    result = evaluate_declared_numerical_stability(
        candidate_graph_path=candidate,
        declaration=prepare_numerical_manifest_declaration(manifest),
        calibration=_frozen_calibration(),
    )

    scope = result.report.scopes[0]
    assert scope.status is expected_status
    assert scope.reason_codes == expected_reason
    assert scope.metrics["classification_policy_id"] == "correctness_calibration_v1"
    assert scope.metrics["calibration_fingerprint"] == "sha256:" + "f" * 64


def test_frozen_calibration_rejects_drift_in_required_exact_bucket(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    reference_graph = tmp_path / "reference.npz"
    write_typed_graph(reference_graph)
    write_typed_graph(
        candidate,
        bucket_values={"feature<-feature": [[1.1, 0.0], [0.0, 1.0]]},
    )
    manifest = tmp_path / "numerical.json"
    _write_manifest(
        manifest,
        references=[_reference(reference_graph, role="repeat")],
    )

    result = evaluate_declared_numerical_stability(
        candidate_graph_path=candidate,
        declaration=prepare_numerical_manifest_declaration(manifest),
        calibration=_frozen_calibration(),
    )

    scope = result.report.scopes[0]
    assert scope.status is NumericalStabilityStatus.DIVERGENT
    assert scope.reason_codes == ("required_exact_bucket_drift",)


def test_exact_arrays_from_a_different_step_are_not_exact_stable(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    candidate = tmp_path / "candidate.npz"
    reference_graph = tmp_path / "reference.npz"
    write_typed_graph(candidate, step_idx=1)
    write_typed_graph(reference_graph, step_idx=0)
    manifest = tmp_path / "numerical.json"
    _write_manifest(
        manifest,
        references=[_reference(reference_graph, role="repeat")],
    )

    result = evaluate_declared_numerical_stability(
        candidate_graph_path=candidate,
        declaration=prepare_numerical_manifest_declaration(manifest),
    )

    scope = result.report.scopes[0]
    assert scope.status is NumericalStabilityStatus.UNKNOWN
    assert scope.reason_codes == ("comparison_scope_incompatible",)
    assert scope.metrics["step_index_match"] is False
    assert scope.metrics["step_fingerprint_match"] is False
    assert scope.metrics["comparison_exact"] is False


def test_raw_frontier_decoder_pairs_are_deterministic_one_to_one() -> None:
    baseline = _frontier(10, 20, "1")
    candidate = _frontier(20, 10, "2")

    metrics, details, exact = _raw_frontier_comparison(
        baseline,
        candidate,
        baseline_descriptor=_descriptor(10, 20, [[1.0, 0.0], [0.0, 1.0]]),
        candidate_descriptor=_descriptor(20, 10, [[0.99, 0.01], [0.0, 1.0]]),
    )

    assert exact is False
    assert metrics["exact_exited_selected_count"] == 1
    assert metrics["exact_entered_selected_count"] == 1
    assert metrics["raw_decoder_pair_count"] == 1
    assert len(details["raw_one_to_one_pairs"]) == 1
    pair = details["raw_one_to_one_pairs"][0]
    assert pair["baseline"]["feature_id"] == 10
    assert pair["candidate"]["feature_id"] == 20
    assert pair["decoder_cosine"] > 0.99


def test_raw_frontier_exact_requires_exact_decoder_evidence() -> None:
    baseline = _frontier(10, 20, "1")
    candidate = _frontier(10, 20, "2")

    metrics, _details, exact = _raw_frontier_comparison(
        baseline,
        candidate,
        baseline_descriptor=_descriptor(10, 20, [[1.0, 0.0], [0.0, 1.0]]),
        candidate_descriptor=_descriptor(10, 20, [[0.99, 0.01], [0.0, 1.0]]),
    )

    assert exact is False
    assert metrics["frontier_records_exact"] is True
    assert metrics["decoder_evidence_exact"] is False
    assert metrics["frontier_evidence_exact"] is False


def test_calibrated_frontier_emits_typed_candidate_alias_handoff() -> None:
    baseline = _frontier(10, 20, "1")
    candidate = _frontier(20, 10, "2")
    baseline_descriptor = _descriptor(10, 20, [[1.0, 0.0], [1.0, 0.0]])
    candidate_descriptor = _descriptor(20, 10, [[1.0, 0.0], [1.0, 0.0]])
    calibration = _frozen_calibration()

    stability = compare_frontiers(
        baseline,
        candidate,
        decoder_source=_combined_descriptor_source(
            baseline_descriptor,
            candidate_descriptor,
        ),
        policy=_alias_match_policy(
            baseline,
            candidate,
            calibration=calibration,
        ),
    )
    classified = classify_frontier_stability(
        stability,
        frozen_policy=_frontier_classification_policy(
            calibration=calibration,
            review=False,
        ),
        exact_graph_or_scope_equal=False,
    )
    aliases = _qualified_alias_pairs_from_stability(
        stability,
        candidate_frontier=candidate,
        calibration=calibration,
    )

    assert classified.status is NumericalStabilityStatus.ALIAS_STABLE
    assert len(aliases) == 1
    alias = aliases[0]
    assert alias.source == FeatureKey(1, 2, 20)
    assert alias.substitute == FeatureKey(1, 2, 10)
    assert alias.qualified_decoder_cosine == pytest.approx(1.0)
    assert alias.selection_policy_id == "correctness_calibration_v1"
    assert alias.comparison_evidence_fingerprint.startswith("sha256:")
    assert alias.baseline_graph_fingerprint == baseline.graph_fingerprint
    assert alias.candidate_graph_fingerprint == candidate.graph_fingerprint


def test_manifest_preparation_rejects_transitive_hash_drift(tmp_path: Path) -> None:
    from typed_graph_fixtures import write_typed_graph

    graph = tmp_path / "reference.npz"
    write_typed_graph(graph)
    reference = _reference(graph, role="repeat")
    reference["graph_sha256"] = "sha256:" + "0" * 64
    manifest = tmp_path / "numerical.json"
    _write_manifest(manifest, references=[reference])

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        prepare_numerical_manifest_declaration(manifest)
