from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.correctness.numerical import (
    _raw_frontier_comparison,
    build_reference_identity_receipt,
    evaluate_declared_numerical_stability,
    prepare_numerical_manifest_declaration,
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
