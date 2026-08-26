from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nlp_research_project.exact_trace_bench.correctness.frontier_artifact import (
    CUTOFF_BASIS,
    DECODER_DESCRIPTOR_KIND,
    FALLBACK_DESCRIPTOR_KIND,
    build_bounded_frontier_artifact,
    load_bounded_frontier_artifact,
    save_bounded_frontier_artifact,
)
from nlp_research_project.exact_trace_bench.trace_runtime.artifacts import (
    load_and_validate_capture_artifact,
    save_feature_semantic_descriptors,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    ArtifactProvenance,
    build_typed_compact_graph,
    load_typed_compact_graph,
    save_typed_compact_graph,
)


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _compact_result() -> dict[str, object]:
    active = torch.tensor(
        [[0, 0, 10], [1, 1, 20], [1, 1, 21], [0, 1, 11]],
        dtype=torch.int64,
    )
    return {
        "active_features": active,
        "selected_features": torch.tensor([0, 1], dtype=torch.int64),
        "feature_row_node_indices": torch.tensor([0, 1], dtype=torch.int64),
        "logit_row_node_indices": torch.tensor([0], dtype=torch.int64),
        "feature_feature_edges": torch.tensor(
            [[1.0, -0.25], [0.5, 2.0]], dtype=torch.float32
        ),
        "feature_error_edges": torch.ones((2, 4), dtype=torch.float32),
        "feature_token_edges": torch.ones((2, 2), dtype=torch.float32),
        "logit_feature_edges": torch.tensor([[0.8, -0.4]], dtype=torch.float32),
        "logit_error_edges": torch.ones((1, 4), dtype=torch.float32),
        "logit_token_edges": torch.ones((1, 2), dtype=torch.float32),
        "n_error_nodes": 4,
        "n_token_nodes": 2,
        "input_tokens": torch.tensor([100, 101], dtype=torch.int64),
        "logit_targets": [SimpleNamespace(vocab_idx=42)],
        "semantic_fingerprint": "semantic-v1",
        "execution_fingerprint": "execution-v1",
    }


def _reopened_graph(tmp_path: Path, compact: dict[str, object]):
    graph = build_typed_compact_graph(
        compact,
        0,
        provenance=ArtifactProvenance(
            provider={"provider": "fixture"},
            trace={
                "semantic_fingerprint": "semantic-v1",
                "execution_fingerprint": "execution-v1",
            },
            target={"target_token_id": 42},
            step={"step_idx": 0},
        ),
    )
    path = tmp_path / "step_000.npz"
    save_typed_compact_graph(graph, path)
    return load_typed_compact_graph(path)


def _descriptors() -> dict[str, object]:
    descriptors: dict[str, object] = {
        "status": "captured",
        "descriptor_kind": DECODER_DESCRIPTOR_KIND,
        "descriptor_version": "v2",
        "descriptor_scope": "active_occurrence_downstream_decoder_rows_v1",
        "descriptor_is_decoder_evidence": True,
        "decoder_source_fingerprint": _fingerprint("decoder"),
        "decoder_evidence_fingerprint": _fingerprint("pending"),
        "projection_id": "fixed_width_countsketch_l2_v1",
        "projection_fingerprint": _fingerprint("projection"),
        "semantic_descriptor_projection_max_bytes": 64 * 1024 * 1024,
        "semantic_descriptor_projection_required_bytes": 4096,
        "semantic_descriptor_projection_workspace_peak_bytes": 2048,
        "semantic_descriptor_projection_admitted": True,
        "semantic_descriptor_projection_released": True,
        "candidate_bound_kind": "final_selected_plus_seed_near_cutoff_controls_v1",
        "semantic_descriptor_control_limit": 2,
        "total_active_features": 4,
        "candidate_features": np.asarray(
            [[0, 0, 10], [1, 1, 20], [1, 1, 21], [0, 1, 11]],
            dtype=np.int64,
        ),
        "candidate_row_indices": np.arange(4, dtype=np.int64),
        "activation_value": np.asarray([2.0, -3.0, -2.9, 0.5]),
        "seed_influence": np.asarray([1.0, 0.8, 0.79, 0.5]),
        "seed_rank": np.arange(4, dtype=np.int64),
        "is_top_seed": np.asarray([True, True, False, False]),
        "is_selected_phase4": np.asarray([True, True, False, False]),
        "phase4_selected_rank": np.asarray([0, 1, -1, -1], dtype=np.int64),
        "phase4_selection_available": True,
        "seed_influence_available": True,
        "semantic_descriptor_transient_policy_id": "bounded_seed_frontier_handoff_v1",
        "semantic_descriptor_transient_max_bytes": 64 * 1024 * 1024,
        "semantic_descriptor_transient_required_bytes": 192,
        "semantic_descriptor_transient_admitted": True,
        "semantic_descriptor_transient_released": True,
        "semantic_descriptor_transient_array_count": 3,
        "semantic_sketch": np.eye(4, 8, dtype=np.float32),
    }
    _refresh_decoder_evidence_fingerprint(descriptors)
    return descriptors


def _refresh_decoder_evidence_fingerprint(descriptors: dict[str, object]) -> None:
    digest = hashlib.sha256()
    digest.update(str(descriptors["decoder_source_fingerprint"]).encode("ascii"))
    digest.update(str(descriptors["projection_fingerprint"]).encode("ascii"))
    digest.update(np.ascontiguousarray(descriptors["candidate_features"]).tobytes())
    digest.update(np.ascontiguousarray(descriptors["semantic_sketch"]).tobytes())
    descriptors["decoder_evidence_fingerprint"] = f"sha256:{digest.hexdigest()}"


def _fallback_descriptors() -> dict[str, object]:
    descriptors = _descriptors()
    descriptors["descriptor_kind"] = FALLBACK_DESCRIPTOR_KIND
    descriptors["descriptor_version"] = "v1"
    for field in (
        "descriptor_scope",
        "descriptor_is_decoder_evidence",
        "decoder_source_fingerprint",
        "decoder_evidence_fingerprint",
        "projection_id",
        "projection_fingerprint",
        "semantic_descriptor_projection_max_bytes",
        "semantic_descriptor_projection_required_bytes",
        "semantic_descriptor_projection_workspace_peak_bytes",
        "semantic_descriptor_projection_admitted",
        "semantic_descriptor_projection_released",
    ):
        descriptors.pop(field)
    return descriptors


def test_builds_and_round_trips_bounded_frontier_sidecar(tmp_path: Path) -> None:
    compact = _compact_result()
    graph = _reopened_graph(tmp_path, compact)

    artifact = build_bounded_frontier_artifact(
        graph=graph,
        compact_result=compact,
        feature_semantic_descriptors=_descriptors(),
        decoder_fingerprint=_fingerprint("decoder"),
        max_near_cutoff=1,
        max_typed_neighborhood_edges=2,
    )

    assert artifact.cutoff_basis == CUTOFF_BASIS
    assert artifact.descriptor_kind == DECODER_DESCRIPTOR_KIND
    assert artifact.evidence.cutoff_score == pytest.approx(0.8)
    assert artifact.evidence.relative_cutoff_gap == pytest.approx(0.01 / 0.8)
    assert artifact.evidence.cutoff_tie_count == 1
    assert artifact.evidence.near_cutoff_count == 1
    first, second = artifact.evidence.selected
    assert (first.seed_rank, first.final_selected_rank) == (0, 0)
    assert (second.seed_rank, second.final_selected_rank) == (1, 1)
    assert first.signed_target_effect == pytest.approx(np.float32(0.8))
    assert [(edge.source.feature_id, edge.weight) for edge in first.feature_from_feature] == [
        (10, 1.0),
        (20, -0.25),
    ]
    assert [(edge.logit_token_id, edge.weight) for edge in first.logit_from_feature] == [
        (42, pytest.approx(np.float32(0.8)))
    ]
    control = artifact.evidence.near_cutoff[0]
    assert control.key.feature_id == 21
    assert control.seed_rank == 2
    assert control.final_selected_rank is None
    assert control.feature_from_feature == ()
    assert control.logit_from_feature == ()

    path = tmp_path / "frontier.json"
    save_bounded_frontier_artifact(artifact, path)
    first_bytes = path.read_bytes()
    save_bounded_frontier_artifact(artifact, path)
    assert path.read_bytes() == first_bytes
    assert load_bounded_frontier_artifact(path) == artifact
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["descriptor_is_decoder_evidence"] is True
    assert "semantic_sketch" not in serialized


def test_refuses_missing_selected_or_near_cutoff_descriptor_coverage(
    tmp_path: Path,
) -> None:
    compact = _compact_result()
    graph = _reopened_graph(tmp_path, compact)
    missing_selected = _descriptors()
    for name in (
        "candidate_features",
        "candidate_row_indices",
        "activation_value",
        "seed_influence",
        "seed_rank",
        "is_top_seed",
        "is_selected_phase4",
        "phase4_selected_rank",
    ):
        missing_selected[name] = np.asarray(missing_selected[name])[[0, 2, 3]]
    missing_selected["semantic_sketch"] = np.asarray(
        missing_selected["semantic_sketch"]
    )[[0, 2, 3]]
    _refresh_decoder_evidence_fingerprint(missing_selected)
    missing_selected["total_active_features"] = 4

    with pytest.raises(ValueError, match="does not cover selected features"):
        build_bounded_frontier_artifact(
            graph=graph,
            compact_result=compact,
            feature_semantic_descriptors=missing_selected,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=1,
        )

    insufficient_band = _descriptors()
    for name in (
        "candidate_features",
        "candidate_row_indices",
        "activation_value",
        "seed_influence",
        "seed_rank",
        "is_top_seed",
        "is_selected_phase4",
        "phase4_selected_rank",
    ):
        insufficient_band[name] = np.asarray(insufficient_band[name])[[0, 1, 2]]
    insufficient_band["semantic_sketch"] = np.asarray(
        insufficient_band["semantic_sketch"]
    )[[0, 1, 2]]
    _refresh_decoder_evidence_fingerprint(insufficient_band)
    with pytest.raises(ValueError, match="required bounded near-cutoff band"):
        build_bounded_frontier_artifact(
            graph=graph,
            compact_result=compact,
            feature_semantic_descriptors=insufficient_band,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=3,
        )


def test_refuses_nonfinite_duplicate_and_fingerprint_drift(tmp_path: Path) -> None:
    compact = _compact_result()
    graph = _reopened_graph(tmp_path, compact)

    nonfinite = _descriptors()
    nonfinite["seed_influence"] = np.asarray([1.0, 0.8, np.nan, 0.5])
    with pytest.raises(ValueError, match="must be finite"):
        build_bounded_frontier_artifact(
            graph=graph,
            compact_result=compact,
            feature_semantic_descriptors=nonfinite,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=1,
        )

    duplicate = _descriptors()
    duplicate["candidate_features"] = np.asarray(
        [[0, 0, 10], [1, 1, 20], [1, 1, 21], [1, 1, 21]], dtype=np.int64
    )
    duplicate["candidate_row_indices"] = np.asarray([0, 1, 2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="candidate_row_indices must be unique"):
        build_bounded_frontier_artifact(
            graph=graph,
            compact_result=compact,
            feature_semantic_descriptors=duplicate,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=1,
        )

    artifact = build_bounded_frontier_artifact(
        graph=graph,
        compact_result=compact,
        feature_semantic_descriptors=_descriptors(),
        decoder_fingerprint=_fingerprint("decoder"),
        max_near_cutoff=1,
    )
    path = tmp_path / "frontier.json"
    save_bounded_frontier_artifact(artifact, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["cutoff_score"] = 0.7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content fingerprint mismatch"):
        load_bounded_frontier_artifact(path)


def test_capture_writer_refuses_transients_and_persists_bound_metadata(
    tmp_path: Path,
) -> None:
    transient = _descriptors()
    transient["_transient_active_features"] = np.zeros((4, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="transients must be finalized"):
        save_feature_semantic_descriptors(transient, tmp_path / "refused.npz")

    path = tmp_path / "bounded.npz"
    save_feature_semantic_descriptors(_descriptors(), path)
    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["candidate_bound_kind"]) == (
            "final_selected_plus_seed_near_cutoff_controls_v1"
        )
        assert int(payload["semantic_descriptor_control_limit"]) == 2
        assert str(payload["semantic_descriptor_transient_policy_id"]) == (
            "bounded_seed_frontier_handoff_v1"
        )
        assert int(payload["semantic_descriptor_transient_max_bytes"]) == 64 * 1024 * 1024
        assert int(payload["semantic_descriptor_transient_required_bytes"]) == 192
        assert bool(payload["semantic_descriptor_transient_admitted"])
        assert bool(payload["semantic_descriptor_transient_released"])
        assert int(payload["semantic_descriptor_transient_array_count"]) == 3
        assert str(payload["descriptor_scope"]) == (
            "active_occurrence_downstream_decoder_rows_v1"
        )
        assert bool(payload["descriptor_is_decoder_evidence"])
        assert str(payload["decoder_source_fingerprint"]) == _fingerprint("decoder")
        assert int(payload["semantic_descriptor_projection_required_bytes"]) == 4096
        assert not any(name.startswith("_transient_") for name in payload.files)

    reopened = load_and_validate_capture_artifact("feature_semantic_descriptors", path)
    assert bool(reopened["semantic_descriptor_transient_released"])


def test_frontier_refuses_fallback_identity_descriptors(tmp_path: Path) -> None:
    descriptors = _fallback_descriptors()

    with pytest.raises(ValueError, match="qualification-grade decoder evidence"):
        build_bounded_frontier_artifact(
            graph=_reopened_graph(tmp_path, _compact_result()),
            compact_result=_compact_result(),
            feature_semantic_descriptors=descriptors,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=1,
        )


def test_fallback_capture_and_legacy_frontier_remain_readable(tmp_path: Path) -> None:
    fallback_path = tmp_path / "fallback.npz"
    save_feature_semantic_descriptors(_fallback_descriptors(), fallback_path)
    fallback = load_and_validate_capture_artifact(
        "feature_semantic_descriptors", fallback_path
    )
    assert str(fallback["descriptor_kind"]) == FALLBACK_DESCRIPTOR_KIND
    assert not bool(fallback["descriptor_is_decoder_evidence"])

    graph = _reopened_graph(tmp_path, _compact_result())
    artifact = build_bounded_frontier_artifact(
        graph=graph,
        compact_result=_compact_result(),
        feature_semantic_descriptors=_descriptors(),
        decoder_fingerprint=_fingerprint("decoder"),
        max_near_cutoff=1,
    )
    legacy = artifact.to_json()
    legacy["schema_version"] = 1
    legacy["artifact_format"] = "bounded_frontier_evidence_v1"
    legacy["descriptor_kind"] = FALLBACK_DESCRIPTOR_KIND
    legacy["descriptor_version"] = "v1"
    legacy["descriptor_is_decoder_evidence"] = False
    for field in (
        "descriptor_scope",
        "decoder_source_fingerprint",
        "decoder_evidence_fingerprint",
        "projection_fingerprint",
    ):
        legacy.pop(field)
    legacy["content_fingerprint"] = None
    canonical = json.dumps(
        legacy, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    legacy["content_fingerprint"] = (
        f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    )
    legacy_path = tmp_path / "legacy-frontier.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    reopened = load_bounded_frontier_artifact(legacy_path)
    assert reopened.schema_version == 1
    assert reopened.descriptor_is_decoder_evidence is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("semantic_descriptor_transient_max_bytes", 1, "exactly 64 MiB"),
        ("semantic_descriptor_transient_required_bytes", 64 * 1024 * 1024 + 1, "exceeds"),
        ("semantic_descriptor_transient_admitted", False, "admitted and released"),
        ("semantic_descriptor_transient_released", False, "admitted and released"),
        ("semantic_descriptor_transient_array_count", 2, "exactly 3"),
    ],
)
def test_capture_and_frontier_refuse_invalid_transient_receipt(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    descriptors = _descriptors()
    descriptors[field] = value
    with pytest.raises(ValueError, match=message):
        save_feature_semantic_descriptors(descriptors, tmp_path / "invalid.npz")

    compact = _compact_result()
    graph = _reopened_graph(tmp_path, compact)
    with pytest.raises(ValueError, match=message):
        build_bounded_frontier_artifact(
            graph=graph,
            compact_result=compact,
            feature_semantic_descriptors=descriptors,
            decoder_fingerprint=_fingerprint("decoder"),
            max_near_cutoff=1,
        )
