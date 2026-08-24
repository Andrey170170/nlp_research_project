from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from nlp_research_project.exact_trace_bench import compact_io
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    ArtifactProvenance,
    CANONICAL_BUCKET_NAMES,
    COMPACT_SAVE_FORMAT,
    DEFAULT_RETENTION_POLICY_ID,
    SCHEMA_VERSION,
    build_typed_compact_graph,
    get_retention_policy,
    load_typed_compact_graph,
    save_typed_compact_graph,
)
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    FullAnswerTypedGraphRequest,
    _build_typed_compact_graph as build_full_answer_graph,
)
from nlp_research_project.exact_trace_bench.full_answer.runner import (
    _save_typed_compact_graph as save_full_answer_graph,
)
from nlp_research_project.exact_trace_bench.trace_runtime.completion_workspace import (
    CompletionWorkspace,
)
from nlp_research_project.exact_trace_bench.trace_runtime.request import (
    trace_policy_from_scenario,
)
from nlp_research_project.exact_trace_bench.trace_runtime.step_artifacts import (
    StepArtifactWriter,
)


def _compact_result() -> dict[str, object]:
    return {
        "active_features": torch.tensor([[0, 0, 4], [1, 1, 9]]),
        "selected_features": torch.tensor([0, 1]),
        "feature_row_node_indices": torch.tensor([0, 1]),
        "logit_row_node_indices": torch.tensor([0]),
        "feature_feature_edges": torch.tensor([[3.0, -1.0], [0.0, 2.0]]),
        "feature_error_edges": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]]
        ),
        "feature_token_edges": torch.tensor([[0.5, 0.0], [0.0, 0.25]]),
        "logit_feature_edges": torch.tensor([[4.0, -3.0]]),
        "logit_error_edges": torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
        "logit_token_edges": torch.tensor([[-1.0, 1.5]]),
        "n_error_nodes": 4,
        "n_token_nodes": 2,
        "input_tokens": torch.tensor([10, 11]),
        "logit_targets": [SimpleNamespace(vocab_idx=42)],
        "semantic_fingerprint": "semantic-v1",
        "execution_fingerprint": "execution-v1",
    }


def _build():
    return build_typed_compact_graph(
        _compact_result(),
        3,
        token_text="answer",
        logprob=-0.25,
        provenance=ArtifactProvenance(
            provider={"family": "gemmascope", "revision": "frozen"},
            trace={
                "semantic_fingerprint": "semantic-v1",
                "execution_fingerprint": "execution-v1",
            },
            target={"token_id": 42, "position": 2},
            step={"trajectory": "fixture", "generated_index": 3},
        ),
    )


def _rewrite_npz(path: Path, **updates: object) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload.update(updates)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _provenance(step_idx: int = 0) -> ArtifactProvenance:
    return ArtifactProvenance(
        provider={"family": "gemmascope", "revision": "frozen"},
        trace={
            "semantic_fingerprint": "semantic-v1",
            "execution_fingerprint": "execution-v1",
        },
        target={"token_id": 42},
        step={"step_idx": step_idx},
    )


def test_policy_is_versioned_and_deterministic() -> None:
    first = get_retention_policy()
    second = get_retention_policy(DEFAULT_RETENTION_POLICY_ID)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == (
        "sha256:fb322e8ac6dcca4e588c4f0e1f39b1bc00c6b320b024afec07250927c17c2375"
    )
    assert first.to_json()["buckets"]["feature<-feature"] == {
        "top_p": 0.95,
        "cap": 1_000_000,
    }
    assert first.to_json()["buckets"]["logit<-token"] == {
        "top_p": 1.0,
        "cap": None,
    }
    with pytest.raises(ValueError, match="unknown typed edge retention policy"):
        get_retention_policy("unreviewed")


def test_build_save_and_strict_load_round_trip_without_legacy_edges(
    tmp_path: Path,
) -> None:
    graph = _build()
    path = tmp_path / "step_003.npz"
    save_typed_compact_graph(graph, path)

    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        assert {"row_idx", "col_idx", "weights"}.isdisjoint(fields)
        assert int(data["schema_version"]) == SCHEMA_VERSION
        assert str(data["compact_save_format"]) == COMPACT_SAVE_FORMAT

    loaded = load_typed_compact_graph(path)
    assert loaded.path == path
    assert loaded.bucket_names == CANONICAL_BUCKET_NAMES
    assert loaded.retention_policy_id == DEFAULT_RETENTION_POLICY_ID
    assert loaded.content_fingerprint == graph.content_fingerprint
    assert loaded.graph_fingerprint == graph.graph_fingerprint
    assert loaded.edge_count == graph.edge_count
    assert loaded.n_features == 2
    np.testing.assert_array_equal(loaded.feature_ids, graph.feature_ids)
    np.testing.assert_array_equal(loaded.bucket_row_idx, graph.bucket_row_idx)
    np.testing.assert_array_equal(loaded.bucket_col_idx, graph.bucket_col_idx)
    np.testing.assert_array_equal(loaded.bucket_weights, graph.bucket_weights)
    for metadata in loaded.bucket_metadata.values():
        assert set(metadata) == {
            "bucket",
            "raw_edge_count",
            "raw_abs_mass",
            "retained_edge_count",
            "retained_abs_mass",
            "retained_fraction",
            "cap_bound_before_top_p",
            "policy",
            "weights_signed",
        }


def test_content_fingerprint_is_data_level_deterministic() -> None:
    first = _build()
    second = _build()
    changed = build_typed_compact_graph(
        {**_compact_result(), "logit_token_edges": torch.tensor([[-1.0, 1.25]])},
        3,
        token_text="answer",
        logprob=-0.25,
        provenance=ArtifactProvenance(
            provider={"family": "gemmascope", "revision": "frozen"},
            trace={
                "semantic_fingerprint": "semantic-v1",
                "execution_fingerprint": "execution-v1",
            },
            target={"token_id": 42, "position": 2},
            step={"trajectory": "fixture", "generated_index": 3},
        ),
    )

    assert first.content_fingerprint == second.content_fingerprint
    assert first.retention_policy_fingerprint == second.retention_policy_fingerprint
    assert changed.content_fingerprint != first.content_fingerprint


def test_bucket_metadata_reports_when_cap_binds_before_top_p() -> None:
    result = {
        "active_features": torch.tensor([[0, 0, 4]]),
        "selected_features": torch.tensor([0]),
        "feature_row_node_indices": torch.tensor([0]),
        "logit_row_node_indices": torch.tensor([0]),
        "feature_feature_edges": torch.tensor([[1.0]]),
        "feature_error_edges": torch.ones((1, 20_000)),
        "feature_token_edges": torch.tensor([[1.0]]),
        "logit_feature_edges": torch.tensor([[1.0]]),
        "logit_error_edges": torch.zeros((1, 20_000)),
        "logit_token_edges": torch.tensor([[1.0]]),
        "n_error_nodes": 20_000,
        "n_token_nodes": 1,
        "input_tokens": torch.tensor([10]),
        "logit_targets": [SimpleNamespace(vocab_idx=42)],
        "semantic_fingerprint": "semantic-v1",
        "execution_fingerprint": "execution-v1",
    }

    graph = build_typed_compact_graph(result, 0, provenance=_provenance())
    metadata = graph.bucket_metadata["feature<-error"]
    assert metadata["raw_edge_count"] == 20_000
    assert metadata["retained_edge_count"] == 16_000
    assert metadata["retained_fraction"] == pytest.approx(0.8)
    assert metadata["cap_bound_before_top_p"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_tokens", torch.tensor([-1, 11]), "token_ids"),
        (
            "logit_targets",
            [SimpleNamespace(vocab_idx=-1)],
            "logit_token_ids",
        ),
        ("logit_targets", [], "logit_token_ids"),
    ],
)
def test_builder_rejects_invalid_token_identity_domains(
    field: str, value: object, message: str
) -> None:
    result = _compact_result()
    result[field] = value

    with pytest.raises(ValueError, match=message):
        build_typed_compact_graph(result, 0, provenance=_provenance())


def test_builder_rejects_all_zero_columns_outside_declared_shape() -> None:
    result = _compact_result()
    result["feature_error_edges"] = torch.zeros((2, 5))

    with pytest.raises(ValueError, match="does not match declared domains"):
        build_typed_compact_graph(result, 0, provenance=_provenance())


def test_builder_requires_compact_trace_fingerprints_for_binding() -> None:
    result = _compact_result()
    del result["execution_fingerprint"]

    with pytest.raises(ValueError, match="compact_result.execution_fingerprint"):
        build_typed_compact_graph(result, 0, provenance=_provenance())


def test_builder_rejects_feature_layer_outside_error_shape() -> None:
    result = _compact_result()
    result["feature_error_edges"] = torch.zeros((2, 2))
    result["logit_error_edges"] = torch.zeros((1, 2))
    result["n_error_nodes"] = 2

    with pytest.raises(ValueError, match="outside declared layer"):
        build_typed_compact_graph(result, 0, provenance=_provenance())


def test_full_answer_writer_boundary_rejects_empty_provider_identity() -> None:
    with pytest.raises(ValueError, match="provider identity has no meaningful value"):
        build_full_answer_graph(
            FullAnswerTypedGraphRequest(
                compact_result=_compact_result(),
                step_idx=0,
                token_text="",
                logprob=None,
                retention_policy_id=DEFAULT_RETENTION_POLICY_ID,
                provider_identity={"provider_id": None},
                trace_identity={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target_identity={"token_id": 42},
                step_identity={"step_idx": 0},
            )
        )


def test_full_answer_seam_binds_trace_result_fingerprints_and_reopens(
    tmp_path: Path,
) -> None:
    output = _compact_result()
    del output["semantic_fingerprint"]
    del output["execution_fingerprint"]
    trace_result = SimpleNamespace(
        output=output,
        semantic_fingerprint="semantic-v1",
        execution_fingerprint="execution-v1",
    )
    graph = build_full_answer_graph(
        FullAnswerTypedGraphRequest(
            compact_result=trace_result.output,
            step_idx=3,
            token_text="answer",
            logprob=-0.25,
            retention_policy_id=DEFAULT_RETENTION_POLICY_ID,
            provider_identity={"provider_id": "test-provider"},
            trace_identity={
                "semantic_fingerprint": trace_result.semantic_fingerprint,
                "execution_fingerprint": trace_result.execution_fingerprint,
            },
            target_identity={"token_id": 42, "token_text": "answer"},
            step_identity={"step_idx": 3},
        )
    )
    path = tmp_path / "token_000003" / "graph.npz"
    save_full_answer_graph(graph, path)

    assert "semantic_fingerprint" not in trace_result.output
    assert "execution_fingerprint" not in trace_result.output
    reopened = load_typed_compact_graph(path)
    assert reopened.trace_fingerprint == graph.trace_fingerprint
    assert reopened.content_fingerprint == graph.content_fingerprint


def test_full_answer_seam_rejects_output_fingerprint_conflict() -> None:
    output = _compact_result()
    output["semantic_fingerprint"] = "untrusted-output-value"

    with pytest.raises(ValueError, match="conflicts with authoritative TraceResult"):
        build_full_answer_graph(
            FullAnswerTypedGraphRequest(
                compact_result=output,
                step_idx=0,
                token_text="",
                logprob=None,
                retention_policy_id=DEFAULT_RETENTION_POLICY_ID,
                provider_identity={"provider_id": "test-provider"},
                trace_identity={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target_identity={"token_id": 42},
                step_identity={"step_idx": 0},
            )
        )


@pytest.mark.parametrize("legacy_field", ["row_idx", "col_idx", "weights"])
def test_strict_loader_rejects_mixed_legacy_fields(
    tmp_path: Path, legacy_field: str
) -> None:
    path = tmp_path / "step_003.npz"
    save_typed_compact_graph(_build(), path)
    _rewrite_npz(path, **{legacy_field: np.asarray([], dtype=np.int64)})

    with pytest.raises(ValueError, match="contains legacy edge fields"):
        load_typed_compact_graph(path)


def test_strict_loader_rejects_legacy_only_artifact(tmp_path: Path) -> None:
    path = tmp_path / "step_003.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_idx=np.asarray([0], dtype=np.int32),
            col_idx=np.asarray([0], dtype=np.int32),
            weights=np.asarray([1.0], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="contains legacy edge fields"):
        load_typed_compact_graph(path)

    historical = tmp_path / "step_004.npz"
    with historical.open("wb") as handle:
        np.savez_compressed(
            handle,
            step_idx=np.asarray(4, dtype=np.int32),
            row_idx=np.asarray([0], dtype=np.int32),
            col_idx=np.asarray([0], dtype=np.int32),
            weights=np.asarray([1.0], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 4]], dtype=np.int64),
            token_text=np.asarray("history"),
            logprob=np.asarray(np.nan),
            n_features=np.asarray(1, dtype=np.int32),
        )
    loaded = compact_io.load_historical_compact_graph(historical)
    assert loaded.step.token_text == "history"
    assert loaded.compact_save_format is None


def test_strict_loader_rejects_unknown_fields_and_fingerprint_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "step_003.npz"
    save_typed_compact_graph(_build(), path)
    _rewrite_npz(path, future_field=np.array("silent-drift"))
    with pytest.raises(ValueError, match="unknown=.*future_field"):
        load_typed_compact_graph(path)

    save_typed_compact_graph(_build(), path)
    _rewrite_npz(path, bucket_weights=np.zeros(_build().edge_count, dtype=np.float32))
    with pytest.raises(ValueError, match="only retained nonzero edges"):
        load_typed_compact_graph(path)


def test_builder_canonicalizes_edge_dtype_to_float32() -> None:
    result = _compact_result()
    result["logit_token_edges"] = torch.tensor([[-1.0, 1.5]], dtype=torch.float64)

    graph = build_typed_compact_graph(result, 0, provenance=_provenance())

    assert graph.bucket_weights.dtype == np.float32
    assert graph.bucket_metadata["logit<-token"]["raw_abs_mass"] == 2.5


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target={"token_id": 42},
                step={"step_idx": 2},
            ),
            "step.step_idx does not match",
        ),
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target={"logit_token_ids": [41]},
                step={"step_idx": 0},
            ),
            "target.logit_token_ids does not match",
        ),
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target={"target_token_id": 41},
                step={"step_idx": 0},
            ),
            "target.target_token_id does not match",
        ),
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "different-semantic",
                    "execution_fingerprint": "execution-v1",
                },
                target={"token_id": 42},
                step={"step_idx": 0},
            ),
            "trace.semantic_fingerprint does not match compact_result",
        ),
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "different-execution",
                },
                target={"token_id": 42},
                step={"step_idx": 0},
            ),
            "trace.execution_fingerprint does not match compact_result",
        ),
        (
            ArtifactProvenance(
                provider={"family": "gemmascope"},
                trace={
                    "semantic_fingerprint": "semantic-v1",
                    "execution_fingerprint": "execution-v1",
                },
                target={"target_token_text": "wrong"},
                step={"step_idx": 0},
            ),
            "target.target_token_text does not match token_text",
        ),
    ],
)
def test_builder_rejects_provenance_binding_mismatch(
    provenance: ArtifactProvenance, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_typed_compact_graph(_compact_result(), 0, provenance=provenance)


def test_strict_loader_rejects_duplicate_and_zero_retained_edges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "step_003.npz"
    save_typed_compact_graph(_build(), path)
    with np.load(path, allow_pickle=False) as data:
        rows = np.asarray(data["bucket_row_idx"])
        cols = np.asarray(data["bucket_col_idx"])
        weights = np.asarray(data["bucket_weights"])
        buckets = np.asarray(data["bucket_ids"])
    _rewrite_npz(
        path,
        bucket_row_idx=np.append(rows, rows[0]),
        bucket_col_idx=np.append(cols, cols[0]),
        bucket_weights=np.append(weights, weights[0]),
        bucket_ids=np.append(buckets, buckets[0]),
    )
    with pytest.raises(ValueError, match="duplicate coordinates"):
        load_typed_compact_graph(path)

    save_typed_compact_graph(_build(), path)
    _rewrite_npz(path, bucket_weights=np.zeros(_build().edge_count, dtype=np.float32))
    with pytest.raises(ValueError, match="only retained nonzero edges"):
        load_typed_compact_graph(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"retained_edge_count": 1_000_001}, "exceeds policy cap"),
        ({"raw_edge_count": 0}, "zero-state disagree"),
        ({"cap_bound_before_top_p": True}, "cap-bound metadata is impossible"),
    ],
)
def test_strict_loader_rejects_impossible_bucket_metadata(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    path = tmp_path / "step_003.npz"
    save_typed_compact_graph(_build(), path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["bucket_metadata_json"]))
    metadata["feature<-feature"].update(updates)
    _rewrite_npz(path, bucket_metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match=message):
        load_typed_compact_graph(path)


def test_loader_checks_step_identity_against_path(tmp_path: Path) -> None:
    path = tmp_path / "step_004.npz"
    save_typed_compact_graph(_build(), path)

    with pytest.raises(ValueError, match="does not match expected step 4"):
        load_typed_compact_graph(path)


def test_multistep_and_full_answer_writers_emit_the_same_v2_schema(
    tmp_path: Path,
) -> None:
    class Model:
        provider_id = "test-provider"

    workspace = CompletionWorkspace.create(
        tmp_path / "multistep", prompt_index=0, completion_index=0
    )
    writer = StepArtifactWriter(
        workspace=workspace,
        trace_policy=trace_policy_from_scenario({"method": "exact"}),
        model=Model(),
    )
    compact_result = _compact_result()
    writer.write(
        step_index=3,
        prefix_token_count=2,
        compact_result=compact_result,
        token_result={"token_id": 42, "token_text": "answer", "token_logprob": -0.25},
        attribution_seconds=1.0,
        token_generation_seconds=0.1,
        step_started=time.perf_counter(),
        stop=False,
    )

    full_graph = build_full_answer_graph(
        FullAnswerTypedGraphRequest(
            compact_result=compact_result,
            step_idx=3,
            token_text="answer",
            logprob=-0.25,
            retention_policy_id=DEFAULT_RETENTION_POLICY_ID,
            provider_identity={"provider_id": "test-provider"},
            trace_identity={
                "semantic_fingerprint": "semantic-v1",
                "execution_fingerprint": "execution-v1",
            },
            target_identity={"token_id": 42},
            step_identity={"step_idx": 3},
        )
    )
    full_answer_path = tmp_path / "full_answer" / "token_000003" / "graph.npz"
    save_full_answer_graph(full_graph, full_answer_path)

    with np.load(workspace.step_path(3), allow_pickle=False) as multistep_data:
        multistep_fields = set(multistep_data.files)
    with np.load(full_answer_path, allow_pickle=False) as full_answer_data:
        full_answer_fields = set(full_answer_data.files)
    assert multistep_fields == full_answer_fields
    assert {"row_idx", "col_idx", "weights"}.isdisjoint(multistep_fields)
    assert load_typed_compact_graph(workspace.step_path(3)).compact_save_format == (
        COMPACT_SAVE_FORMAT
    )
    assert (
        load_typed_compact_graph(
            full_answer_path, expected_step_idx=3
        ).retention_policy_id
        == DEFAULT_RETENTION_POLICY_ID
    )
