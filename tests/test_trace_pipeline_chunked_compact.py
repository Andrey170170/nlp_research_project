import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nlp_research_project.exact_trace_bench import compact_io as circuit_utils  # noqa: E402
from nlp_research_project.exact_trace_bench.trace_runtime import (  # noqa: E402
    compact_result_to_bucketed_compact,
    compact_result_to_step_data,
)


def _bucket_schema(active_bucket: str) -> tuple[np.ndarray, int, str]:
    names = circuit_utils.CANONICAL_TYPED_BUCKET_NAMES
    metadata = [
        {
            "bucket": name,
            "raw_total_abs_mass": float(name == active_bucket),
            "retained_abs_mass": float(name == active_bucket),
            "retained_fraction": 1.0 if name == active_bucket else None,
            "raw_nnz": int(name == active_bucket),
            "retained_nnz": int(name == active_bucket),
            "policy": {"top_p": 1.0, "cap": None},
            "weights_signed": True,
        }
        for name in names
    ]
    return np.asarray(names), names.index(active_bucket), json.dumps(metadata)


def test_compact_result_to_step_data_saves_selected_feature_view() -> None:
    active_features = torch.tensor(
        [
            [0, 0, 10],
            [1, 0, 11],
            [2, 0, 12],
            [3, 0, 13],
            [4, 0, 14],
        ],
        dtype=torch.int64,
    )
    selected_features = torch.tensor([3, 1, 4], dtype=torch.int64)
    compact_result = {
        "active_features": active_features,
        "selected_features": selected_features,
        # Feature rows are stored in attribution order, not selected-feature order.
        "feature_row_node_indices": torch.tensor([4, 3, 1], dtype=torch.int64),
        "logit_row_node_indices": torch.tensor([999], dtype=torch.int64),
        "feature_feature_edges": torch.tensor(
            [
                [0.0, 4.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "logit_feature_edges": torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
    }

    step = compact_result_to_step_data(
        compact_result,
        step_idx=7,
        token_text="Yes",
        logprob=-0.25,
        max_edges=10,
    )

    assert step.n_features == 3
    np.testing.assert_array_equal(
        step.feature_ids, active_features[selected_features].numpy()
    )
    assert step.token_text == "Yes"
    assert step.logprob == -0.25
    assert step.step_idx == 7

    edge_weights = {
        (int(row), int(col)): float(weight)
        for row, col, weight in zip(step.row_idx, step.col_idx, step.weights)
    }
    assert set(edge_weights) == {(2, 1), (0, 0), (1, 2), (3, 2)}
    assert np.isclose(edge_weights[(2, 1)], 4.0 / 10.0)
    assert np.isclose(edge_weights[(0, 0)], 3.0 / 10.0)
    assert np.isclose(edge_weights[(1, 2)], 1.0 / 10.0)
    assert np.isclose(edge_weights[(3, 2)], 2.0 / 10.0)


def test_bucketed_compact_policy_and_optional_npz_fields(tmp_path) -> None:
    class Target:
        vocab_idx = 42

    active_features = torch.tensor([[0, 0, 10], [1, 0, 11]], dtype=torch.int64)
    compact_result = {
        "active_features": active_features,
        "selected_features": torch.tensor([0, 1], dtype=torch.int64),
        "feature_row_node_indices": torch.tensor([0, 1], dtype=torch.int64),
        "logit_row_node_indices": torch.tensor([5], dtype=torch.int64),
        "feature_feature_edges": torch.tensor([[-0.9, 0.1], [0.0, 0.2]]),
        "logit_feature_edges": torch.tensor([[0.3, 0.4]]),
        "feature_error_edges": torch.tensor([[1.0, 0.01], [0.5, 0.0]]),
        "feature_token_edges": torch.tensor([[0.7, 0.1], [0.0, 0.2]]),
        "logit_error_edges": torch.tensor([[0.2, 0.0]]),
        "logit_token_edges": torch.tensor([[0.6, 0.4]]),
        "n_error_nodes": 2,
        "n_token_nodes": 2,
        "input_tokens": torch.tensor([100, 101], dtype=torch.int64),
        "logit_targets": [Target()],
    }

    bucketed = compact_result_to_bucketed_compact(
        compact_result,
        step_idx=3,
        token_text="x",
        policies={
            "feature<-error": {"top_p": 1.0, "cap": 1},
            "feature<-feature": {"top_p": 0.80, "cap": 10},
        },
    )
    path = tmp_path / "graph.npz"
    circuit_utils.save_bucketed_compact(bucketed, path)
    loaded = circuit_utils.load_compact(path)
    data = np.load(path, allow_pickle=False)

    assert loaded.step_idx == 3
    assert str(data["compact_save_format"]) == "typed_bucketed"
    names = [str(x) for x in data["bucket_names"].tolist()]
    assert set(names) == {
        "feature<-feature",
        "feature<-error",
        "feature<-token",
        "logit<-feature",
        "logit<-error",
        "logit<-token",
    }
    metadata = {
        row["bucket"]: row for row in json.loads(str(data["bucket_metadata_json"]))
    }
    assert metadata["feature<-error"]["retained_nnz"] == 1
    assert metadata["feature<-error"]["raw_nnz"] == 3
    assert metadata["feature<-feature"]["weights_signed"] is True
    feature_feature_bucket = names.index("feature<-feature")
    feature_feature_weights = data["bucket_weights"][
        data["bucket_ids"] == feature_feature_bucket
    ]
    assert np.any(feature_feature_weights < 0)


def test_compact_loader_rejects_typed_bucket_schema_drift(tmp_path) -> None:
    names, bucket_id, metadata = _bucket_schema("feature<-feature")
    step = circuit_utils.StepData(
        step_idx=2,
        row_idx=np.asarray([0], dtype=np.int32),
        col_idx=np.asarray([0], dtype=np.int32),
        weights=np.asarray([1.0], dtype=np.float32),
        feature_ids=np.asarray([[0, 1, 10]], dtype=np.int64),
        token_text="x",
        logprob=None,
        n_features=1,
    )
    graph = circuit_utils.BucketedCompact(
        step=step,
        bucket_row_idx=np.asarray([0], dtype=np.int64),
        bucket_col_idx=np.asarray([1_000_010], dtype=np.int64),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000002" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload["bucket_col_idx"] = np.asarray([], dtype=np.int64)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="typed bucket COO arrays"):
        circuit_utils.load_compact_graph(path)


def test_compact_loader_enforces_step_identity_and_position_constraint(
    tmp_path,
) -> None:
    path = tmp_path / "token_000003" / "graph.npz"
    circuit_utils.save_compact(
        circuit_utils.StepData(
            step_idx=3,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 2, 10], [1, 3, 11]], dtype=np.int64),
            token_text="x",
            logprob=-0.25,
            n_features=2,
        ),
        path,
    )

    graph = circuit_utils.load_compact_graph(path, expected_step_idx=3)
    positions = circuit_utils.summarize_feature_positions(
        graph, max_position_exclusive=3
    )

    assert positions.feature_count == 2
    assert positions.typed_feature_endpoint_count == 0
    assert positions.max_position == 3
    assert positions.future_position_count == 1
    assert positions.has_future_positions
    with pytest.raises(ValueError, match="does not match expected step 4"):
        circuit_utils.load_compact_graph(path, expected_step_idx=4)


@pytest.mark.parametrize(
    ("bucket_name", "row", "col", "message"),
    [
        ("logit<-feature", 1, 10, "outside logit_token_ids"),
        ("feature<-error", 10, 2, "outside error_node_shape"),
        ("feature<-token", 10, 2, "outside token_ids"),
    ],
)
def test_compact_loader_rejects_typed_endpoint_outside_domain(
    tmp_path, bucket_name: str, row: int, col: int, message: str
) -> None:
    names, bucket_id, metadata = _bucket_schema(bucket_name)
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        bucket_row_idx=np.asarray([row], dtype=np.int64),
        bucket_col_idx=np.asarray([col], dtype=np.int64),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)

    with pytest.raises(ValueError, match=message):
        circuit_utils.load_compact_graph(path)


def test_position_constraint_includes_typed_feature_endpoints(tmp_path) -> None:
    names, bucket_id, metadata = _bucket_schema("feature<-feature")
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10], [0, 3, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=2,
        ),
        bucket_row_idx=np.asarray([3 * circuit_utils.FEATURE_ID_BASE + 10]),
        bucket_col_idx=np.asarray([10]),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 4], dtype=np.int32),
        token_ids=np.asarray([100, 101, 102, 103], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)

    positions = circuit_utils.summarize_feature_positions(
        circuit_utils.load_compact_graph(path), max_position_exclusive=3
    )

    assert positions.typed_feature_endpoint_count == 2
    assert positions.max_position == 3
    assert positions.future_position_count == 2


def test_compact_loader_rejects_unpersisted_feature_source(tmp_path) -> None:
    names, bucket_id, metadata = _bucket_schema("logit<-feature")
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        bucket_row_idx=np.asarray([0], dtype=np.int64),
        bucket_col_idx=np.asarray([11], dtype=np.int64),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)

    with pytest.raises(ValueError, match="not a persisted selected feature"):
        circuit_utils.load_compact_graph(path)


def test_compact_loader_rejects_unpersisted_feature_target(tmp_path) -> None:
    names, bucket_id, metadata = _bucket_schema("feature<-token")
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        bucket_row_idx=np.asarray([11], dtype=np.int64),
        bucket_col_idx=np.asarray([0], dtype=np.int64),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)

    with pytest.raises(ValueError, match="target row is not a persisted selected"):
        circuit_utils.load_compact_graph(path)


def test_compact_loader_requires_canonical_typed_bucket_names(tmp_path) -> None:
    names, bucket_id, metadata = _bucket_schema("feature<-token")
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        bucket_row_idx=np.asarray([10]),
        bucket_col_idx=np.asarray([0]),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload["bucket_names"] = payload["bucket_names"][:-1]
    payload["bucket_metadata_json"] = np.asarray(
        json.dumps(json.loads(str(payload["bucket_metadata_json"]))[:-1])
    )
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="canonical six typed buckets"):
        circuit_utils.load_compact_graph(path)


@pytest.mark.parametrize("field", ["n_features", "step_idx"])
def test_compact_loader_requires_integer_identity_scalars(tmp_path, field: str) -> None:
    path = tmp_path / "graph.npz"
    circuit_utils.save_compact(
        circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        path,
    )
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload[field] = np.asarray(float(payload[field]))
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=f"{field} must be an integer scalar"):
        circuit_utils.load_compact_graph(path)


def test_compact_loader_rejects_duplicate_feature_identities(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    circuit_utils.save_compact(
        circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10], [0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=2,
        ),
        path,
    )

    with pytest.raises(ValueError, match="unique identities"):
        circuit_utils.load_compact_graph(path)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_retained_nnz", "missing required fields"),
        ("raw_nnz_too_small", "raw_nnz must be an integer"),
        ("nonfinite_raw_mass", "finite and nonnegative"),
        ("retained_exceeds_raw", "retained mass exceeds raw mass"),
        ("retained_mass_mismatch", "retained mass does not match weights"),
        ("fraction_mismatch", "retained_fraction is inconsistent"),
        ("zero_mass_fraction", "retained_fraction must be None"),
        ("invalid_policy", "policy top_p is invalid"),
        ("nonbool_weights_signed", "weights_signed must be bool"),
    ],
)
def test_compact_loader_rejects_corrupt_bucket_coverage_metadata(
    tmp_path, corruption: str, message: str
) -> None:
    names, bucket_id, metadata_json = _bucket_schema("feature<-token")
    graph = circuit_utils.BucketedCompact(
        step=circuit_utils.StepData(
            step_idx=0,
            row_idx=np.asarray([], dtype=np.int32),
            col_idx=np.asarray([], dtype=np.int32),
            weights=np.asarray([], dtype=np.float32),
            feature_ids=np.asarray([[0, 0, 10]], dtype=np.int64),
            token_text="x",
            logprob=None,
            n_features=1,
        ),
        bucket_row_idx=np.asarray([10]),
        bucket_col_idx=np.asarray([0]),
        bucket_weights=np.asarray([1.0], dtype=np.float32),
        bucket_ids=np.asarray([bucket_id], dtype=np.int16),
        bucket_names=names,
        bucket_metadata_json=metadata_json,
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    path = tmp_path / "token_000000" / "graph.npz"
    circuit_utils.save_bucketed_compact(graph, path)
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    metadata = json.loads(str(payload["bucket_metadata_json"]))
    active = metadata[bucket_id]
    if corruption == "missing_retained_nnz":
        del active["retained_nnz"]
    elif corruption == "raw_nnz_too_small":
        active["raw_nnz"] = 0
    elif corruption == "nonfinite_raw_mass":
        active["raw_total_abs_mass"] = float("nan")
    elif corruption == "retained_exceeds_raw":
        active["retained_abs_mass"] = 2.0
    elif corruption == "retained_mass_mismatch":
        active["retained_abs_mass"] = 0.5
        active["retained_fraction"] = 0.5
    elif corruption == "fraction_mismatch":
        active["retained_fraction"] = 0.5
    elif corruption == "zero_mass_fraction":
        metadata[0]["retained_fraction"] = 0.0
    elif corruption == "invalid_policy":
        active["policy"] = {"top_p": 2.0, "cap": None}
    elif corruption == "nonbool_weights_signed":
        active["weights_signed"] = 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corruption)
    payload["bucket_metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        circuit_utils.load_compact_graph(path)
