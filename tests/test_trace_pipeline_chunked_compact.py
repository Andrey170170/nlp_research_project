import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nlp_research_project.exact_trace_bench import compact_io as circuit_utils  # noqa: E402


def _save_historical_step(step: circuit_utils.StepData, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_idx=step.row_idx,
        col_idx=step.col_idx,
        weights=step.weights,
        feature_ids=step.feature_ids,
        token_text=np.asarray(step.token_text),
        logprob=np.asarray(step.logprob if step.logprob is not None else np.nan),
        n_features=np.asarray(step.n_features, dtype=np.int32),
        step_idx=np.asarray(step.step_idx, dtype=np.int32),
    )


def _save_historical_bucketed(bundle: circuit_utils.BucketedCompact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    step = bundle.step
    np.savez_compressed(
        path,
        row_idx=step.row_idx,
        col_idx=step.col_idx,
        weights=step.weights,
        feature_ids=step.feature_ids,
        token_text=np.asarray(step.token_text),
        logprob=np.asarray(step.logprob if step.logprob is not None else np.nan),
        n_features=np.asarray(step.n_features, dtype=np.int32),
        step_idx=np.asarray(step.step_idx, dtype=np.int32),
        compact_save_format=np.asarray("typed_bucketed"),
        bucket_row_idx=bundle.bucket_row_idx,
        bucket_col_idx=bundle.bucket_col_idx,
        bucket_weights=bundle.bucket_weights,
        bucket_ids=bundle.bucket_ids,
        bucket_names=bundle.bucket_names,
        bucket_metadata_json=np.asarray(bundle.bucket_metadata_json),
        error_node_shape=bundle.error_node_shape,
        token_ids=bundle.token_ids,
        logit_token_ids=bundle.logit_token_ids,
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
    _save_historical_bucketed(graph, path)
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload["bucket_col_idx"] = np.asarray([], dtype=np.int64)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="typed bucket COO arrays"):
        circuit_utils.load_historical_compact_graph(path)


def test_compact_loader_enforces_step_identity_and_position_constraint(
    tmp_path,
) -> None:
    path = tmp_path / "token_000003" / "graph.npz"
    _save_historical_step(
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

    graph = circuit_utils.load_historical_compact_graph(path, expected_step_idx=3)
    positions = circuit_utils.summarize_feature_positions(
        graph, max_position_exclusive=3
    )

    assert positions.feature_count == 2
    assert positions.typed_feature_endpoint_count == 0
    assert positions.max_position == 3
    assert positions.future_position_count == 1
    assert positions.has_future_positions
    with pytest.raises(ValueError, match="does not match expected step 4"):
        circuit_utils.load_historical_compact_graph(path, expected_step_idx=4)


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
    _save_historical_bucketed(graph, path)

    with pytest.raises(ValueError, match=message):
        circuit_utils.load_historical_compact_graph(path)


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
    _save_historical_bucketed(graph, path)

    positions = circuit_utils.summarize_feature_positions(
        circuit_utils.load_historical_compact_graph(path), max_position_exclusive=3
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
    _save_historical_bucketed(graph, path)

    with pytest.raises(ValueError, match="not a persisted selected feature"):
        circuit_utils.load_historical_compact_graph(path)


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
    _save_historical_bucketed(graph, path)

    with pytest.raises(ValueError, match="target row is not a persisted selected"):
        circuit_utils.load_historical_compact_graph(path)


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
    _save_historical_bucketed(graph, path)
    with np.load(path, allow_pickle=False) as valid:
        payload = {name: valid[name] for name in valid.files}
    payload["bucket_names"] = payload["bucket_names"][:-1]
    payload["bucket_metadata_json"] = np.asarray(
        json.dumps(json.loads(str(payload["bucket_metadata_json"]))[:-1])
    )
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="canonical six typed buckets"):
        circuit_utils.load_historical_compact_graph(path)


@pytest.mark.parametrize("field", ["n_features", "step_idx"])
def test_compact_loader_requires_integer_identity_scalars(tmp_path, field: str) -> None:
    path = tmp_path / "graph.npz"
    _save_historical_step(
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
        circuit_utils.load_historical_compact_graph(path)


def test_compact_loader_rejects_duplicate_feature_identities(tmp_path) -> None:
    path = tmp_path / "graph.npz"
    _save_historical_step(
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
        circuit_utils.load_historical_compact_graph(path)


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
    _save_historical_bucketed(graph, path)
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
        circuit_utils.load_historical_compact_graph(path)
