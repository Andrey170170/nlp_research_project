import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import circuit_utils  # noqa: E402
from trace_pipeline_chunked import (  # noqa: E402
    compact_result_to_bucketed_compact,
    compact_result_to_step_data,
)


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
