import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trace_pipeline_chunked import compact_result_to_step_data  # noqa: E402


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
