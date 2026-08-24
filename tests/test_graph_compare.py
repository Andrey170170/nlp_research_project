from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from nlp_research_project.exact_trace_bench.graph_compare import (  # noqa: E402
    compare_artifact_dirs,
    compare_compact_paths,
    compare_historical_step_pair,
)
from nlp_research_project.exact_trace_bench.baselines import (  # noqa: E402
    evaluate_thresholds,
)


@dataclass
class SimpleStep:
    step_idx: int
    row_idx: np.ndarray
    col_idx: np.ndarray
    weights: np.ndarray
    feature_ids: np.ndarray
    token_text: str
    logprob: float | None
    n_features: int


def _step(
    *,
    feature_ids,
    rows,
    cols,
    weights,
    step_idx: int = 0,
    token_text: str = "Let",
) -> SimpleStep:
    return SimpleStep(
        step_idx=step_idx,
        row_idx=np.asarray(rows, dtype=np.int32),
        col_idx=np.asarray(cols, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float32),
        feature_ids=np.asarray(feature_ids, dtype=np.int64),
        token_text=token_text,
        logprob=-0.1,
        n_features=len(feature_ids),
    )


def test_compare_step_pair_reports_shared_unique_edge_decomposition() -> None:
    left = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1, 2, 3],
        cols=[0, 0, 0],
        weights=[0.4, 0.2, 0.4],
    )
    right = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 4)],
        rows=[1, 2, 3],
        cols=[0, 1, 2],
        weights=[0.5, 0.2, 0.3],
    )

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))

    assert result["feature_support_decomposition"]["shared_count"] == 2
    assert result["feature_support_decomposition"]["left_unique_count"] == 1
    assert result["feature_support_decomposition"]["right_unique_count"] == 1

    left_classes = result["edge_class_decomposition_a"]
    right_classes = result["edge_class_decomposition_b"]
    assert left_classes["shared_to_shared"]["edge_count"] == 1
    assert left_classes["shared_to_unique"]["edge_count"] == 1
    assert left_classes["shared_to_logit"]["edge_count"] == 1
    assert right_classes["shared_to_shared"]["edge_count"] == 1
    assert right_classes["shared_to_unique"]["edge_count"] == 1
    assert right_classes["unique_to_logit"]["edge_count"] == 1
    assert left_classes["shared_to_logit"]["mass_fraction"] == pytest.approx(0.4)

    shared_edge_stability = result["shared_endpoint_edge_stability"]
    assert shared_edge_stability["common_edge_count"] == 1
    assert shared_edge_stability["edge_jaccard"] == 1.0
    assert shared_edge_stability["weighted_edge_jaccard"] == pytest.approx(0.8)
    assert shared_edge_stability["topk_overlap"]["64"]["shared_count"] == 1

    assert result["all_edge_weighted_jaccard"] < 1.0


def test_all_edge_metrics_and_token_identity_detect_logit_drift() -> None:
    left = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2)],
        rows=[1, 2],
        cols=[0, 0],
        weights=[0.5, 0.5],
        token_text="A",
    )
    right = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2)],
        rows=[1, 3],
        cols=[0, 0],
        weights=[0.5, 0.5],
        token_text="B",
    )

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))

    assert result["edge_jaccard"] == 1.0
    assert result["all_edge_jaccard"] == pytest.approx(1 / 3)
    assert result["all_edge_topk_overlap"]["256"]["jaccard"] == pytest.approx(1 / 3)
    assert result["target_token_match"] == 0.0


def test_all_edge_deviation_detects_magnitude_difference() -> None:
    left = _step(
        feature_ids=[(0, 0, 1)],
        rows=[1],
        cols=[0],
        weights=[1.0],
    )
    right = _step(
        feature_ids=[(0, 0, 1)],
        rows=[1],
        cols=[0],
        weights=[0.5],
    )

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))

    assert result["all_edge_jaccard"] == 1.0
    assert result["all_edge_weighted_jaccard"] == 0.5
    assert result["all_edge_normalized_l1_deviation"] == 0.5


def test_signed_compact_gate_rejects_sign_flip_hidden_by_magnitude_metrics() -> None:
    left = _step(
        feature_ids=[(0, 0, 1)],
        rows=[1],
        cols=[0],
        weights=[1.0],
    )
    right = _step(
        feature_ids=[(0, 0, 1)],
        rows=[1],
        cols=[0],
        weights=[-1.0],
    )

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))

    assert result["all_edge_weighted_jaccard"] == 1.0
    assert result["all_edge_normalized_l1_deviation"] == 0.0
    assert result["all_edge_shared_sign_agreement"] == 0.0
    assert result["all_edge_signed_normalized_l1_deviation"] == 2.0
    passed, reasons = evaluate_thresholds(
        {
            "worst_step_all_edge_shared_sign_agreement": result[
                "all_edge_shared_sign_agreement"
            ],
            "worst_step_all_edge_signed_normalized_l1_deviation": result[
                "all_edge_signed_normalized_l1_deviation"
            ],
        },
        {
            "worst_step_all_edge_shared_sign_agreement_min": 1.0,
            "worst_step_all_edge_signed_normalized_l1_deviation_max": 0.000001,
        },
    )
    assert passed is False
    assert any("shared_sign_agreement" in reason for reason in reasons)


def test_topk_overlap_penalizes_edges_present_on_only_one_side() -> None:
    left = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1],
        cols=[0],
        weights=[1.0],
    )
    right = _step(
        feature_ids=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        rows=[1, 2],
        cols=[0, 0],
        weights=[1.0, 0.5],
    )

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))
    top256 = result["topk_edge_overlap"]["256"]

    assert top256["left_k_effective"] == 1
    assert top256["right_k_effective"] == 2
    assert top256["jaccard"] == 0.5


def test_topk_overlap_defines_both_empty_as_identical() -> None:
    left = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])
    right = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])

    result = compare_historical_step_pair(cast(Any, left), cast(Any, right))

    assert result["topk_edge_overlap"]["256"]["jaccard"] == 1.0


def test_typed_comparison_reports_six_named_buckets(tmp_path: Path) -> None:
    from typed_graph_fixtures import write_typed_graph

    left = tmp_path / "step_000.npz"
    right = tmp_path / "candidate" / "step_000.npz"
    write_typed_graph(left)
    write_typed_graph(right)

    result = compare_compact_paths(left, right)

    assert result["policy_compatible"] is True
    assert result["typed_bucket_comparison"]["classification"] == "strict_exact"
    assert set(result["typed_bucket_comparison"]["buckets"]) == {
        "feature<-feature",
        "feature<-error",
        "feature<-token",
        "logit<-feature",
        "logit<-error",
        "logit<-token",
    }
    assert result["bucket_feature_error_exact"] == 1.0
    assert not any(key.startswith("all_edge_") for key in result)


def test_typed_comparison_localizes_drift_by_bucket(tmp_path: Path) -> None:
    from typed_graph_fixtures import write_typed_graph

    left = tmp_path / "left" / "step_000.npz"
    right = tmp_path / "right" / "step_000.npz"
    write_typed_graph(left)
    write_typed_graph(
        right,
        bucket_values={"feature<-error": [[1.0, 0.0], [0.0, -2.0]]},
    )

    result = compare_compact_paths(left, right)

    assert result["typed_bucket_comparison"]["non_exact_buckets"] == ["feature<-error"]
    assert result["bucket_feature_feature_exact"] == 1.0
    assert result["bucket_feature_error_exact"] == 0.0


def test_historical_tie_cutoff_is_not_canonical_policy_compatible(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import (
        write_historical_tie_cutoff_graph,
        write_typed_graph,
    )

    candidate = tmp_path / "candidate" / "step_000.npz"
    historical = tmp_path / "historical" / "step_000.npz"
    write_typed_graph(
        candidate,
        bucket_values={"feature<-feature": [[18.0, 1.0], [1.0, 0.0]]},
    )
    write_historical_tie_cutoff_graph(historical, canonical_source=candidate)

    result = compare_compact_paths(historical, candidate, historical_reference=True)
    fingerprints = result["fingerprints"]

    assert result["typed_bucket_comparison"]["non_exact_buckets"] == [
        "feature<-feature"
    ]
    assert fingerprints["reference_bucket_rules_match_candidate"] is True
    assert fingerprints["reference_policy_identity"] == (
        "historical_bucket_top_p_unstable_argsort_v0"
    )
    assert fingerprints["reference_retention_algorithm_version"] == (
        "absolute_mass_top_p_then_cap_unstable_ties_v0"
    )
    assert (
        fingerprints["reference_retention_policy_fingerprint"]
        != (fingerprints["candidate_retention_policy_fingerprint"])
    )
    assert fingerprints["policy_compatible"] is False
    assert fingerprints["candidate_current_policy_valid"] is True
    assert result["policy_compatible"] is False


@pytest.mark.parametrize(
    ("historical_overrides", "expected_error"),
    [
        ({"token_ids": [999, 11]}, "token_ids domain mismatch"),
        ({"logit_token_ids": [999]}, "logit_token_ids domain mismatch"),
        ({"error_node_shape": (3, 2)}, "error_node_shape domain mismatch"),
        (
            {"logit_token_ids": [-1]},
            "logit_token_ids comparison scope is unavailable",
        ),
    ],
)
def test_historical_comparison_rejects_domain_drift_or_unknown_identity(
    tmp_path: Path,
    historical_overrides: dict[str, Any],
    expected_error: str,
) -> None:
    from typed_graph_fixtures import (
        write_historical_tie_cutoff_graph,
        write_typed_graph,
    )

    candidate = tmp_path / "candidate" / "step_000.npz"
    historical = tmp_path / "historical" / "step_000.npz"
    write_typed_graph(
        candidate,
        bucket_values={"feature<-feature": [[18.0, 1.0], [1.0, 0.0]]},
    )
    write_historical_tie_cutoff_graph(
        historical,
        canonical_source=candidate,
        **historical_overrides,
    )

    with pytest.raises(ValueError, match=expected_error):
        compare_compact_paths(historical, candidate, historical_reference=True)


@pytest.mark.parametrize(
    ("token_ids", "logit_token_ids", "error_layer_count", "expected_domain"),
    [
        ([999, 11], None, None, "token_ids"),
        (None, [999], None, "logit_token_ids"),
        (None, None, 3, "error_node_shape"),
    ],
)
def test_typed_comparison_rejects_endpoint_domain_drift(
    tmp_path: Path,
    token_ids: list[int] | None,
    logit_token_ids: list[int] | None,
    error_layer_count: int | None,
    expected_domain: str,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    left = tmp_path / "left" / "step_000.npz"
    right = tmp_path / "right" / "step_000.npz"
    write_typed_graph(left)
    write_typed_graph(
        right,
        token_ids=token_ids,
        logit_token_ids=logit_token_ids,
        error_layer_count=error_layer_count,
    )

    with pytest.raises(ValueError, match=expected_domain):
        compare_compact_paths(left, right)


def test_compare_artifact_dirs_uses_strict_v2_and_ignores_sidecars(
    tmp_path: Path,
) -> None:
    from typed_graph_fixtures import write_typed_graph

    for side in ("left", "right"):
        completion = tmp_path / side / "prompt_000" / "completion_000"
        write_typed_graph(completion / "step_000.npz")
        np.savez_compressed(
            completion / "step_000_feature_semantic_descriptors.npz",
            payload=np.asarray([1]),
        )

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert result["comparison_complete"] is True
    assert result["aligned_step_count"] == 1
    assert result["policy_compatible"] is True
    assert result["candidate_current_policy_valid"] is True
    assert result["worst_step_bucket_feature_feature_exact"] == 1.0
