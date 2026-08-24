from __future__ import annotations

import sys
from dataclasses import dataclass
import json
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
    compare_step_pair,
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

    result = compare_step_pair(cast(Any, left), cast(Any, right))

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

    result = compare_step_pair(cast(Any, left), cast(Any, right))

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

    result = compare_step_pair(cast(Any, left), cast(Any, right))

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

    result = compare_step_pair(cast(Any, left), cast(Any, right))

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

    result = compare_step_pair(cast(Any, left), cast(Any, right))
    top256 = result["topk_edge_overlap"]["256"]

    assert top256["left_k_effective"] == 1
    assert top256["right_k_effective"] == 2
    assert top256["jaccard"] == 0.5


def test_topk_overlap_defines_both_empty_as_identical() -> None:
    left = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])
    right = _step(feature_ids=[(0, 0, 1)], rows=[], cols=[], weights=[])

    result = compare_step_pair(cast(Any, left), cast(Any, right))

    assert result["topk_edge_overlap"]["256"]["jaccard"] == 1.0


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), payload=np.asarray([1], dtype=np.int32))


def _write_bucketed_graph(
    path: Path,
    *,
    bucket_names: list[str],
    bucket_rows: list[int],
    bucket_cols: list[int],
    bucket_weights: list[float],
    bucket_ids: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from nlp_research_project.exact_trace_bench.compact_io import (
        CANONICAL_TYPED_BUCKET_NAMES,
    )

    persisted_names = [
        *bucket_names,
        *(name for name in CANONICAL_TYPED_BUCKET_NAMES if name not in bucket_names),
    ]
    persisted_ids = [persisted_names.index(bucket_names[value]) for value in bucket_ids]
    persisted_features = sorted(
        {
            int(endpoint)
            for name, row, col in zip(
                (bucket_names[value] for value in bucket_ids),
                bucket_rows,
                bucket_cols,
            )
            for endpoint in (
                ([row] if name.startswith("feature<-") else [])
                + ([col] if name.endswith("<-feature") else [])
            )
        }
    )
    feature_ids = np.asarray(
        [[0, 0, feature] for feature in persisted_features] or [[0, 0, 1]],
        dtype=np.int64,
    )
    counts = {
        name: sum(persisted_names[value] == name for value in persisted_ids)
        for name in persisted_names
    }
    masses = {
        name: sum(
            abs(weight)
            for value, weight in zip(persisted_ids, bucket_weights)
            if persisted_names[value] == name
        )
        for name in persisted_names
    }
    np.savez_compressed(
        path,
        row_idx=np.asarray([1], dtype=np.int32),
        col_idx=np.asarray([0], dtype=np.int32),
        weights=np.asarray([1.0], dtype=np.float32),
        feature_ids=feature_ids,
        token_text=np.asarray("A"),
        logprob=np.asarray(-0.1),
        n_features=np.asarray(len(feature_ids), dtype=np.int32),
        step_idx=np.asarray(0, dtype=np.int32),
        compact_save_format=np.asarray("typed_bucketed"),
        bucket_row_idx=np.asarray(bucket_rows, dtype=np.int64),
        bucket_col_idx=np.asarray(bucket_cols, dtype=np.int64),
        bucket_weights=np.asarray(bucket_weights, dtype=np.float32),
        bucket_ids=np.asarray(persisted_ids, dtype=np.int16),
        bucket_names=np.asarray(persisted_names),
        bucket_metadata_json=np.asarray(
            json.dumps(
                [
                    {
                        "bucket": name,
                        "raw_total_abs_mass": masses[name],
                        "retained_abs_mass": masses[name],
                        "retained_fraction": 1.0 if masses[name] else None,
                        "raw_nnz": counts[name],
                        "retained_nnz": counts[name],
                        "policy": {"top_p": 1.0, "cap": None},
                        "weights_signed": True,
                    }
                    for name in persisted_names
                ]
            )
        ),
        error_node_shape=np.asarray([1, 2], dtype=np.int32),
        token_ids=np.asarray([100, 101], dtype=np.int64),
        logit_token_ids=np.asarray([200], dtype=np.int64),
    )


def test_typed_bucket_comparison_is_invariant_to_coo_and_bucket_order(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_bucketed_graph(
        left,
        bucket_names=["feature<-feature", "feature<-error"],
        bucket_rows=[10, 10, 20],
        bucket_cols=[11, 11, 1],
        bucket_weights=[0.25, 0.75, -0.5],
        bucket_ids=[0, 0, 1],
    )
    _write_bucketed_graph(
        right,
        bucket_names=["feature<-error", "feature<-feature"],
        bucket_rows=[20, 10, 10],
        bucket_cols=[1, 11, 11],
        bucket_weights=[-0.5, 0.75, 0.25],
        bucket_ids=[0, 1, 1],
    )

    result = compare_compact_paths(left, right)
    typed = result["typed_bucket_comparison"]

    assert typed["classification"] == "strict_exact"
    assert typed["aggregate"]["exact"] is True
    assert typed["buckets"]["feature<-feature"]["edge_count_a"] == 1
    assert typed["buckets"]["feature<-feature"]["support_jaccard"] == 1.0
    assert typed["buckets"]["feature<-feature"]["topk_overlap"]["64"]["jaccard"] == 1.0


def test_typed_bucket_comparison_is_separate_from_legacy_all_edge_scope(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    common = {
        "bucket_names": ["feature<-feature", "feature<-error"],
        "bucket_rows": [10, 20],
        "bucket_cols": [11, 1],
        "bucket_ids": [0, 1],
    }
    _write_bucketed_graph(left, bucket_weights=[1.0, -1.0], **common)
    _write_bucketed_graph(right, bucket_weights=[1.0, -0.995], **common)

    result = compare_compact_paths(left, right)
    typed = result["typed_bucket_comparison"]
    error_bucket = typed["buckets"]["feature<-error"]

    assert result["all_edge_jaccard"] == 1.0
    assert result["all_edge_weighted_jaccard"] == 1.0
    assert result["all_edge_scope"] == "legacy_feature_source_edges_only"
    assert result["all_edge_includes_typed_buckets"] is False
    assert typed["classification"] == "non_exact"
    assert typed["non_exact_buckets"] == ["feature<-error"]
    assert typed["buckets"]["feature<-feature"]["exact"] is True
    assert error_bucket["support_jaccard"] == 1.0
    assert error_bucket["weighted_jaccard"] == pytest.approx(0.995)
    assert error_bucket["signed_normalized_l1_deviation"] == pytest.approx(0.005)
    assert error_bucket["shared_sign_agreement"] == 1.0
    assert error_bucket["exact"] is False


def test_compare_artifact_dirs_ignores_auxiliary_step_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_completion = tmp_path / "left" / "prompt_000" / "completion_000"
    right_completion = tmp_path / "right" / "prompt_000" / "completion_000"
    _write_npz(left_completion / "step_000.npz")
    _write_npz(right_completion / "step_000.npz")
    _write_npz(left_completion / "step_000_feature_semantic_descriptors.npz")
    _write_npz(right_completion / "step_000_phase3_seed_bundle.npz")

    loaded_names: list[str] = []

    def load_compact(path: Path) -> SimpleStep:
        loaded_names.append(path.name)
        return _step(
            feature_ids=[(0, 0, 1)],
            rows=[0],
            cols=[0],
            weights=[1.0],
        )

    from nlp_research_project.exact_trace_bench import compact_io
    from nlp_research_project.circuit_stability_analysis import signed_graph

    monkeypatch.setattr(compact_io, "load_compact", load_compact)
    monkeypatch.setattr(signed_graph, "load_signed_graph", lambda _path: None)

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert loaded_names == ["step_000.npz", "step_000.npz"]
    assert result["shared_completion_count"] == 1
    assert result["aligned_completion_count"] == 1
    assert result["aligned_step_count"] == 1
    assert result["comparison_complete"] is True
    assert len(result["step_comparisons"]) == 1
    assert result["step_comparisons"][0]["feature_jaccard"] == 1.0
    assert result["completion_comparisons"][0]["mean_top256_edge_jaccard"] == 1.0
    assert result["overall_mean_top256_edge_jaccard"] == 1.0
    assert result["step_comparisons"][0]["topk_edge_overlap"]["256"]["jaccard"] == 1.0


def test_compare_artifact_dirs_reports_incomplete_step_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_completion = tmp_path / "left" / "prompt_000" / "completion_000"
    right_completion = tmp_path / "right" / "prompt_000" / "completion_000"
    _write_npz(left_completion / "step_000.npz")
    _write_npz(left_completion / "step_001.npz")
    _write_npz(right_completion / "step_000.npz")

    from nlp_research_project.exact_trace_bench import compact_io
    from nlp_research_project.circuit_stability_analysis import signed_graph

    monkeypatch.setattr(
        compact_io,
        "load_compact",
        lambda path: SimpleStep(
            **{
                **_step(
                    feature_ids=[(0, 0, 1)],
                    rows=[0],
                    cols=[0],
                    weights=[1.0],
                ).__dict__,
                "step_idx": int(path.stem.split("_")[1]),
            }
        ),
    )
    monkeypatch.setattr(signed_graph, "load_signed_graph", lambda _path: None)

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert result["aligned_step_count"] == 1
    assert result["comparison_complete"] is False
    assert result["completion_comparisons"][0]["left_only_step_count"] == 1


def test_compare_artifact_dirs_persists_worst_step_not_only_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for side in ("left", "right"):
        completion = tmp_path / side / "prompt_000" / "completion_000"
        _write_npz(completion / "step_000.npz")
        _write_npz(completion / "step_001.npz")

    from nlp_research_project.exact_trace_bench import compact_io
    from nlp_research_project.circuit_stability_analysis import signed_graph

    def load_compact(path: Path) -> SimpleStep:
        step_idx = int(path.stem.split("_")[1])
        is_right = "right" in path.parts
        rows = [0] if step_idx == 0 or not is_right else [1]
        return _step(
            feature_ids=[(0, 0, 1)],
            rows=rows,
            cols=[0],
            weights=[1.0],
            step_idx=step_idx,
        )

    monkeypatch.setattr(compact_io, "load_compact", load_compact)
    monkeypatch.setattr(signed_graph, "load_signed_graph", lambda _path: None)

    result = compare_artifact_dirs(tmp_path / "left", tmp_path / "right")

    assert result["overall_mean_all_edge_jaccard"] == 0.5
    assert result["worst_step_all_edge_jaccard"] == 0.0
    assert result["worst_step_evidence"]["worst_step_all_edge_jaccard"] == [
        {
            "completion_key": "prompt_000/completion_000",
            "step_index_a": 1,
            "step_index_b": 1,
            "value": 0.0,
        }
    ]
    passed, reasons = evaluate_thresholds(
        result,
        {"worst_step_all_edge_jaccard_min": 0.98},
    )
    assert passed is False
    assert reasons == ["worst_step_all_edge_jaccard=0.0 < min 0.98"]
