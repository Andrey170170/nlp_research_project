from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nlp_research_project.exact_trace_bench.full_answer.calibration import (
    build_scorecard,
)
from nlp_research_project.exact_trace_bench.full_answer.role_classification import (
    apply_role_classification_reviews,
    build_role_matched_calibration_manifest,
    classify_analysis_pair_manifest,
    classify_generated_token_roles,
)


def _trajectory(tokens: list[str]) -> dict:
    return {
        "schema_version": 1,
        "trajectory_id": "traj",
        "prompt_token_count": 3,
        "prompt_token_ids": [1, 2, 3],
        "generated_tokens": [
            {
                "generated_index": index,
                "absolute_token_position": 3 + index,
                "token_id": 100 + index,
                "token_text": token,
                "is_stop": False,
            }
            for index, token in enumerate(tokens)
        ],
    }


def _write_graph(run_root: Path, index: int) -> None:
    path = run_root / "shards" / "shard_000" / f"token_{index:06d}" / "graph.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_idx=np.asarray([1], dtype=np.int64),
        col_idx=np.asarray([0], dtype=np.int64),
        weights=np.asarray([1.0], dtype=np.float32),
        feature_ids=np.asarray([(0, index, 1)], dtype=np.int64),
        token_text=np.asarray(str(index)),
        logprob=np.asarray(np.nan),
        n_features=np.asarray(1, dtype=np.int64),
        step_idx=np.asarray(index, dtype=np.int64),
    )


def test_classify_generated_token_roles_identifies_math_and_answer() -> None:
    roles = classify_generated_token_roles(
        _trajectory(
            [
                "We",
                " compute",
                " 2",
                " +",
                " 3",
                " =",
                " 5",
                ".",
                " Answer",
                " is",
                " 5",
            ]
        )
    )
    by_index = {role.generated_index: role for role in roles}

    assert by_index[2].role_cluster == "math_number"
    assert by_index[3].role_cluster == "math_operator"
    assert by_index[8].role_cluster == "answer_marker"
    assert by_index[10].role_cluster == "answer_number"


def test_role_matched_manifest_builds_temporal_and_null_pairs(tmp_path: Path) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    records = []
    pair: dict[str, object] = {"prompt_id": "p1"}
    for label, suffix in [("correct", "c"), ("wrong", "w")]:
        name = f"p1_{suffix}"
        traj = prep / f"{name}.json"
        traj.write_text(
            json.dumps(_trajectory(["Let", " 2", " +", " 3", " =", " 5", "."])),
            encoding="utf-8",
        )
        run_root = prep / name / "run"
        for index in range(7):
            _write_graph(run_root, index)
        records.append({"name": name, "trajectory": str(traj)})
        pair[label] = {
            "name": name,
            "run_root": str(run_root),
            "token_count": 7,
        }
    pair_manifest = tmp_path / "pairs.json"
    pair_manifest.write_text(json.dumps({"pairs": [pair]}), encoding="utf-8")
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"records": records}), encoding="utf-8")

    catalog = classify_analysis_pair_manifest(
        pair_manifest_path=pair_manifest,
        launch_prep_manifest=launch,
        output_dir=tmp_path / "roles",
    )
    manifest = build_role_matched_calibration_manifest(
        classification_catalog=Path(catalog["output_dir"])
        / "role_classification_catalog.json",
        output_path=tmp_path / "role_pairs.json",
        role_clusters=["math_number", "math_operator"],
    )

    assert manifest["pair_count"] > 0
    assert set(manifest["pair_counts_by_category"]) == {"noise", "temporal", "null"}
    assert {pair["metadata"]["role_cluster"] for pair in manifest["pairs"]} <= {
        "math_number",
        "math_operator",
    }
    noise_pair = next(
        pair for pair in manifest["pairs"] if pair["pair_category"] == "noise"
    )
    assert noise_pair["left_graph"] == noise_pair["right_graph"]
    assert noise_pair["metadata"]["noise_anchor_kind"] == "identity_self_pair"


def test_scorecard_counts_role_matched_temporal_pairs_by_role() -> None:
    rows = []
    for role in ["math_number", "math_operator"]:
        rows.extend(
            [
                {
                    "bucket": "feature",
                    "metric": "cosine",
                    "params_json": "{}",
                    "position_band": "mid",
                    "role_cluster": role,
                    "pair_category": "noise",
                    "sub_category": "identity_self_same_role",
                    "value": 1.0,
                },
                {
                    "bucket": "feature",
                    "metric": "cosine",
                    "params_json": "{}",
                    "position_band": "mid",
                    "role_cluster": role,
                    "pair_category": "temporal",
                    "sub_category": "role_matched_within_trajectory",
                    "lag": 7,
                    "value": 0.8,
                },
                {
                    "bucket": "feature",
                    "metric": "cosine",
                    "params_json": "{}",
                    "position_band": "mid",
                    "role_cluster": role,
                    "pair_category": "null",
                    "sub_category": "same_prompt_correct_wrong_same_role",
                    "value": 0.2,
                },
            ]
        )

    scorecard = build_scorecard(rows)

    assert {row["role_cluster"] for row in scorecard} == {
        "math_number",
        "math_operator",
    }
    assert all(row["temporal_adjacent_count"] == 1 for row in scorecard)


def test_apply_role_classification_reviews_writes_reviewed_catalog(
    tmp_path: Path,
) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    records = []
    pair: dict[str, object] = {"prompt_id": "p1"}
    for label, suffix in [("correct", "c"), ("wrong", "w")]:
        name = f"p1_{suffix}"
        traj = prep / f"{name}.json"
        traj.write_text(
            json.dumps(_trajectory(["Let", " 2", " +", " 3", "."])),
            encoding="utf-8",
        )
        run_root = prep / name / "run"
        records.append({"name": name, "trajectory": str(traj)})
        pair[label] = {
            "name": name,
            "run_root": str(run_root),
            "token_count": 5,
        }
    pair_manifest = tmp_path / "pairs.json"
    pair_manifest.write_text(json.dumps({"pairs": [pair]}), encoding="utf-8")
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"records": records}), encoding="utf-8")

    catalog = classify_analysis_pair_manifest(
        pair_manifest_path=pair_manifest,
        launch_prep_manifest=launch,
        output_dir=tmp_path / "roles",
    )
    for record in catalog["records"]:
        role_path = Path(record["classification_path"])
        corrections = []
        if record["label"] == "correct":
            corrections.append(
                {
                    "generated_index": 2,
                    "suggested_role_cluster": "punctuation_format",
                    "reason": "treat plus as formatting for test",
                }
            )
        review_path = role_path.with_name(
            role_path.name.removesuffix(".roles.json") + ".roles.review.json"
        )
        review_path.write_text(
            json.dumps({"corrections": corrections}), encoding="utf-8"
        )

    reviewed = apply_role_classification_reviews(
        classification_catalog=Path(catalog["output_dir"])
        / "role_classification_catalog.json",
        output_dir=tmp_path / "reviewed_roles",
    )

    assert reviewed["total_applied_corrections"] == 1
    assert reviewed["missing_review_count"] == 0
    corrected_record = next(
        record for record in reviewed["records"] if record["label"] == "correct"
    )
    corrected_payload = json.loads(
        Path(corrected_record["classification_path"]).read_text(encoding="utf-8")
    )
    corrected_token = corrected_payload["tokens"][2]
    assert corrected_token["role_cluster"] == "punctuation_format"
    assert "subagent_review_corrected" in corrected_token["tags"]
