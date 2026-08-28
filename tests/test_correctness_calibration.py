from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nlp_research_project.exact_trace_bench.correctness.calibration import (
    REQUIRED_NUMERICAL_SCOPES,
    calibration_declaration_from_graph_knobs,
    load_declared_correctness_calibration,
    prepare_correctness_calibration_declaration,
)
from nlp_research_project.exact_trace_bench.correctness.numerical import (
    build_reference_identity_receipt,
    prepare_numerical_manifest_declaration,
)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_calibration_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    from typed_graph_fixtures import write_typed_graph

    repeat = tmp_path / "repeat.npz"
    canonical = tmp_path / "canonical.npz"
    write_typed_graph(repeat)
    write_typed_graph(canonical)
    numerical = tmp_path / "numerical.json"
    numerical.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "exact_trace_numerical_reference_manifest_v1",
                "references": [
                    _reference(repeat, role="repeat"),
                    _reference(canonical, role="canonical"),
                ],
            }
        ),
        encoding="utf-8",
    )
    numerical_declaration = prepare_numerical_manifest_declaration(numerical)
    source_receipts = _write_source_receipts(tmp_path)
    manifest = tmp_path / "correctness_calibration_v1.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "format": "exact_trace_correctness_calibration_v1",
        "calibration_id": "correctness_calibration_v1",
        "policy_id": "behavioral_closure_v1",
        "numerical_reference": numerical_declaration,
        "required_numerical_scopes": list(REQUIRED_NUMERICAL_SCOPES),
        "numerical": {
            "feature_jaccard": {"pass_min": 0.995, "review_min": 0.99},
            "buckets": {
                "feature<-error": _bucket(0.03, 0.05),
                "logit<-error": _bucket(0.01, 0.02),
            },
            "exact_buckets": [
                "feature<-feature",
                "feature<-token",
                "logit<-feature",
                "logit<-token",
            ],
            "frontier": {
                "selected_recovery": {"pass_min": 0.995, "review_min": 0.99},
                "unmatched_influence_mass_fraction": {
                    "pass_max": 0.005,
                    "review_max": 0.01,
                },
                "alias_min_decoder_cosine": 0.95,
                "bounded_deltas": {
                    "max_cutoff_score_abs_delta": 1e-5,
                    "max_relative_cutoff_gap_abs_delta": 1e-5,
                    "max_cutoff_tie_count_abs_delta": 0,
                    "max_record_scalar_abs_delta": 1e-4,
                    "max_typed_edge_weight_abs_delta": 1e-4,
                },
            },
        },
        "behavioral": {
            "no_op_absolute_tolerance": 1e-6,
            "no_op_relative_tolerance": 1e-6,
            "direct_min_abs_predicted_target_delta": 0.25,
            "relative_error_epsilon": 1e-6,
            "direct_max_mean_relative_closure": 0.05,
            "direct_max_relative_closure": 0.10,
            "direct_min_sign_agreement": 0.95,
            "downstream_max_mean_relative_closure": 0.05,
            "downstream_max_p95_relative_closure": 0.10,
            "necessity_min_predicted_realized_spearman": 0.8,
            "necessity_min_median_high_control_effect_ratio": 2.0,
            "alias_max_relative_effect_error": 0.2,
        },
        "source_receipts": source_receipts,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def _reference(path: Path, *, role: str) -> dict[str, object]:
    return {
        "reference_id": f"{role}-fixture",
        "role": role,
        "graph_path": str(path.resolve()),
        "graph_sha256": _sha256(path),
        "identity": build_reference_identity_receipt(path),
    }


def _bucket(pass_l1: float, review_l1: float) -> dict[str, object]:
    return {
        "normalized_l1_deviation": {
            "pass_max": pass_l1,
            "review_max": review_l1,
        },
        "weighted_jaccard": {"pass_min": 0.97, "review_min": 0.95},
        "shared_sign_agreement": {"pass_min": 0.995, "review_min": 0.99},
    }


def _write_source_receipts(tmp_path: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for run in ("r4", "r5", "r6"):
        source = tmp_path / f"{run}_numerical.json"
        source.write_text(
            json.dumps(_numerical_source_payload()), encoding="utf-8"
        )
        receipts.append(
            {
                "source_id": f"step1b_{run}_numerical",
                "artifact_kind": "exact_trace_numerical_comparison_details_v1",
                "path": str(source.resolve()),
                "sha256": _sha256(source),
            }
        )
    behavioral = tmp_path / "r6_behavioral.json"
    behavioral.write_text(
        json.dumps(_behavioral_source_payload()), encoding="utf-8"
    )
    receipts.append(
        {
            "source_id": "step1b_r6_behavioral",
            "artifact_kind": "circuit_tracer_behavioral_faithfulness_report_v1",
            "path": str(behavioral.resolve()),
            "sha256": _sha256(behavioral),
        }
    )
    return receipts


def _numerical_source_payload() -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    for role in ("repeat", "canonical"):
        comparison: dict[str, object] = {
            "feature_jaccard": 1.0,
            "bucket_feature_error_normalized_l1_deviation": 0.01,
            "bucket_feature_error_weighted_jaccard": 0.99,
            "bucket_feature_error_shared_sign_agreement": 1.0,
            "bucket_logit_error_normalized_l1_deviation": 0.005,
            "bucket_logit_error_weighted_jaccard": 0.995,
            "bucket_logit_error_shared_sign_agreement": 1.0,
        }
        for bucket in (
            "feature_feature",
            "feature_token",
            "logit_feature",
            "logit_token",
        ):
            comparison[f"bucket_{bucket}_exact"] = 1.0
        comparisons.append(
            {
                "reference_id": f"{role}-fixture",
                "role": role,
                "graph_path": f"/tmp/{role}.npz",
                "graph_sha256": "sha256:" + "a" * 64,
                "identity": {},
                "scope_compatible": True,
                "compatibility": {},
                "comparison_exact": False,
                "comparison": comparison,
            }
        )
    return {
        "schema_version": 1,
        "format": "exact_trace_numerical_comparison_details_v1",
        "manifest_path": "/tmp/numerical_reference_manifest.json",
        "manifest_sha256": "sha256:" + "b" * 64,
        "candidate_graph_path": "/tmp/candidate.npz",
        "candidate_graph_sha256": "sha256:" + "c" * 64,
        "comparisons": comparisons,
        "frontier_comparisons": [],
    }


def _behavioral_source_payload() -> dict[str, object]:
    return {
        "schema": "behavioral_faithfulness_report",
        "schema_version": 1,
        "evidence_fingerprint": "fixture-evidence",
        "report": {
            "evidence_completeness": "complete",
            "verdict": "unknown",
            "runtime_status": "complete",
            "variants_planned": 4,
            "variants_completed": 4,
            "no_op_required": True,
            "no_op_passed": True,
            "raw_execution": {
                "status": "complete",
                "cleanup_completed": True,
            },
        },
    }


def test_calibration_preparation_validates_complete_receipt_closure(
    tmp_path: Path,
) -> None:
    manifest, _payload = _write_calibration_fixture(tmp_path)

    declaration = prepare_correctness_calibration_declaration(manifest)
    calibration = load_declared_correctness_calibration(declaration)

    assert calibration.calibration_id == "correctness_calibration_v1"
    assert calibration.required_numerical_scopes == REQUIRED_NUMERICAL_SCOPES
    assert (
        calibration.numerical.buckets["feature<-error"].normalized_l1_deviation.pass_max
        == 0.03
    )
    assert calibration.behavioral.direct_max_relative_closure == 0.10
    assert calibration.calibration_fingerprint == declaration["manifest_sha256"]
    policy_kwargs = calibration.behavioral_probe_policy_kwargs()
    assert policy_kwargs["no_op_absolute_tolerance"] == 1e-6
    assert policy_kwargs["calibration"].direct_max_mean_relative_closure == 0.05


def test_calibration_preparation_rejects_source_receipt_drift(tmp_path: Path) -> None:
    manifest, payload = _write_calibration_fixture(tmp_path)
    source_path = Path(payload["source_receipts"][0]["path"])
    source_path.write_text('{"status":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source receipt sha256 mismatch"):
        prepare_correctness_calibration_declaration(manifest)


def test_calibration_rejects_semantically_incomplete_behavioral_source(
    tmp_path: Path,
) -> None:
    manifest, payload = _write_calibration_fixture(tmp_path)
    receipt = payload["source_receipts"][3]
    source_path = Path(receipt["path"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload["report"]["evidence_completeness"] = "partial"
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    receipt["sha256"] = _sha256(source_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_completeness must be complete"):
        prepare_correctness_calibration_declaration(manifest)


def test_calibration_rejects_incomplete_numerical_roles(tmp_path: Path) -> None:
    manifest, payload = _write_calibration_fixture(tmp_path)
    receipt = payload["source_receipts"][0]
    source_path = Path(receipt["path"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload["comparisons"] = source_payload["comparisons"][:1]
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    receipt["sha256"] = _sha256(source_path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repeat and canonical comparisons"):
        prepare_correctness_calibration_declaration(manifest)


def test_calibration_rejects_wrong_v1_source_role_set(tmp_path: Path) -> None:
    manifest, payload = _write_calibration_fixture(tmp_path)
    payload["source_receipts"] = payload["source_receipts"][:-1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source role set"):
        prepare_correctness_calibration_declaration(manifest)


def test_alias_effect_threshold_has_one_authoritative_location(tmp_path: Path) -> None:
    manifest, payload = _write_calibration_fixture(tmp_path)
    payload["numerical"]["frontier"][
        "alias_max_behavioral_effect_relative_error"
    ] = 0.3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="numerical.frontier has unexpected fields"):
        prepare_correctness_calibration_declaration(manifest)


def test_calibration_rejects_disagreeing_legacy_numerical_declaration(
    tmp_path: Path,
) -> None:
    manifest, _payload = _write_calibration_fixture(tmp_path)
    declaration = prepare_correctness_calibration_declaration(manifest)

    with pytest.raises(ValueError, match="disagrees with the legacy"):
        load_declared_correctness_calibration(
            declaration,
            graph_knobs={
                "correctness_numerical_manifest_path": "/tmp/other.json",
                "correctness_numerical_manifest_sha256": "sha256:" + "0" * 64,
            },
        )


def test_calibration_graph_knobs_require_path_and_hash() -> None:
    with pytest.raises(ValueError, match="manifest_path is required"):
        calibration_declaration_from_graph_knobs(
            {
                "correctness_calibration_manifest_path": None,
                "correctness_calibration_manifest_sha256": "sha256:" + "a" * 64,
            }
        )
