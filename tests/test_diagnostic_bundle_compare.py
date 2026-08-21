from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlp_research_project.exact_trace_bench.cli import build_parser
from nlp_research_project.exact_trace_bench.diagnostic_bundle_compare import (
    DiagnosticBundleError,
    compare_diagnostic_bundle_dirs,
)


def _write_bundles(token_dir: Path, *, activation_delta: float = 0.0) -> None:
    token_dir.mkdir(parents=True)
    features = np.asarray([[0, 0, 1], [1, 2, 3]], dtype=np.int64)
    activations = np.asarray([1.0 + activation_delta, 0.5], dtype=np.float32)
    target_ids = np.asarray([9], dtype=np.int64)
    target_probabilities = np.asarray([0.25], dtype=np.float32)
    np.savez_compressed(
        token_dir / "phase0_donor_bundle.npz",
        input_tokens=np.asarray([1, 2, 3], dtype=np.int64),
        target_token_ids=target_ids,
        target_probabilities=target_probabilities,
        target_logits=np.asarray([2.0], dtype=np.float32),
        active_features=features,
        activation_values=activations,
        activation_values_raw_uint16=np.asarray([], dtype=np.uint16),
        status=np.asarray("captured"),
    )
    np.savez_compressed(
        token_dir / "phase3_gradient_bundle.npz",
        target_token_ids=target_ids,
        target_probabilities=target_probabilities,
        gradients=np.asarray([1.0, np.nan, np.inf], dtype=np.float32),
        gradient_hash=np.asarray("fixture-gradient"),
    )
    np.savez_compressed(
        token_dir / "phase3_row_bundle.npz",
        target_token_ids=target_ids,
        target_probabilities=target_probabilities,
        phase3_feature_rows=np.asarray([[0.75, -0.25]], dtype=np.float32),
        feature_abs_sums=np.asarray([1.0], dtype=np.float64),
        error_abs_sums=np.asarray([0.5], dtype=np.float64),
        token_abs_sums=np.asarray([0.25], dtype=np.float64),
        row_abs_sums=np.asarray([1.75], dtype=np.float64),
        row_hash=np.asarray("fixture-row"),
    )
    np.savez_compressed(
        token_dir / "phase3_seed_bundle.npz",
        active_features=features,
        activation_values=activations,
        seed_feature_influences=np.asarray([0.75, 0.25], dtype=np.float64),
        frontier_pre_locality=np.asarray([0, 1], dtype=np.int64),
        frontier_post_locality=np.asarray([0, 1], dtype=np.int64),
        status=np.asarray("captured"),
    )


def test_compare_diagnostic_bundles_reports_exact_arrays_and_neutral_boundary(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)

    result = compare_diagnostic_bundle_dirs(left, right)

    assert result["required_artifacts_complete"] is True
    assert result["summary"]["all_arrays_raw_exact"] is True
    assert result["earliest_divergence"]["classification"] == "none_exact"
    assert result["claim_boundary"]["source"] == "neutral_default"
    gradients = result["bundles"]["phase3_gradient"]["arrays"]["gradients"]
    assert gradients["left"]["shape"] == [3]
    assert gradients["left"]["dtype"] == "float32"
    assert len(gradients["left"]["sha256"]) == 64
    assert gradients["numeric"]["nonfinite_pair_count"] == 2
    assert gradients["numeric"]["nonfinite_mismatch_count"] == 0


def test_compare_diagnostic_bundles_localizes_phase0_before_phase3(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right, activation_delta=0.125)
    with np.load(right / "phase3_gradient_bundle.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["gradients"] = np.asarray([1.5, np.nan, np.inf], dtype=np.float32)
    np.savez_compressed(right / "phase3_gradient_bundle.npz", **payload)

    result = compare_diagnostic_bundle_dirs(
        left,
        right,
        claim_boundary="Only the forward graph width differs between these arms.",
    )

    assert result["claim_boundary"]["source"] == "caller_supplied"
    assert result["earliest_divergence"]["classification"] == "phase0_activation_values"
    comparison = result["bundles"]["phase0_donor"]["arrays"]["activation_values"]
    assert comparison["numeric"]["max_absolute_error"] == pytest.approx(0.125)
    assert comparison["numeric"]["max_error_index"] == [0]


def test_compare_diagnostic_bundles_localizes_gradient_and_numeric_error(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    with np.load(right / "phase3_gradient_bundle.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["gradients"] = np.asarray([1.5, np.nan, np.inf], dtype=np.float32)
    np.savez_compressed(right / "phase3_gradient_bundle.npz", **payload)

    result = compare_diagnostic_bundle_dirs(left, right)

    assert result["earliest_divergence"]["classification"] == "phase3_gradients"
    metrics = result["bundles"]["phase3_gradient"]["arrays"]["gradients"]["numeric"]
    assert metrics["finite_pair_count"] == 1
    assert metrics["max_absolute_error"] == pytest.approx(0.5)
    assert metrics["left_reference_normalized_l1"] == pytest.approx(0.5)
    assert metrics["allclose"] is False


def test_compare_diagnostic_bundles_classifies_target_state_after_phase0(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    with np.load(right / "phase0_donor_bundle.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["target_logits"] = np.asarray([2.25], dtype=np.float32)
    np.savez_compressed(right / "phase0_donor_bundle.npz", **payload)

    result = compare_diagnostic_bundle_dirs(left, right)

    assert result["earliest_divergence"]["classification"] == "phase1_target_state"
    assert result["earliest_divergence"]["array"] == "target_logits"


@pytest.mark.parametrize(
    ("array_name", "classification"),
    [
        ("feature_abs_sums", "phase3_feature_column_l1_contribution"),
        ("error_abs_sums", "phase3_error_column_l1_contribution"),
        ("token_abs_sums", "phase3_token_column_l1_contribution"),
        ("row_abs_sums", "phase3_total_row_l1_denominator"),
    ],
)
def test_compare_diagnostic_bundles_localizes_phase3_row_l1_components(
    tmp_path: Path,
    array_name: str,
    classification: str,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    with np.load(right / "phase3_row_bundle.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload[array_name] = payload[array_name] + np.asarray([0.125])
    np.savez_compressed(right / "phase3_row_bundle.npz", **payload)

    result = compare_diagnostic_bundle_dirs(left, right)

    assert result["earliest_divergence"] == {
        "classification": classification,
        "bundle": "phase3_row",
        "array": array_name,
        "basis": "first raw-exact mismatch in declared checkpoint order",
    }


@pytest.mark.parametrize(
    ("array_name", "left_values", "right_values"),
    [
        (
            "phase3_error_rows_by_layer",
            np.asarray([[[0.5, -0.25], [0.125, 0.0]]], dtype=np.float32),
            np.asarray([[[0.5, -0.25], [0.25, 0.0]]], dtype=np.float32),
        ),
        (
            "phase3_token_rows",
            np.asarray([[0.25, -0.125]], dtype=np.float32),
            np.asarray([[0.25, -0.25]], dtype=np.float32),
        ),
    ],
)
def test_compare_diagnostic_bundles_localizes_raw_nonfeature_rows_before_summaries(
    tmp_path: Path,
    array_name: str,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    shared_error_rows = np.asarray([[[0.5, -0.25], [0.125, 0.0]]], dtype=np.float32)
    shared_token_rows = np.asarray([[0.25, -0.125]], dtype=np.float32)
    for token_dir, values in ((left, left_values), (right, right_values)):
        with np.load(
            token_dir / "phase3_row_bundle.npz", allow_pickle=False
        ) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["phase3_error_rows_by_layer"] = (
            values if array_name == "phase3_error_rows_by_layer" else shared_error_rows
        )
        payload["phase3_token_rows"] = (
            values if array_name == "phase3_token_rows" else shared_token_rows
        )
        # Also vary the derived summary to prove the raw evidence is ordered first.
        payload["error_abs_sums"] = np.abs(payload["phase3_error_rows_by_layer"]).sum(
            axis=(1, 2), dtype=np.float64
        )
        payload["token_abs_sums"] = np.abs(payload["phase3_token_rows"]).sum(
            axis=1, dtype=np.float64
        )
        payload["row_abs_sums"] = (
            payload["feature_abs_sums"]
            + payload["error_abs_sums"]
            + payload["token_abs_sums"]
        )
        np.savez_compressed(token_dir / "phase3_row_bundle.npz", **payload)

    result = compare_diagnostic_bundle_dirs(left, right)

    expected_classification = {
        "phase3_error_rows_by_layer": ("phase3_error_column_signed_rows_by_layer"),
        "phase3_token_rows": "phase3_token_column_signed_rows",
    }[array_name]
    assert result["earliest_divergence"]["classification"] == (expected_classification)
    assert result["earliest_divergence"]["array"] == array_name


def test_compare_diagnostic_bundles_checks_denominator_before_seed_influence(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    with np.load(right / "phase3_row_bundle.npz", allow_pickle=False) as archive:
        row_payload = {key: archive[key] for key in archive.files}
    row_payload["error_abs_sums"] = np.asarray([0.5001], dtype=np.float64)
    row_payload["row_abs_sums"] = np.asarray([1.7501], dtype=np.float64)
    np.savez_compressed(right / "phase3_row_bundle.npz", **row_payload)
    with np.load(right / "phase3_seed_bundle.npz", allow_pickle=False) as archive:
        seed_payload = {key: archive[key] for key in archive.files}
    seed_payload["seed_feature_influences"] = np.asarray(
        [0.7499, 0.2499], dtype=np.float64
    )
    np.savez_compressed(right / "phase3_seed_bundle.npz", **seed_payload)

    result = compare_diagnostic_bundle_dirs(left, right)

    assert result["earliest_divergence"]["classification"] == (
        "phase3_error_column_l1_contribution"
    )
    assert result["earliest_divergence"]["array"] == "error_abs_sums"


def test_compare_diagnostic_bundles_requires_phase3_row_l1_arrays(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    with np.load(right / "phase3_row_bundle.npz", allow_pickle=False) as archive:
        payload = {
            key: archive[key] for key in archive.files if key != "error_abs_sums"
        }
    np.savez_compressed(right / "phase3_row_bundle.npz", **payload)

    with pytest.raises(
        DiagnosticBundleError,
        match=r"phase3_row_bundle\.npz:error_abs_sums",
    ):
        compare_diagnostic_bundle_dirs(left, right)


def test_compare_diagnostic_bundles_fails_closed_on_missing_bundle(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_bundles(left)
    _write_bundles(right)
    (right / "phase3_row_bundle.npz").unlink()

    with pytest.raises(DiagnosticBundleError, match="phase3_row_bundle.npz"):
        compare_diagnostic_bundle_dirs(left, right)


def test_compare_diagnostic_bundles_cli_writes_json(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison.json"
    _write_bundles(left)
    _write_bundles(right)
    args = build_parser().parse_args(
        [
            "compare-diagnostic-bundles",
            str(left),
            str(right),
            "--output-json",
            str(output),
            "--claim-boundary",
            "The pair differs only in the VJP kernel.",
        ]
    )

    args.func(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["claim_boundary"]["source"] == "caller_supplied"
    assert payload["summary"]["all_arrays_raw_exact"] is True
