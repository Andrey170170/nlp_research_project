"""Frozen, provenance-bound correctness calibration declarations.

This module owns the calibration seam.  Callers receive one validated object;
manifest hashing, transitive receipt checks, numerical-reference validation, and
the sibling behavioral projection remain implementation details here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from .numerical import (
    declaration_from_graph_knobs as numerical_declaration_from_graph_knobs,
)
from .numerical import (
    file_sha256,
    prepare_numerical_manifest_declaration,
)

CORRECTNESS_CALIBRATION_SCHEMA_VERSION = 1
CORRECTNESS_CALIBRATION_FORMAT = "exact_trace_correctness_calibration_v1"
CORRECTNESS_CALIBRATION_ID = "correctness_calibration_v1"
CORRECTNESS_CALIBRATION_PATH_KNOB = "correctness_calibration_manifest_path"
CORRECTNESS_CALIBRATION_SHA256_KNOB = "correctness_calibration_manifest_sha256"
REQUIRED_NUMERICAL_SCOPES = (
    "typed_graph.repeat",
    "typed_graph.canonical",
    "feature_frontier.repeat",
)

_CALIBRATED_BUCKETS = ("feature<-error", "logit<-error")
_EXACT_BUCKETS = (
    "feature<-feature",
    "feature<-token",
    "logit<-feature",
    "logit<-token",
)
_NUMERICAL_SOURCE_KIND = "exact_trace_numerical_comparison_details_v1"
_BEHAVIORAL_SOURCE_KIND = "circuit_tracer_behavioral_faithfulness_report_v1"
_V1_SOURCE_ROLES = (
    ("step1b_r4_numerical", _NUMERICAL_SOURCE_KIND),
    ("step1b_r5_numerical", _NUMERICAL_SOURCE_KIND),
    ("step1b_r6_numerical", _NUMERICAL_SOURCE_KIND),
    ("step1b_r6_behavioral", _BEHAVIORAL_SOURCE_KIND),
)


@dataclass(frozen=True)
class LowerBoundBand:
    """A pass threshold and a weaker review threshold for a larger-is-better metric."""

    pass_min: float
    review_min: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.pass_min, label="pass_min")
        _finite_nonnegative(self.review_min, label="review_min")
        if self.review_min > self.pass_min:
            raise ValueError("lower-bound review_min must not exceed pass_min")


@dataclass(frozen=True)
class UpperBoundBand:
    """A pass threshold and a weaker review threshold for a smaller-is-better metric."""

    pass_max: float
    review_max: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.pass_max, label="pass_max")
        _finite_nonnegative(self.review_max, label="review_max")
        if self.pass_max > self.review_max:
            raise ValueError("upper-bound pass_max must not exceed review_max")


@dataclass(frozen=True)
class BucketCalibration:
    normalized_l1_deviation: UpperBoundBand
    weighted_jaccard: LowerBoundBand
    shared_sign_agreement: LowerBoundBand


@dataclass(frozen=True)
class FrontierBoundedDeltas:
    max_cutoff_score_abs_delta: float
    max_relative_cutoff_gap_abs_delta: float
    max_cutoff_tie_count_abs_delta: int
    max_record_scalar_abs_delta: float
    max_typed_edge_weight_abs_delta: float

    def __post_init__(self) -> None:
        for name in (
            "max_cutoff_score_abs_delta",
            "max_relative_cutoff_gap_abs_delta",
            "max_record_scalar_abs_delta",
            "max_typed_edge_weight_abs_delta",
        ):
            _finite_nonnegative(getattr(self, name), label=name)
        if (
            isinstance(self.max_cutoff_tie_count_abs_delta, bool)
            or not isinstance(self.max_cutoff_tie_count_abs_delta, int)
            or self.max_cutoff_tie_count_abs_delta < 0
        ):
            raise ValueError("max_cutoff_tie_count_abs_delta must be non-negative int")


@dataclass(frozen=True)
class FrontierCalibration:
    selected_recovery: LowerBoundBand
    unmatched_influence_mass_fraction: UpperBoundBand
    alias_min_decoder_cosine: float
    bounded_deltas: FrontierBoundedDeltas

    def __post_init__(self) -> None:
        if not 0.0 <= self.alias_min_decoder_cosine <= 1.0:
            raise ValueError("alias_min_decoder_cosine must be in [0, 1]")


@dataclass(frozen=True)
class NumericalCalibration:
    feature_jaccard: LowerBoundBand
    buckets: Mapping[str, BucketCalibration]
    exact_buckets: tuple[str, ...]
    frontier: FrontierCalibration

    def __post_init__(self) -> None:
        if tuple(self.buckets) != _CALIBRATED_BUCKETS:
            raise ValueError(f"buckets must be {_CALIBRATED_BUCKETS!r} in order")
        if self.exact_buckets != _EXACT_BUCKETS:
            raise ValueError(f"exact_buckets must be {_EXACT_BUCKETS!r}")


@dataclass(frozen=True)
class BehavioralCalibration:
    no_op_absolute_tolerance: float
    no_op_relative_tolerance: float
    direct_min_abs_predicted_target_delta: float
    relative_error_epsilon: float
    direct_max_mean_relative_closure: float
    direct_max_relative_closure: float
    direct_min_sign_agreement: float
    downstream_max_mean_relative_closure: float
    downstream_max_p95_relative_closure: float
    necessity_min_predicted_realized_spearman: float
    necessity_min_median_high_control_effect_ratio: float
    alias_max_relative_effect_error: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite_nonnegative(getattr(self, name), label=name)
        for name in (
            "direct_min_sign_agreement",
            "necessity_min_predicted_realized_spearman",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")


@dataclass(frozen=True)
class CalibrationSourceReceipt:
    source_id: str
    artifact_kind: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class FrozenCorrectnessCalibration:
    calibration_id: str
    policy_id: str
    manifest_path: Path
    manifest_sha256: str
    numerical_reference_declaration: Mapping[str, Any]
    required_numerical_scopes: tuple[str, ...]
    numerical: NumericalCalibration
    behavioral: BehavioralCalibration
    source_receipts: tuple[CalibrationSourceReceipt, ...]

    @property
    def calibration_fingerprint(self) -> str:
        return self.manifest_sha256

    def to_sibling_behavioral_calibration(self) -> Any:
        """Project the frozen thresholds onto the sibling verifier contract."""

        from circuit_tracer.verification import FrozenBehavioralCalibration

        return FrozenBehavioralCalibration(
            calibration_id=self.calibration_id,
            policy_id=self.policy_id,
            direct_min_abs_predicted_target_delta=(
                self.behavioral.direct_min_abs_predicted_target_delta
            ),
            relative_error_epsilon=self.behavioral.relative_error_epsilon,
            direct_max_mean_relative_closure=(
                self.behavioral.direct_max_mean_relative_closure
            ),
            direct_max_relative_closure=self.behavioral.direct_max_relative_closure,
            direct_min_sign_agreement=self.behavioral.direct_min_sign_agreement,
            downstream_max_mean_relative_closure=(
                self.behavioral.downstream_max_mean_relative_closure
            ),
            downstream_max_p95_relative_closure=(
                self.behavioral.downstream_max_p95_relative_closure
            ),
            necessity_min_predicted_realized_spearman=(
                self.behavioral.necessity_min_predicted_realized_spearman
            ),
            necessity_min_median_high_control_effect_ratio=(
                self.behavioral.necessity_min_median_high_control_effect_ratio
            ),
            alias_max_relative_effect_error=(
                self.behavioral.alias_max_relative_effect_error
            ),
        )

    def behavioral_probe_policy_kwargs(self) -> dict[str, Any]:
        """Return the complete calibrated projection for BehavioralProbePolicy."""

        return {
            "calibration": self.to_sibling_behavioral_calibration(),
            "no_op_absolute_tolerance": self.behavioral.no_op_absolute_tolerance,
            "no_op_relative_tolerance": self.behavioral.no_op_relative_tolerance,
        }


def prepare_correctness_calibration_declaration(
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate and hash one calibration plus every transitive receipt."""

    path = Path(manifest_path).resolve()
    manifest_sha256 = file_sha256(path)
    load_frozen_correctness_calibration(path, expected_sha256=manifest_sha256)
    return {
        "schema_version": CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
        "format": CORRECTNESS_CALIBRATION_FORMAT,
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
    }


def calibration_declaration_from_graph_knobs(
    knobs: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = knobs.get(CORRECTNESS_CALIBRATION_PATH_KNOB)
    sha256 = knobs.get(CORRECTNESS_CALIBRATION_SHA256_KNOB)
    if path is None and sha256 is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError(f"graph_knobs.{CORRECTNESS_CALIBRATION_PATH_KNOB} is required")
    _require_sha256(sha256, label=f"graph_knobs.{CORRECTNESS_CALIBRATION_SHA256_KNOB}")
    return {
        "schema_version": CORRECTNESS_CALIBRATION_SCHEMA_VERSION,
        "format": CORRECTNESS_CALIBRATION_FORMAT,
        "manifest_path": path,
        "manifest_sha256": sha256,
    }


def load_declared_correctness_calibration(
    declaration: Mapping[str, Any],
    *,
    graph_knobs: Mapping[str, Any] | None = None,
) -> FrozenCorrectnessCalibration:
    """Revalidate a graph-knob declaration and its complete receipt closure."""

    _validate_declaration(declaration)
    calibration = load_frozen_correctness_calibration(
        Path(str(declaration["manifest_path"])),
        expected_sha256=str(declaration["manifest_sha256"]),
    )
    if graph_knobs is not None:
        legacy = numerical_declaration_from_graph_knobs(graph_knobs)
        if legacy is not None and dict(legacy) != dict(
            calibration.numerical_reference_declaration
        ):
            raise ValueError(
                "correctness calibration numerical declaration disagrees with "
                "the legacy numerical manifest declaration"
            )
    return calibration


def load_frozen_correctness_calibration(
    path: Path,
    *,
    expected_sha256: str,
) -> FrozenCorrectnessCalibration:
    """Strictly load the v1 calibration and validate all referenced artifacts."""

    _require_sha256(expected_sha256, label="manifest_sha256")
    resolved = Path(path).resolve()
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(
            "correctness calibration manifest sha256 mismatch: "
            f"expected {expected_sha256}, observed {actual}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _exact_fields(
        payload,
        (
            "schema_version",
            "format",
            "calibration_id",
            "policy_id",
            "numerical_reference",
            "required_numerical_scopes",
            "numerical",
            "behavioral",
            "source_receipts",
        ),
        label="correctness calibration manifest",
    )
    if payload["schema_version"] != CORRECTNESS_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported correctness calibration schema_version")
    if payload["format"] != CORRECTNESS_CALIBRATION_FORMAT:
        raise ValueError("unsupported correctness calibration format")
    calibration_id = _nonempty_str(payload["calibration_id"], label="calibration_id")
    if calibration_id != CORRECTNESS_CALIBRATION_ID:
        raise ValueError(
            f"calibration_id must be {CORRECTNESS_CALIBRATION_ID!r} for v1"
        )
    policy_id = _nonempty_str(payload["policy_id"], label="policy_id")
    if not policy_id.rsplit("_v", 1)[-1].isdigit():
        raise ValueError("policy_id must end in a numeric _vN version")

    numerical_reference = _mapping(
        payload["numerical_reference"], label="numerical_reference"
    )
    _exact_fields(
        numerical_reference,
        ("schema_version", "format", "manifest_path", "manifest_sha256"),
        label="numerical_reference",
    )
    prepared_reference = prepare_numerical_manifest_declaration(
        Path(
            _absolute_path(
                numerical_reference["manifest_path"],
                label="numerical_reference.manifest_path",
            )
        )
    )
    if dict(numerical_reference) != prepared_reference:
        raise ValueError("frozen numerical reference declaration does not revalidate")

    scopes_raw = payload["required_numerical_scopes"]
    if (
        not isinstance(scopes_raw, list)
        or tuple(scopes_raw) != REQUIRED_NUMERICAL_SCOPES
    ):
        raise ValueError(
            "required_numerical_scopes must declare typed repeat/canonical and "
            "repeat frontier in canonical order"
        )
    numerical = _parse_numerical(_mapping(payload["numerical"], label="numerical"))
    behavioral = _parse_behavioral(_mapping(payload["behavioral"], label="behavioral"))
    source_receipts = _parse_source_receipts(payload["source_receipts"])
    return FrozenCorrectnessCalibration(
        calibration_id=calibration_id,
        policy_id=policy_id,
        manifest_path=resolved,
        manifest_sha256=actual,
        numerical_reference_declaration=MappingProxyType(prepared_reference),
        required_numerical_scopes=REQUIRED_NUMERICAL_SCOPES,
        numerical=numerical,
        behavioral=behavioral,
        source_receipts=source_receipts,
    )


def _parse_numerical(value: Mapping[str, Any]) -> NumericalCalibration:
    _exact_fields(
        value,
        ("feature_jaccard", "buckets", "exact_buckets", "frontier"),
        label="numerical",
    )
    buckets_raw = _mapping(value["buckets"], label="numerical.buckets")
    if tuple(buckets_raw) != _CALIBRATED_BUCKETS:
        raise ValueError(f"numerical.buckets must be {_CALIBRATED_BUCKETS!r} in order")
    buckets = {
        name: _parse_bucket(
            _mapping(buckets_raw[name], label=f"numerical.buckets.{name}")
        )
        for name in _CALIBRATED_BUCKETS
    }
    exact_buckets = value["exact_buckets"]
    if not isinstance(exact_buckets, list) or tuple(exact_buckets) != _EXACT_BUCKETS:
        raise ValueError(f"numerical.exact_buckets must be {_EXACT_BUCKETS!r}")
    frontier = _mapping(value["frontier"], label="numerical.frontier")
    _exact_fields(
        frontier,
        (
            "selected_recovery",
            "unmatched_influence_mass_fraction",
            "alias_min_decoder_cosine",
            "bounded_deltas",
        ),
        label="numerical.frontier",
    )
    bounded = _mapping(frontier["bounded_deltas"], label="bounded_deltas")
    _exact_fields(
        bounded,
        tuple(FrontierBoundedDeltas.__dataclass_fields__),
        label="numerical.frontier.bounded_deltas",
    )
    return NumericalCalibration(
        feature_jaccard=_lower_band(value["feature_jaccard"], label="feature_jaccard"),
        buckets=MappingProxyType(buckets),
        exact_buckets=_EXACT_BUCKETS,
        frontier=FrontierCalibration(
            selected_recovery=_lower_band(
                frontier["selected_recovery"], label="selected_recovery"
            ),
            unmatched_influence_mass_fraction=_upper_band(
                frontier["unmatched_influence_mass_fraction"],
                label="unmatched_influence_mass_fraction",
            ),
            alias_min_decoder_cosine=_number(
                frontier["alias_min_decoder_cosine"],
                label="alias_min_decoder_cosine",
            ),
            bounded_deltas=FrontierBoundedDeltas(**dict(bounded)),
        ),
    )


def _parse_bucket(value: Mapping[str, Any]) -> BucketCalibration:
    _exact_fields(
        value,
        ("normalized_l1_deviation", "weighted_jaccard", "shared_sign_agreement"),
        label="numerical bucket",
    )
    return BucketCalibration(
        normalized_l1_deviation=_upper_band(
            value["normalized_l1_deviation"], label="normalized_l1_deviation"
        ),
        weighted_jaccard=_lower_band(
            value["weighted_jaccard"], label="weighted_jaccard"
        ),
        shared_sign_agreement=_lower_band(
            value["shared_sign_agreement"], label="shared_sign_agreement"
        ),
    )


def _parse_behavioral(value: Mapping[str, Any]) -> BehavioralCalibration:
    fields = tuple(BehavioralCalibration.__dataclass_fields__)
    _exact_fields(value, fields, label="behavioral")
    return BehavioralCalibration(
        **{name: _number(value[name], label=name) for name in fields}
    )


def _parse_source_receipts(value: object) -> tuple[CalibrationSourceReceipt, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_receipts must be a non-empty list")
    receipts: list[CalibrationSourceReceipt] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        row = _mapping(raw, label=f"source_receipts[{index}]")
        _exact_fields(
            row,
            ("source_id", "artifact_kind", "path", "sha256"),
            label=f"source_receipts[{index}]",
        )
        source_id = _nonempty_str(row["source_id"], label="source_id")
        if source_id in seen:
            raise ValueError(f"duplicate calibration source_id {source_id!r}")
        artifact_kind = _nonempty_str(row["artifact_kind"], label="artifact_kind")
        path = Path(_absolute_path(row["path"], label="source receipt path"))
        expected = row["sha256"]
        _require_sha256(expected, label="source receipt sha256")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"calibration source receipt sha256 mismatch for {source_id!r}: "
                f"expected {expected}, observed {observed}"
            )
        _validate_source_artifact(
            path,
            source_id=source_id,
            artifact_kind=artifact_kind,
        )
        receipts.append(
            CalibrationSourceReceipt(source_id, artifact_kind, path, str(expected))
        )
        seen.add(source_id)
    observed_roles = tuple(
        (receipt.source_id, receipt.artifact_kind) for receipt in receipts
    )
    if observed_roles != _V1_SOURCE_ROLES:
        raise ValueError(
            "correctness_calibration_v1 source role set must be r4/r5/r6 "
            "numerical plus r6 behavioral in canonical order"
        )
    return tuple(receipts)


def _validate_source_artifact(
    path: Path,
    *,
    source_id: str,
    artifact_kind: str,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"calibration source {source_id!r} must be valid JSON"
        ) from exc
    document = _mapping(payload, label=f"calibration source {source_id!r}")
    if artifact_kind == _NUMERICAL_SOURCE_KIND:
        _validate_numerical_source(document, source_id=source_id)
        return
    if artifact_kind == _BEHAVIORAL_SOURCE_KIND:
        _validate_behavioral_source(document, source_id=source_id)
        return
    raise ValueError(
        f"calibration source {source_id!r} has unsupported artifact_kind "
        f"{artifact_kind!r}"
    )


def _validate_numerical_source(
    document: Mapping[str, Any],
    *,
    source_id: str,
) -> None:
    _exact_fields(
        document,
        (
            "schema_version",
            "format",
            "manifest_path",
            "manifest_sha256",
            "candidate_graph_path",
            "candidate_graph_sha256",
            "comparisons",
            "frontier_comparisons",
        ),
        label=f"calibration source {source_id!r}",
    )
    if document["schema_version"] != 1:
        raise ValueError(
            f"calibration source {source_id!r} has unsupported schema_version"
        )
    if document["format"] != _NUMERICAL_SOURCE_KIND:
        raise ValueError(f"calibration source {source_id!r} format mismatch")
    _absolute_path(document["manifest_path"], label=f"{source_id}.manifest_path")
    _require_sha256(document["manifest_sha256"], label=f"{source_id}.manifest_sha256")
    _absolute_path(
        document["candidate_graph_path"], label=f"{source_id}.candidate_graph_path"
    )
    _require_sha256(
        document["candidate_graph_sha256"],
        label=f"{source_id}.candidate_graph_sha256",
    )
    comparisons = document["comparisons"]
    if not isinstance(comparisons, list):
        raise ValueError(f"calibration source {source_id!r} comparisons must be a list")
    roles: list[str] = []
    for index, raw in enumerate(comparisons):
        row = _mapping(raw, label=f"{source_id}.comparisons[{index}]")
        role = _nonempty_str(row.get("role"), label=f"{source_id}.comparison role")
        roles.append(role)
        if row.get("scope_compatible") is not True:
            raise ValueError(
                f"calibration source {source_id!r} comparison {role!r} "
                "must be scope-compatible"
            )
        if not isinstance(row.get("comparison_exact"), bool):
            raise ValueError(
                f"calibration source {source_id!r} comparison_exact must be boolean"
            )
        comparison = _mapping(
            row.get("comparison"), label=f"{source_id}.{role}.comparison"
        )
        _validate_calibration_metrics(
            comparison,
            source_id=source_id,
            role=role,
        )
    if tuple(roles) != ("repeat", "canonical"):
        raise ValueError(
            f"calibration source {source_id!r} must contain complete repeat and "
            "canonical comparisons in order"
        )
    frontier_comparisons = document["frontier_comparisons"]
    if not isinstance(frontier_comparisons, list):
        raise ValueError(
            f"calibration source {source_id!r} frontier_comparisons must be a list"
        )
    for index, raw in enumerate(frontier_comparisons):
        row = _mapping(raw, label=f"{source_id}.frontier_comparisons[{index}]")
        if row.get("role") not in {"repeat", "canonical"}:
            raise ValueError(
                f"calibration source {source_id!r} frontier role is unsupported"
            )
        if row.get("status") != "compared":
            raise ValueError(
                f"calibration source {source_id!r} frontier comparison must be terminal"
            )
        _mapping(row.get("metrics"), label=f"{source_id}.frontier metrics")


def _validate_calibration_metrics(
    comparison: Mapping[str, Any],
    *,
    source_id: str,
    role: str,
) -> None:
    required_metrics = ["feature_jaccard"]
    for bucket in ("feature_error", "logit_error"):
        required_metrics.extend(
            (
                f"bucket_{bucket}_normalized_l1_deviation",
                f"bucket_{bucket}_weighted_jaccard",
                f"bucket_{bucket}_shared_sign_agreement",
            )
        )
    required_metrics.extend(
        f"bucket_{bucket.replace('<-', '_')}_exact" for bucket in _EXACT_BUCKETS
    )
    for metric in required_metrics:
        if metric not in comparison:
            raise ValueError(
                f"calibration source {source_id!r} {role!r} comparison is missing "
                f"metric {metric!r}"
            )
        _number(comparison[metric], label=f"{source_id}.{role}.{metric}")


def _validate_behavioral_source(
    document: Mapping[str, Any],
    *,
    source_id: str,
) -> None:
    _exact_fields(
        document,
        ("schema", "schema_version", "evidence_fingerprint", "report"),
        label=f"calibration source {source_id!r}",
    )
    if document["schema"] != "behavioral_faithfulness_report":
        raise ValueError(f"calibration source {source_id!r} schema mismatch")
    if document["schema_version"] != 1:
        raise ValueError(
            f"calibration source {source_id!r} has unsupported schema_version"
        )
    _nonempty_str(
        document["evidence_fingerprint"],
        label=f"{source_id}.evidence_fingerprint",
    )
    report = _mapping(document["report"], label=f"{source_id}.report")
    if report.get("evidence_completeness") != "complete":
        raise ValueError(
            f"calibration source {source_id!r} evidence_completeness must be complete"
        )
    if report.get("runtime_status") != "complete":
        raise ValueError(
            f"calibration source {source_id!r} runtime_status must be complete"
        )
    if report.get("verdict") not in {
        "supported",
        "contradicted",
        "inconclusive",
        "unknown",
    }:
        raise ValueError(f"calibration source {source_id!r} verdict must be terminal")
    planned = _integer(report.get("variants_planned"), label="variants_planned")
    completed = _integer(
        report.get("variants_completed"), label="variants_completed"
    )
    if completed != planned:
        raise ValueError(
            f"calibration source {source_id!r} must complete every planned variant"
        )
    if report.get("no_op_required") is not True or report.get("no_op_passed") is not True:
        raise ValueError(
            f"calibration source {source_id!r} must contain a passing required no-op"
        )
    raw_execution = _mapping(
        report.get("raw_execution"), label=f"{source_id}.raw_execution"
    )
    if raw_execution.get("status") != "complete":
        raise ValueError(
            f"calibration source {source_id!r} raw execution must be complete"
        )
    if raw_execution.get("cleanup_completed") is not True:
        raise ValueError(
            f"calibration source {source_id!r} cleanup must be complete"
        )


def _lower_band(value: object, *, label: str) -> LowerBoundBand:
    row = _mapping(value, label=label)
    _exact_fields(row, ("pass_min", "review_min"), label=label)
    return LowerBoundBand(
        pass_min=_number(row["pass_min"], label=f"{label}.pass_min"),
        review_min=_number(row["review_min"], label=f"{label}.review_min"),
    )


def _upper_band(value: object, *, label: str) -> UpperBoundBand:
    row = _mapping(value, label=label)
    _exact_fields(row, ("pass_max", "review_max"), label=label)
    return UpperBoundBand(
        pass_max=_number(row["pass_max"], label=f"{label}.pass_max"),
        review_max=_number(row["review_max"], label=f"{label}.review_max"),
    )


def _validate_declaration(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        ("schema_version", "format", "manifest_path", "manifest_sha256"),
        label="correctness calibration declaration",
    )
    if value["schema_version"] != CORRECTNESS_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("correctness calibration declaration schema_version mismatch")
    if value["format"] != CORRECTNESS_CALIBRATION_FORMAT:
        raise ValueError("correctness calibration declaration format mismatch")
    _absolute_path(value["manifest_path"], label="manifest_path")
    _require_sha256(value["manifest_sha256"], label="manifest_sha256")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _exact_fields(value: object, fields: tuple[str, ...], *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} has unexpected fields")


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> None:
    number = _number(value, label=label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")


def _nonempty_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _absolute_path(value: object, *, label: str) -> str:
    text = _nonempty_str(value, label=label)
    if not Path(text).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return text


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a sha256 fingerprint")
