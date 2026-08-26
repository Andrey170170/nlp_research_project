"""Production and strict persistence of bounded frontier evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    TypedCompactGraph,
)

from .frontier import (
    FeatureFeatureEdge,
    FeatureKey,
    FrontierBounds,
    FrontierEvidence,
    FrontierRecord,
    LogitFeatureEdge,
    build_frontier_evidence,
)

LEGACY_FRONTIER_ARTIFACT_SCHEMA_VERSION = 1
LEGACY_FRONTIER_ARTIFACT_FORMAT = "bounded_frontier_evidence_v1"
FRONTIER_ARTIFACT_SCHEMA_VERSION = 2
FRONTIER_ARTIFACT_FORMAT = "bounded_frontier_evidence_v2"
CUTOFF_BASIS = "captured_seed_selection_cutoff_v1"
FALLBACK_DESCRIPTOR_KIND = "fallback_identity_metadata_v1"
DECODER_DESCRIPTOR_KIND = "active_decoder_countsketch_v1"
DECODER_DESCRIPTOR_SCOPE = "active_occurrence_downstream_decoder_rows_v1"
_SEMANTIC_DESCRIPTOR_TRANSIENT_POLICY_ID = "bounded_seed_frontier_handoff_v1"
_SEMANTIC_DESCRIPTOR_TRANSIENT_MAX_BYTES = 64 * 1024 * 1024
_SEMANTIC_DESCRIPTOR_TRANSIENT_ARRAY_COUNT = 3
_SEMANTIC_DESCRIPTOR_TRANSIENT_RECEIPT_FIELDS = (
    "semantic_descriptor_transient_policy_id",
    "semantic_descriptor_transient_max_bytes",
    "semantic_descriptor_transient_required_bytes",
    "semantic_descriptor_transient_admitted",
    "semantic_descriptor_transient_released",
    "semantic_descriptor_transient_array_count",
)


@dataclass(frozen=True)
class BoundedFrontierArtifact:
    evidence: FrontierEvidence
    descriptor_kind: str
    descriptor_version: str
    cutoff_basis: str
    max_typed_neighborhood_edges: int
    content_fingerprint: str
    descriptor_is_decoder_evidence: bool = True
    descriptor_scope: str | None = None
    decoder_source_fingerprint: str | None = None
    decoder_evidence_fingerprint: str | None = None
    projection_fingerprint: str | None = None
    schema_version: int = FRONTIER_ARTIFACT_SCHEMA_VERSION
    artifact_format: str = FRONTIER_ARTIFACT_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_FRONTIER_ARTIFACT_SCHEMA_VERSION,
            FRONTIER_ARTIFACT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported frontier artifact schema_version")
        expected_format = (
            LEGACY_FRONTIER_ARTIFACT_FORMAT
            if self.schema_version == LEGACY_FRONTIER_ARTIFACT_SCHEMA_VERSION
            else FRONTIER_ARTIFACT_FORMAT
        )
        if self.artifact_format != expected_format:
            raise ValueError("frontier artifact format does not match its schema")
        if not self.descriptor_kind or not self.descriptor_version:
            raise ValueError("descriptor kind and version are required")
        if self.cutoff_basis != CUTOFF_BASIS:
            raise ValueError("unsupported frontier cutoff basis")
        if self.max_typed_neighborhood_edges <= 0:
            raise ValueError("max_typed_neighborhood_edges must be positive")
        _require_fingerprint("content_fingerprint", self.content_fingerprint)
        if self.schema_version == LEGACY_FRONTIER_ARTIFACT_SCHEMA_VERSION:
            if self.descriptor_is_decoder_evidence or any(
                value is not None
                for value in (
                    self.descriptor_scope,
                    self.decoder_source_fingerprint,
                    self.decoder_evidence_fingerprint,
                    self.projection_fingerprint,
                )
            ):
                raise ValueError("legacy frontier artifacts cannot claim decoder evidence")
            return
        if not self.descriptor_is_decoder_evidence:
            raise ValueError("v2 frontier artifacts require decoder evidence")
        if self.descriptor_scope != DECODER_DESCRIPTOR_SCOPE:
            raise ValueError("unsupported frontier decoder descriptor scope")
        for field, value in (
            ("decoder_source_fingerprint", self.decoder_source_fingerprint),
            ("decoder_evidence_fingerprint", self.decoder_evidence_fingerprint),
            ("projection_fingerprint", self.projection_fingerprint),
        ):
            if value is None:
                raise ValueError(f"{field} is required")
            _require_fingerprint(field, value)
        if self.decoder_source_fingerprint != self.evidence.decoder_fingerprint:
            raise ValueError("frontier decoder source does not match selection evidence")

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "artifact_format": self.artifact_format,
            "content_fingerprint": self.content_fingerprint,
            "descriptor_kind": self.descriptor_kind,
            "descriptor_version": self.descriptor_version,
            "descriptor_is_decoder_evidence": self.descriptor_is_decoder_evidence,
            "cutoff_basis": self.cutoff_basis,
            "max_typed_neighborhood_edges": self.max_typed_neighborhood_edges,
            "evidence": _evidence_to_json(self.evidence),
        }
        if self.schema_version == FRONTIER_ARTIFACT_SCHEMA_VERSION:
            payload.update(
                {
                    "descriptor_scope": self.descriptor_scope,
                    "decoder_source_fingerprint": self.decoder_source_fingerprint,
                    "decoder_evidence_fingerprint": self.decoder_evidence_fingerprint,
                    "projection_fingerprint": self.projection_fingerprint,
                }
            )
        return payload

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:
        schema_version = _integer(value.get("schema_version"), "schema_version")
        v2_fields: tuple[str, ...] = ()
        if schema_version == FRONTIER_ARTIFACT_SCHEMA_VERSION:
            v2_fields = (
                "descriptor_scope",
                "decoder_source_fingerprint",
                "decoder_evidence_fingerprint",
                "projection_fingerprint",
            )
        _require_fields(
            value,
            (
                "schema_version",
                "artifact_format",
                "content_fingerprint",
                "descriptor_kind",
                "descriptor_version",
                "descriptor_is_decoder_evidence",
                "cutoff_basis",
                "max_typed_neighborhood_edges",
                "evidence",
                *v2_fields,
            ),
            label="frontier artifact",
        )
        is_decoder_evidence = value["descriptor_is_decoder_evidence"]
        if not isinstance(is_decoder_evidence, bool):
            raise ValueError("descriptor_is_decoder_evidence must be a bool")
        extra: dict[str, Any] = {}
        if schema_version == FRONTIER_ARTIFACT_SCHEMA_VERSION:
            extra = {
                "descriptor_scope": _string(value["descriptor_scope"], "descriptor_scope"),
                "decoder_source_fingerprint": _string(
                    value["decoder_source_fingerprint"], "decoder_source_fingerprint"
                ),
                "decoder_evidence_fingerprint": _string(
                    value["decoder_evidence_fingerprint"], "decoder_evidence_fingerprint"
                ),
                "projection_fingerprint": _string(
                    value["projection_fingerprint"], "projection_fingerprint"
                ),
            }
        artifact = cls(
            schema_version=schema_version,
            artifact_format=_string(value["artifact_format"], "artifact_format"),
            content_fingerprint=_string(
                value["content_fingerprint"], "content_fingerprint"
            ),
            descriptor_kind=_string(value["descriptor_kind"], "descriptor_kind"),
            descriptor_version=_string(
                value["descriptor_version"], "descriptor_version"
            ),
            cutoff_basis=_string(value["cutoff_basis"], "cutoff_basis"),
            max_typed_neighborhood_edges=_integer(
                value["max_typed_neighborhood_edges"],
                "max_typed_neighborhood_edges",
            ),
            evidence=_evidence_from_json(
                _mapping(value["evidence"], "evidence")
            ),
            descriptor_is_decoder_evidence=is_decoder_evidence,
            **extra,
        )
        expected = _content_fingerprint(artifact)
        if artifact.content_fingerprint != expected:
            raise ValueError("frontier artifact content fingerprint mismatch")
        return artifact


def build_bounded_frontier_artifact(
    *,
    graph: TypedCompactGraph,
    compact_result: Mapping[str, Any],
    feature_semantic_descriptors: Mapping[str, Any],
    decoder_fingerprint: str,
    max_near_cutoff: int,
    max_typed_neighborhood_edges: int = 32,
) -> BoundedFrontierArtifact:
    """Build bounded evidence from one strictly reopened typed graph."""

    if graph.path is None or not graph.path.is_file():
        raise ValueError("frontier production requires a strictly reopened graph")
    _require_fingerprint("decoder_fingerprint", decoder_fingerprint)
    if max_near_cutoff < 0:
        raise ValueError("max_near_cutoff must be non-negative")
    if max_typed_neighborhood_edges <= 0:
        raise ValueError("max_typed_neighborhood_edges must be positive")

    active = _integer_array(compact_result, "active_features", ndim=2)
    if active.shape[1:] != (3,):
        raise ValueError("active_features must have shape (N, 3)")
    selected_indices = _integer_array(compact_result, "selected_features", ndim=1)
    _require_unique_indices(selected_indices, len(active), "selected_features")
    selected_features = active[selected_indices]
    if not np.array_equal(selected_features, graph.feature_ids):
        raise ValueError("compact selected features do not match reopened graph")

    descriptors = _DescriptorTable.from_payload(
        feature_semantic_descriptors,
        active_features=active,
    )
    if descriptors.decoder_source_fingerprint != decoder_fingerprint:
        raise ValueError("descriptor decoder source fingerprint mismatch")
    selected_rows = descriptors.rows_for_features(selected_features)
    if any(not descriptors.is_selected[row] for row in selected_rows):
        raise ValueError("descriptor payload does not mark every graph feature selected")
    observed_selected_ranks = descriptors.final_selected_rank[
        np.asarray(selected_rows, dtype=np.int64)
    ]
    if not np.array_equal(
        observed_selected_ranks,
        np.arange(len(selected_indices), dtype=np.int64),
    ):
        raise ValueError("descriptor final selected ranks disagree with compact order")

    cutoff_rank, cutoff_score = descriptors.seed_cutoff()
    control_rows = descriptors.near_cutoff_control_rows(
        selected_rows=frozenset(selected_rows),
        cutoff_rank=cutoff_rank,
        limit=max_near_cutoff,
    )
    expected_control_count = min(
        max_near_cutoff,
        max(0, descriptors.total_active_features - len(selected_indices)),
    )
    if len(control_rows) != expected_control_count:
        raise ValueError(
            "descriptor payload does not cover the required bounded near-cutoff band"
        )

    next_score = (
        float(descriptors.seed_influence[control_rows[0]])
        if control_rows
        else cutoff_score
    )
    gap = max(0.0, cutoff_score - next_score)
    relative_gap = float(
        gap / max(abs(cutoff_score), float(np.finfo(np.float64).tiny))
    )
    tie_count = int(np.count_nonzero(descriptors.seed_influence == cutoff_score))

    feature_rows = _integer_array(
        compact_result,
        "feature_row_node_indices",
        ndim=1,
    )
    _require_unique_indices(feature_rows, len(active), "feature_row_node_indices")
    feature_matrix = _float_array(compact_result, "feature_feature_edges", ndim=2)
    logit_matrix = _float_array(compact_result, "logit_feature_edges", ndim=2)
    if feature_matrix.shape != (len(feature_rows), len(selected_indices)):
        raise ValueError("feature_feature_edges has incompatible shape")
    if logit_matrix.shape != (len(graph.logit_token_ids), len(selected_indices)):
        raise ValueError("logit_feature_edges has incompatible shape")
    row_for_active_index = {
        int(active_index): row for row, active_index in enumerate(feature_rows.tolist())
    }
    selected_keys = tuple(_feature_key(row) for row in selected_features)

    selected_records = tuple(
        _selected_record(
            key=selected_keys[column],
            active_index=int(selected_indices[column]),
            descriptor_row=selected_rows[column],
            final_rank=column,
            cutoff_score=cutoff_score,
            descriptors=descriptors,
            feature_matrix=feature_matrix,
            logit_matrix=logit_matrix,
            logit_token_ids=graph.logit_token_ids,
            selected_keys=selected_keys,
            feature_row=row_for_active_index.get(int(selected_indices[column])),
            edge_limit=max_typed_neighborhood_edges,
        )
        for column in range(len(selected_indices))
    )
    near_records = tuple(
        _near_record(
            descriptor_row=row,
            rank=len(selected_records) + offset,
            cutoff_score=cutoff_score,
            descriptors=descriptors,
        )
        for offset, row in enumerate(control_rows)
    )
    evidence = build_frontier_evidence(
        graph_fingerprint=graph.graph_fingerprint,
        provider_fingerprint=graph.provider_fingerprint,
        decoder_fingerprint=decoder_fingerprint,
        cutoff_score=cutoff_score,
        relative_cutoff_gap=relative_gap,
        cutoff_tie_count=tie_count,
        near_cutoff_count=len(near_records),
        selected=selected_records,
        near_cutoff=near_records,
        bounds=FrontierBounds(
            max_selected=len(selected_records),
            max_near_cutoff=max_near_cutoff,
        ),
    )
    artifact = BoundedFrontierArtifact(
        evidence=evidence,
        descriptor_kind=descriptors.descriptor_kind,
        descriptor_version=descriptors.descriptor_version,
        cutoff_basis=CUTOFF_BASIS,
        max_typed_neighborhood_edges=max_typed_neighborhood_edges,
        content_fingerprint="sha256:" + "0" * 64,
        descriptor_scope=descriptors.descriptor_scope,
        decoder_source_fingerprint=descriptors.decoder_source_fingerprint,
        decoder_evidence_fingerprint=descriptors.decoder_evidence_fingerprint,
        projection_fingerprint=descriptors.projection_fingerprint,
    )
    return replace(artifact, content_fingerprint=_content_fingerprint(artifact))


def save_bounded_frontier_artifact(
    artifact: BoundedFrontierArtifact,
    path: Path,
) -> None:
    if artifact.content_fingerprint != _content_fingerprint(artifact):
        raise ValueError("frontier artifact content fingerprint is stale")
    payload = json.dumps(
        artifact.to_json(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(
        f".{artifact_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, artifact_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bounded_frontier_artifact(path: Path) -> BoundedFrontierArtifact:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load bounded frontier artifact") from exc
    return BoundedFrontierArtifact.from_json(_mapping(raw, "frontier artifact"))


@dataclass(frozen=True)
class _DescriptorTable:
    descriptor_kind: str
    descriptor_version: str
    total_active_features: int
    features: np.ndarray
    active_indices: np.ndarray
    activation: np.ndarray
    seed_influence: np.ndarray
    seed_rank: np.ndarray
    is_top_seed: np.ndarray
    is_selected: np.ndarray
    final_selected_rank: np.ndarray
    decoder_source_fingerprint: str
    decoder_evidence_fingerprint: str
    projection_fingerprint: str
    descriptor_scope: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        active_features: np.ndarray,
    ) -> Self:
        descriptor_kind = _string_scalar(payload.get("descriptor_kind"), "descriptor_kind")
        if descriptor_kind != DECODER_DESCRIPTOR_KIND:
            raise ValueError("frontier qualification requires qualification-grade decoder evidence")
        required = (
            "descriptor_kind",
            "descriptor_version",
            "total_active_features",
            "candidate_features",
            "candidate_row_indices",
            "activation_value",
            "seed_influence",
            "seed_rank",
            "is_top_seed",
            "is_selected_phase4",
            "phase4_selected_rank",
            "semantic_sketch",
            "phase4_selection_available",
            "seed_influence_available",
            "descriptor_scope",
            "descriptor_is_decoder_evidence",
            "decoder_source_fingerprint",
            "decoder_evidence_fingerprint",
            "projection_id",
            "projection_fingerprint",
            "semantic_descriptor_projection_max_bytes",
            "semantic_descriptor_projection_required_bytes",
            "semantic_descriptor_projection_workspace_peak_bytes",
            "semantic_descriptor_projection_admitted",
            "semantic_descriptor_projection_released",
            *_SEMANTIC_DESCRIPTOR_TRANSIENT_RECEIPT_FIELDS,
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"descriptor payload missing required fields: {missing}")
        _validate_descriptor_transient_receipt(payload)
        if descriptor_kind != DECODER_DESCRIPTOR_KIND or not _boolean_scalar(
            payload["descriptor_is_decoder_evidence"]
        ):
            raise ValueError("frontier qualification requires qualification-grade decoder evidence")
        if _string_scalar(payload["descriptor_scope"], "descriptor_scope") != DECODER_DESCRIPTOR_SCOPE:
            raise ValueError("unsupported decoder descriptor scope")
        decoder_source_fingerprint = _string_scalar(
            payload["decoder_source_fingerprint"], "decoder_source_fingerprint"
        )
        _require_fingerprint("decoder_source_fingerprint", decoder_source_fingerprint)
        decoder_evidence_fingerprint = _string_scalar(
            payload["decoder_evidence_fingerprint"], "decoder_evidence_fingerprint"
        )
        projection_fingerprint = _string_scalar(
            payload["projection_fingerprint"], "projection_fingerprint"
        )
        for field, value in (
            ("decoder_evidence_fingerprint", decoder_evidence_fingerprint),
            ("projection_fingerprint", projection_fingerprint),
        ):
            _require_fingerprint(field, value)
        projection_max = _integer_scalar(
            payload["semantic_descriptor_projection_max_bytes"],
            "semantic_descriptor_projection_max_bytes",
        )
        projection_required = _integer_scalar(
            payload["semantic_descriptor_projection_required_bytes"],
            "semantic_descriptor_projection_required_bytes",
        )
        projection_workspace = _integer_scalar(
            payload["semantic_descriptor_projection_workspace_peak_bytes"],
            "semantic_descriptor_projection_workspace_peak_bytes",
        )
        if projection_max != _SEMANTIC_DESCRIPTOR_TRANSIENT_MAX_BYTES or not (
            0 <= projection_required <= projection_max
            and 0 <= projection_workspace <= projection_max
        ):
            raise ValueError("decoder descriptor projection exceeds its byte bound")
        if not _boolean_scalar(payload["semantic_descriptor_projection_admitted"]) or not (
            _boolean_scalar(payload["semantic_descriptor_projection_released"])
        ):
            raise ValueError("decoder descriptor projection must be admitted and released")
        if not _boolean_scalar(payload["phase4_selection_available"]):
            raise ValueError("descriptor Phase4 selection annotation is required")
        if not _boolean_scalar(payload["seed_influence_available"]):
            raise ValueError("descriptor seed influence evidence is required")
        features = _array(payload["candidate_features"], dtype=np.int64)
        if features.ndim != 2 or features.shape[1:] != (3,):
            raise ValueError("candidate_features must have shape (N, 3)")
        count = len(features)
        active_indices = _aligned_array(
            payload["candidate_row_indices"], count, np.int64, "candidate_row_indices"
        )
        if np.any(active_indices < 0) or np.any(active_indices >= len(active_features)):
            raise ValueError("candidate_row_indices is outside active_features")
        if len(set(active_indices.tolist())) != count:
            raise ValueError("candidate_row_indices must be unique")
        if not np.array_equal(features, active_features[active_indices]):
            raise ValueError("descriptor features do not match active row indices")
        feature_keys = tuple(_feature_key(row) for row in features)
        if len(set(feature_keys)) != count:
            raise ValueError("descriptor candidate features must be unique")
        semantic_sketch = _array(payload["semantic_sketch"], dtype=np.float32)
        if semantic_sketch.ndim != 2 or semantic_sketch.shape[0] != count:
            raise ValueError("semantic_sketch must align with descriptor candidates")
        if not np.isfinite(semantic_sketch).all():
            raise ValueError("semantic_sketch must be finite")
        evidence_digest = hashlib.sha256()
        evidence_digest.update(decoder_source_fingerprint.encode("ascii"))
        evidence_digest.update(projection_fingerprint.encode("ascii"))
        evidence_digest.update(np.ascontiguousarray(features).tobytes())
        evidence_digest.update(np.ascontiguousarray(semantic_sketch).tobytes())
        if decoder_evidence_fingerprint != f"sha256:{evidence_digest.hexdigest()}":
            raise ValueError("decoder_evidence_fingerprint does not match descriptor arrays")
        activation = _aligned_array(
            payload["activation_value"], count, np.float64, "activation_value"
        )
        influence = _aligned_array(
            payload["seed_influence"], count, np.float64, "seed_influence"
        )
        if not np.isfinite(activation).all() or not np.isfinite(influence).all():
            raise ValueError("descriptor numeric evidence must be finite")
        seed_rank = _aligned_array(
            payload["seed_rank"], count, np.int64, "seed_rank"
        )
        if np.any(seed_rank < 0) or len(set(seed_rank.tolist())) != count:
            raise ValueError("descriptor seed ranks must be unique and non-negative")
        is_top_seed = _aligned_array(
            payload["is_top_seed"], count, np.bool_, "is_top_seed"
        )
        is_selected = _aligned_array(
            payload["is_selected_phase4"], count, np.bool_, "is_selected_phase4"
        )
        final_rank = _aligned_array(
            payload["phase4_selected_rank"],
            count,
            np.int64,
            "phase4_selected_rank",
        )
        if np.any(final_rank[is_selected] < 0) or np.any(final_rank[~is_selected] != -1):
            raise ValueError("descriptor final selection ranks are inconsistent")
        total = _integer_scalar(payload["total_active_features"], "total_active_features")
        if total != len(active_features):
            raise ValueError("descriptor total_active_features does not match compact")
        return cls(
            descriptor_kind=descriptor_kind,
            descriptor_version=_string_scalar(
                payload["descriptor_version"], "descriptor_version"
            ),
            total_active_features=total,
            features=features,
            active_indices=active_indices,
            activation=activation,
            seed_influence=influence,
            seed_rank=seed_rank,
            is_top_seed=is_top_seed,
            is_selected=is_selected,
            final_selected_rank=final_rank,
            decoder_source_fingerprint=decoder_source_fingerprint,
            decoder_evidence_fingerprint=decoder_evidence_fingerprint,
            projection_fingerprint=projection_fingerprint,
            descriptor_scope=DECODER_DESCRIPTOR_SCOPE,
        )

    def rows_for_features(self, features: np.ndarray) -> tuple[int, ...]:
        rows = {_feature_key(row): index for index, row in enumerate(self.features)}
        missing = [_feature_key(feature) for feature in features if _feature_key(feature) not in rows]
        if missing:
            raise ValueError(f"descriptor payload does not cover selected features: {missing}")
        return tuple(rows[_feature_key(feature)] for feature in features)

    def seed_cutoff(self) -> tuple[int, float]:
        rows = np.flatnonzero(self.is_top_seed)
        if rows.size == 0:
            raise ValueError("descriptor payload lacks captured seed cutoff evidence")
        cutoff_row = int(rows[np.argmax(self.seed_rank[rows])])
        return int(self.seed_rank[cutoff_row]), float(self.seed_influence[cutoff_row])

    def near_cutoff_control_rows(
        self,
        *,
        selected_rows: frozenset[int],
        cutoff_rank: int,
        limit: int,
    ) -> tuple[int, ...]:
        candidates = [
            row
            for row in range(len(self.features))
            if row not in selected_rows
        ]
        candidates.sort(
            key=lambda row: (
                abs(int(self.seed_rank[row]) - cutoff_rank),
                int(self.seed_rank[row]),
                -float(self.seed_influence[row]),
                _feature_key(self.features[row]),
            )
        )
        return tuple(candidates[:limit])


def _selected_record(
    *,
    key: FeatureKey,
    active_index: int,
    descriptor_row: int,
    final_rank: int,
    cutoff_score: float,
    descriptors: _DescriptorTable,
    feature_matrix: np.ndarray,
    logit_matrix: np.ndarray,
    logit_token_ids: np.ndarray,
    selected_keys: tuple[FeatureKey, ...],
    feature_row: int | None,
    edge_limit: int,
) -> FrontierRecord:
    del active_index
    feature_edges: tuple[FeatureFeatureEdge, ...] = ()
    if feature_row is not None:
        feature_edges = tuple(
            FeatureFeatureEdge(source=selected_keys[column], weight=weight)
            for column, weight in _bounded_weights(feature_matrix[feature_row], edge_limit)
        )
    logit_edges = tuple(
        LogitFeatureEdge(logit_token_id=int(logit_token_ids[row]), weight=weight)
        for row, weight in _bounded_weights(logit_matrix[:, final_rank], edge_limit)
    )
    seed_influence = float(descriptors.seed_influence[descriptor_row])
    return FrontierRecord(
        key=key,
        selected=True,
        rank=final_rank,
        selection_score=seed_influence,
        cutoff_distance=cutoff_score - seed_influence,
        activation=float(descriptors.activation[descriptor_row]),
        signed_target_effect=math.fsum(
            float(value) for value in logit_matrix[:, final_rank]
        ),
        influence=seed_influence,
        seed_rank=int(descriptors.seed_rank[descriptor_row]),
        final_selected_rank=final_rank,
        feature_from_feature=feature_edges,
        logit_from_feature=logit_edges,
    )


def _near_record(
    *,
    descriptor_row: int,
    rank: int,
    cutoff_score: float,
    descriptors: _DescriptorTable,
) -> FrontierRecord:
    seed_influence = float(descriptors.seed_influence[descriptor_row])
    return FrontierRecord(
        key=_feature_key(descriptors.features[descriptor_row]),
        selected=False,
        rank=rank,
        selection_score=seed_influence,
        cutoff_distance=cutoff_score - seed_influence,
        activation=float(descriptors.activation[descriptor_row]),
        signed_target_effect=0.0,
        influence=seed_influence,
        seed_rank=int(descriptors.seed_rank[descriptor_row]),
        final_selected_rank=None,
    )


def _bounded_weights(values: np.ndarray, limit: int) -> tuple[tuple[int, float], ...]:
    entries = [
        (index, float(value))
        for index, value in enumerate(values.tolist())
        if float(value) != 0.0
    ]
    entries.sort(key=lambda item: (-abs(item[1]), item[0]))
    return tuple(entries[:limit])


def _content_fingerprint(artifact: BoundedFrontierArtifact) -> str:
    payload = artifact.to_json()
    payload["content_fingerprint"] = None
    return _fingerprint_json(payload)


def _evidence_to_json(evidence: FrontierEvidence) -> dict[str, Any]:
    return {
        "graph_fingerprint": evidence.graph_fingerprint,
        "provider_fingerprint": evidence.provider_fingerprint,
        "decoder_fingerprint": evidence.decoder_fingerprint,
        "cutoff_score": evidence.cutoff_score,
        "relative_cutoff_gap": evidence.relative_cutoff_gap,
        "cutoff_tie_count": evidence.cutoff_tie_count,
        "near_cutoff_count": evidence.near_cutoff_count,
        "bounds": {
            "max_selected": evidence.bounds.max_selected,
            "max_near_cutoff": evidence.bounds.max_near_cutoff,
        },
        "selected": [_record_to_json(record) for record in evidence.selected],
        "near_cutoff": [_record_to_json(record) for record in evidence.near_cutoff],
    }


def _record_to_json(record: FrontierRecord) -> dict[str, Any]:
    return {
        "key": [record.key.layer, record.key.position, record.key.feature_id],
        "selected": record.selected,
        "rank": record.rank,
        "selection_score": record.selection_score,
        "cutoff_distance": record.cutoff_distance,
        "activation": record.activation,
        "signed_target_effect": record.signed_target_effect,
        "influence": record.influence,
        "seed_rank": record.seed_rank,
        "final_selected_rank": record.final_selected_rank,
        "feature_from_feature": [
            {
                "source": [edge.source.layer, edge.source.position, edge.source.feature_id],
                "weight": edge.weight,
            }
            for edge in record.feature_from_feature
        ],
        "logit_from_feature": [
            {"logit_token_id": edge.logit_token_id, "weight": edge.weight}
            for edge in record.logit_from_feature
        ],
    }


def _evidence_from_json(value: Mapping[str, Any]) -> FrontierEvidence:
    _require_fields(
        value,
        (
            "graph_fingerprint",
            "provider_fingerprint",
            "decoder_fingerprint",
            "cutoff_score",
            "relative_cutoff_gap",
            "cutoff_tie_count",
            "near_cutoff_count",
            "bounds",
            "selected",
            "near_cutoff",
        ),
        label="frontier evidence",
    )
    bounds = _mapping(value["bounds"], "bounds")
    _require_fields(bounds, ("max_selected", "max_near_cutoff"), label="bounds")
    return build_frontier_evidence(
        graph_fingerprint=_string(value["graph_fingerprint"], "graph_fingerprint"),
        provider_fingerprint=_string(
            value["provider_fingerprint"], "provider_fingerprint"
        ),
        decoder_fingerprint=_string(
            value["decoder_fingerprint"], "decoder_fingerprint"
        ),
        cutoff_score=_float(value["cutoff_score"], "cutoff_score"),
        relative_cutoff_gap=_float(
            value["relative_cutoff_gap"], "relative_cutoff_gap"
        ),
        cutoff_tie_count=_integer(value["cutoff_tie_count"], "cutoff_tie_count"),
        near_cutoff_count=_integer(
            value["near_cutoff_count"], "near_cutoff_count"
        ),
        selected=tuple(
            _record_from_json(item)
            for item in _sequence(value["selected"], "selected")
        ),
        near_cutoff=tuple(
            _record_from_json(item)
            for item in _sequence(value["near_cutoff"], "near_cutoff")
        ),
        bounds=FrontierBounds(
            max_selected=_integer(bounds["max_selected"], "max_selected"),
            max_near_cutoff=_integer(
                bounds["max_near_cutoff"], "max_near_cutoff"
            ),
        ),
    )


def _record_from_json(value: Any) -> FrontierRecord:
    record = _mapping(value, "frontier record")
    _require_fields(
        record,
        (
            "key",
            "selected",
            "rank",
            "selection_score",
            "cutoff_distance",
            "activation",
            "signed_target_effect",
            "influence",
            "seed_rank",
            "final_selected_rank",
            "feature_from_feature",
            "logit_from_feature",
        ),
        label="frontier record",
    )
    return FrontierRecord(
        key=_key_from_json(record["key"]),
        selected=_boolean(record["selected"], "selected"),
        rank=_integer(record["rank"], "rank"),
        selection_score=_float(record["selection_score"], "selection_score"),
        cutoff_distance=_float(record["cutoff_distance"], "cutoff_distance"),
        activation=_float(record["activation"], "activation"),
        signed_target_effect=_float(
            record["signed_target_effect"], "signed_target_effect"
        ),
        influence=_float(record["influence"], "influence"),
        seed_rank=_optional_integer(record["seed_rank"], "seed_rank"),
        final_selected_rank=_optional_integer(
            record["final_selected_rank"], "final_selected_rank"
        ),
        feature_from_feature=tuple(
            _feature_edge_from_json(item)
            for item in _sequence(
                record["feature_from_feature"], "feature_from_feature"
            )
        ),
        logit_from_feature=tuple(
            _logit_edge_from_json(item)
            for item in _sequence(record["logit_from_feature"], "logit_from_feature")
        ),
    )


def _feature_edge_from_json(value: Any) -> FeatureFeatureEdge:
    edge = _mapping(value, "feature edge")
    _require_fields(edge, ("source", "weight"), label="feature edge")
    return FeatureFeatureEdge(
        source=_key_from_json(edge["source"]),
        weight=_float(edge["weight"], "weight"),
    )


def _logit_edge_from_json(value: Any) -> LogitFeatureEdge:
    edge = _mapping(value, "logit edge")
    _require_fields(edge, ("logit_token_id", "weight"), label="logit edge")
    return LogitFeatureEdge(
        logit_token_id=_integer(edge["logit_token_id"], "logit_token_id"),
        weight=_float(edge["weight"], "weight"),
    )


def _key_from_json(value: Any) -> FeatureKey:
    items = _sequence(value, "feature key")
    if len(items) != 3:
        raise ValueError("feature key must contain three integers")
    return FeatureKey(*(_integer(item, "feature key") for item in items))


def _feature_key(row: np.ndarray) -> FeatureKey:
    return FeatureKey(int(row[0]), int(row[1]), int(row[2]))


def _integer_array(payload: Mapping[str, Any], name: str, *, ndim: int) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"compact result missing {name}")
    array = _array(payload[name], dtype=np.int64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _float_array(payload: Mapping[str, Any], name: str, *, ndim: int) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"compact result missing {name}")
    array = _array(payload[name], dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return array


def _array(value: Any, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _aligned_array(
    value: Any,
    count: int,
    dtype: np.dtype[Any] | type[Any],
    label: str,
) -> np.ndarray:
    array = _array(value, dtype=dtype).reshape(-1)
    if len(array) != count:
        raise ValueError(f"{label} must have one value per descriptor candidate")
    return array


def _require_unique_indices(values: np.ndarray, bound: int, label: str) -> None:
    if np.any(values < 0) or np.any(values >= bound):
        raise ValueError(f"{label} is outside its domain")
    if len(set(values.tolist())) != len(values):
        raise ValueError(f"{label} must contain unique indices")


def _boolean_scalar(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("boolean descriptor field must be scalar")
    return bool(array.item())


def _validate_descriptor_transient_receipt(payload: Mapping[str, Any]) -> None:
    policy_id = _string_scalar(
        payload["semantic_descriptor_transient_policy_id"],
        "semantic_descriptor_transient_policy_id",
    )
    max_bytes = _integer_scalar(
        payload["semantic_descriptor_transient_max_bytes"],
        "semantic_descriptor_transient_max_bytes",
    )
    required_bytes = _integer_scalar(
        payload["semantic_descriptor_transient_required_bytes"],
        "semantic_descriptor_transient_required_bytes",
    )
    admitted = _strict_boolean_scalar(
        payload["semantic_descriptor_transient_admitted"],
        "semantic_descriptor_transient_admitted",
    )
    released = _strict_boolean_scalar(
        payload["semantic_descriptor_transient_released"],
        "semantic_descriptor_transient_released",
    )
    array_count = _integer_scalar(
        payload["semantic_descriptor_transient_array_count"],
        "semantic_descriptor_transient_array_count",
    )
    if policy_id != _SEMANTIC_DESCRIPTOR_TRANSIENT_POLICY_ID:
        raise ValueError("unsupported semantic descriptor transient policy_id")
    if max_bytes != _SEMANTIC_DESCRIPTOR_TRANSIENT_MAX_BYTES:
        raise ValueError("semantic descriptor transient max_bytes must be exactly 64 MiB")
    if required_bytes < 0 or required_bytes > max_bytes:
        raise ValueError("semantic descriptor transient required_bytes exceeds its bound")
    if not admitted or not released:
        raise ValueError("semantic descriptor transient receipt must be admitted and released")
    if array_count != _SEMANTIC_DESCRIPTOR_TRANSIENT_ARRAY_COUNT:
        raise ValueError("semantic descriptor transient array_count must be exactly 3")


def _strict_boolean_scalar(value: Any, label: str) -> bool:
    array = np.asarray(value)
    if array.size != 1 or not isinstance(array.item(), (bool, np.bool_)):
        raise ValueError(f"{label} must be a boolean scalar")
    return bool(array.item())


def _integer_scalar(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be scalar")
    return _integer(array.item(), label)


def _string_scalar(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be scalar")
    return _string(array.item(), label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _require_fields(
    value: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise ValueError(
            f"{label} fields differ: missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def _fingerprint_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _require_fingerprint(label: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256 fingerprint") from exc
