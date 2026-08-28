"""Project adapter for sibling-owned live behavioral verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol, cast
import uuid

import numpy as np
import torch

from circuit_tracer.transcoder.decoder_row_source import (
    DecoderRowKey,
    DecoderRowRefusal,
)
from circuit_tracer.transcoder.provider import get_transcoder_capabilities
from circuit_tracer.verification import (
    AcceptedGraphView,
    AliasControlCandidate,
    AliasSelectionEvidence,
    AliasSubstitution,
    BehavioralFaithfulnessReport as SiblingBehavioralReport,
    BehavioralProbePolicy,
    BehavioralVerificationRequest,
    FeatureEvidence,
    FeatureNode,
    FeatureValue,
    TraceIdentity,
    select_direct_features,
    select_necessity_features,
)

from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    TypedCompactGraph,
    load_typed_compact_graph,
)

from .contracts import (
    BehavioralFaithfulnessReport,
    BehavioralFaithfulnessStatus,
)
from .frontier import FeatureKey, FrontierEvidence, FrontierRecord


MAX_UNSELECTED_CONTROLS = 8
MAX_NECESSITY_CONTROLS = 3
MAX_DOWNSTREAM_DELTAS_PER_SELECTED = 8
MAX_QUALIFIED_ALIASES = 2
ALIAS_COSINE_ABS_TOLERANCE = 1e-6


class BehavioralProbeMode(str, Enum):
    OFF = "off"
    SMOKE = "smoke"
    REQUIRED = "required"


class BehavioralRequestStatus(str, Enum):
    READY = "ready"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class DecoderVectorSource(Protocol):
    """Authoritative bounded decoder-vector seam; semantic sketches do not satisfy it."""

    decoder_fingerprint: str
    decoder_output_topology: str

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray: ...


class LiveProviderDecoderSource:
    """Bounded exact decoder access for the live CLT/PLT provider."""

    def __init__(self, model: object, *, decoder_fingerprint: str) -> None:
        provider = getattr(model, "transcoders", None)
        self._provider = getattr(provider, "_module", provider)
        if self._provider is None:
            raise ValueError("live model does not expose a transcoder provider")
        capabilities = get_transcoder_capabilities(self._provider)
        if capabilities.architecture not in {"clt", "plt"}:
            raise ValueError("alias decoder evidence requires a CLT or PLT provider")
        self._architecture = capabilities.architecture
        self.decoder_fingerprint = decoder_fingerprint
        self.decoder_output_topology = (
            "complete_downstream_block"
            if capabilities.decoder_output_topology == "cross_layer"
            else "same_layer"
        )
        self._cache: dict[FeatureKey, np.ndarray] = {}

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray:
        cached = self._cache.get(feature)
        if cached is not None:
            return cached
        if len(self._cache) >= MAX_UNSELECTED_CONTROLS + 2 * MAX_QUALIFIED_ALIASES:
            raise ValueError("bounded alias decoder-vector budget exceeded")
        if self._architecture == "clt":
            vector = self._clt_vector(feature)
        else:
            vector = self._plt_vector(feature)
        result = _exact_cpu_vector(vector)
        self._cache[feature] = result
        return result

    def _clt_vector(self, feature: FeatureKey) -> torch.Tensor:
        getter = getattr(self._provider, "get_decoder_block", None)
        if not callable(getter):
            raise ValueError("CLT provider lacks exact decoder-block access")
        block = getter(
            feature.layer,
            torch.tensor([feature.feature_id], dtype=torch.long),
        )
        if not isinstance(block, torch.Tensor) or block.ndim != 3 or block.shape[0] != 1:
            raise ValueError("CLT decoder block has an invalid shape")
        return block[0].reshape(-1)

    def _plt_vector(self, feature: FeatureKey) -> torch.Tensor:
        factory = getattr(self._provider, "create_decoder_row_source", None)
        if not callable(factory):
            raise ValueError("PLT provider lacks exact selective decoder-row access")
        row_source = factory(feature.layer, max_staging_bytes=8 * 1024 * 1024)
        try:
            materialized = row_source.materialize(
                (DecoderRowKey(feature.layer, feature.feature_id, 0),),
                destination="cpu",
                output_dtype=torch.float32,
            )
            if isinstance(materialized, DecoderRowRefusal):
                raise ValueError(
                    "PLT decoder row was refused: "
                    f"{materialized.code.value}: {materialized.reason}"
                )
            if materialized.rows.ndim != 2 or materialized.rows.shape[0] != 1:
                raise ValueError("PLT decoder row has an invalid shape")
            return materialized.rows[0]
        finally:
            row_source.release("behavioral_alias_decoder_capture_complete")


@dataclass(frozen=True)
class QualifiedAliasPair:
    """One alias already admitted by the frozen reference/candidate comparison."""

    source: FeatureKey
    substitute: FeatureKey
    selection_policy_id: str
    calibration_fingerprint: str
    comparison_evidence_fingerprint: str
    baseline_graph_fingerprint: str
    candidate_graph_fingerprint: str
    qualified_decoder_cosine: float

    def __post_init__(self) -> None:
        if self.source == self.substitute or self.source.layer != self.substitute.layer:
            raise ValueError("qualified alias nodes must be distinct and in the same layer")
        if not re.fullmatch(r".+_v[0-9]+", self.selection_policy_id):
            raise ValueError("alias selection_policy_id must be versioned")
        for name in (
            "calibration_fingerprint",
            "comparison_evidence_fingerprint",
            "baseline_graph_fingerprint",
            "candidate_graph_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if not np.isfinite(self.qualified_decoder_cosine) or not (
            -1.0 <= self.qualified_decoder_cosine <= 1.0
        ):
            raise ValueError("qualified_decoder_cosine must be finite and in [-1, 1]")


@dataclass(frozen=True)
class LiveBehavioralCandidate:
    graph_path: Path
    compact_result: Mapping[str, Any]
    frontier: FrontierEvidence
    trace_id: str
    trace_fingerprint: str
    target_fingerprint: str
    step_fingerprint: str
    semantic_fingerprint: str
    execution_fingerprint: str
    provider_fingerprint: str
    decoder_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    target_position: int
    target_token_id: int
    policy_mode: BehavioralProbeMode
    policy: BehavioralProbePolicy
    qualified_aliases: tuple[QualifiedAliasPair, ...] = ()
    decoder_source: DecoderVectorSource | None = None

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id is required")
        for name in (
            "trace_fingerprint",
            "target_fingerprint",
            "step_fingerprint",
            "provider_fingerprint",
            "decoder_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if not self.semantic_fingerprint or not self.execution_fingerprint:
            raise ValueError("semantic and execution fingerprints are required")
        if not self.prompt_token_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be non-empty non-negative integers")
        for name in ("target_position", "target_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.target_position == 0 or self.target_position > len(
            self.prompt_token_ids
        ):
            raise ValueError("target_position is outside prompt_token_ids")
        if len(self.qualified_aliases) > MAX_QUALIFIED_ALIASES:
            raise ValueError("qualified aliases exceed the bounded evidence limit")
        if bool(self.qualified_aliases) != (self.decoder_source is not None):
            raise ValueError(
                "qualified aliases and an authoritative decoder source are required together"
            )


@dataclass(frozen=True)
class BehavioralRequestPreparation:
    status: BehavioralRequestStatus
    request: BehavioralVerificationRequest | None
    assessment: BehavioralFaithfulnessReport

    def __post_init__(self) -> None:
        if (self.status is BehavioralRequestStatus.READY) != (self.request is not None):
            raise ValueError("only a ready preparation may contain a sibling request")


class SerializableSiblingBehavioralReport(Protocol):
    @property
    def evidence_fingerprint(self) -> str: ...

    def to_json(self) -> str: ...


def prepare_live_behavioral_request(
    candidate: LiveBehavioralCandidate,
) -> BehavioralRequestPreparation:
    """Strictly bind live trace evidence into the sibling request contract."""

    if candidate.policy_mode is BehavioralProbeMode.OFF:
        return _not_applicable("behavioral_probe_mode_off")
    try:
        graph = load_typed_compact_graph(candidate.graph_path)
    except (EOFError, OSError, ValueError) as exc:
        return _unknown("typed_graph_reopen_failed", detail=exc)
    try:
        _bind_identities(candidate, graph)
        view = _accepted_graph_view(candidate, graph)
        aliases = _alias_substitutions(candidate, view, graph)
        request = BehavioralVerificationRequest(
            identity=TraceIdentity(
                trace_id=candidate.trace_id,
                graph_fingerprint=graph.graph_fingerprint,
                provider_fingerprint=candidate.provider_fingerprint,
                semantic_fingerprint=candidate.semantic_fingerprint,
                execution_fingerprint=candidate.execution_fingerprint,
                prompt_token_ids=candidate.prompt_token_ids,
                target_position=candidate.target_position,
                target_token_id=candidate.target_token_id,
            ),
            graph=view,
            policy=candidate.policy,
            aliases=aliases,
        )
    except _EvidenceRefusal as exc:
        return _unknown(exc.code, detail=exc)
    except ValueError as exc:
        return _unknown("behavioral_request_contract_refused", detail=exc)
    return BehavioralRequestPreparation(
        status=BehavioralRequestStatus.READY,
        request=request,
        assessment=BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.UNKNOWN,
            metrics={"policy_id": candidate.policy.policy_id},
            reason_codes=("behavioral_probe_not_yet_executed",),
        ),
    )


def map_sibling_behavioral_report(
    report: SiblingBehavioralReport,
) -> BehavioralFaithfulnessReport:
    """Project summary of a sibling-owned evidence report."""

    status = BehavioralFaithfulnessStatus(report.verdict.value)
    metrics = {
        "policy_id": report.policy_id,
        "calibration_id": report.calibration_id,
        "evidence_completeness": report.evidence_completeness.value,
        "runtime_status": report.runtime_status.value,
        "variants_planned": report.variants_planned,
        "variants_completed": report.variants_completed,
        "no_op_required": report.no_op_required,
        "no_op_passed": report.no_op_passed,
        "direct_mean_abs_closure": report.metrics.direct_mean_abs_closure,
        "direct_sign_agreement": report.metrics.direct_sign_agreement,
        "necessity_high_vs_control_separation": (
            report.metrics.necessity_high_vs_control_separation
        ),
        "alias_mean_abs_target_delta": report.metrics.alias_mean_abs_target_delta,
        "alias_mean_abs_closure": report.metrics.alias_mean_abs_closure,
        "alias_substitution_vs_source_ablation": (
            report.metrics.alias_substitution_vs_source_ablation
        ),
        "alias_control_vs_source_ablation": (
            report.metrics.alias_control_vs_source_ablation
        ),
        "alias_substitution_advantage": report.metrics.alias_substitution_advantage,
        "downstream_mean_abs_closure": report.metrics.downstream_mean_abs_closure,
    }
    # New calibrated aggregates are additive to the sibling report contract. Keep
    # this adapter able to summarize reports produced by either side of an atomic
    # paired-workspace rollout; absent fields remain explicit rather than being
    # inferred from the older absolute-error aggregates.
    for name in (
        "direct_mean_relative_closure",
        "direct_max_relative_closure",
        "downstream_mean_relative_closure",
        "downstream_p95_relative_closure",
        "necessity_predicted_realized_spearman",
        "necessity_median_high_control_effect_ratio",
        "alias_relative_effect_error",
    ):
        metrics[name] = getattr(report.metrics, name, None)
    reasons = [_reason_code(reason) for reason in report.reasons]
    if report.refusal is not None:
        reasons.append(f"runtime_refusal_{_slug(report.refusal.code)}")
    reason_codes = tuple(dict.fromkeys(reasons))
    serialized = cast(SerializableSiblingBehavioralReport, report)
    return BehavioralFaithfulnessReport(
        status=status,
        metrics=metrics,
        reason_codes=reason_codes,
        evidence_fingerprint=_project_fingerprint(serialized.evidence_fingerprint),
    )


def persist_sibling_behavioral_report(
    report: SerializableSiblingBehavioralReport,
    path: Path,
) -> None:
    """Persist only the sibling-owned canonical JSON representation."""

    payload = report.to_json()
    if not isinstance(payload, str):
        raise TypeError("sibling behavioral report to_json() must return a string")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("sibling behavioral report JSON must be an object")
    if parsed.get("evidence_fingerprint") != report.evidence_fingerprint:
        raise ValueError("sibling behavioral report fingerprint field mismatch")
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload.rstrip("\n") + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)


def _bind_identities(
    candidate: LiveBehavioralCandidate,
    graph: TypedCompactGraph,
) -> None:
    expected = {
        "trace_fingerprint": candidate.trace_fingerprint,
        "target_fingerprint": candidate.target_fingerprint,
        "step_fingerprint": candidate.step_fingerprint,
        "provider_fingerprint": candidate.provider_fingerprint,
    }
    for name, value in expected.items():
        if getattr(graph, name) != value:
            raise _EvidenceRefusal("live_candidate_identity_mismatch", name)
    if tuple(int(value) for value in graph.token_ids.tolist()) != (
        candidate.prompt_token_ids
    ):
        raise _EvidenceRefusal("prompt_token_identity_mismatch")
    if candidate.target_token_id not in graph.logit_token_ids.tolist():
        raise _EvidenceRefusal("target_token_not_in_graph")
    frontier = candidate.frontier
    frontier_expected = {
        "graph_fingerprint": graph.graph_fingerprint,
        "provider_fingerprint": candidate.provider_fingerprint,
        "decoder_fingerprint": candidate.decoder_fingerprint,
    }
    for name, expected_value in frontier_expected.items():
        if getattr(frontier, name) != expected_value:
            raise _EvidenceRefusal("frontier_identity_mismatch", name)


def _accepted_graph_view(
    candidate: LiveBehavioralCandidate,
    graph: TypedCompactGraph,
) -> AcceptedGraphView:
    compact = candidate.compact_result
    active = _integer_array(compact, "active_features", ndim=2)
    if active.shape[1:] != (3,):
        raise _EvidenceRefusal("invalid_compact_field", "active_features")
    active_keys = tuple(_feature_key(row) for row in active)
    if len(set(active_keys)) != len(active_keys):
        raise _EvidenceRefusal("invalid_compact_field", "active_features")
    activation_values = _floating_vector(compact, "activation_values")
    if len(activation_values) != len(active) or not np.isfinite(
        activation_values
    ).all():
        raise _EvidenceRefusal("invalid_compact_field", "activation_values")
    activation_by_key = {
        key: float(activation_values[index]) for index, key in enumerate(active_keys)
    }
    selected_indices = _integer_array(compact, "selected_features", ndim=1)
    feature_rows = _integer_array(compact, "feature_row_node_indices", ndim=1)
    if (
        selected_indices.size == 0
        or np.any(selected_indices < 0)
        or np.any(selected_indices >= len(active))
    ):
        raise _EvidenceRefusal("invalid_compact_field", "selected_features")
    if len(set(selected_indices.tolist())) != len(selected_indices):
        raise _EvidenceRefusal("invalid_compact_field", "selected_features")
    if (
        feature_rows.size == 0
        or np.any(feature_rows < 0)
        or np.any(feature_rows >= len(active))
    ):
        raise _EvidenceRefusal("invalid_compact_field", "feature_row_node_indices")
    if len(set(feature_rows.tolist())) != len(feature_rows):
        raise _EvidenceRefusal("invalid_compact_field", "feature_row_node_indices")
    selected_nodes = active[selected_indices]
    if not np.array_equal(selected_nodes, graph.feature_ids):
        raise _EvidenceRefusal("compact_selected_graph_mismatch")
    selected_keys = tuple(_feature_key(row) for row in selected_nodes)
    frontier_selected = {record.key: record for record in candidate.frontier.selected}
    if set(frontier_selected) != set(selected_keys):
        raise _EvidenceRefusal("frontier_selected_graph_mismatch")
    _validate_frontier_active_values(
        candidate.frontier.selected + candidate.frontier.near_cutoff,
        activation_by_key=activation_by_key,
    )

    feature_matrix = _floating_matrix(compact, "feature_feature_edges")
    logit_matrix = _floating_matrix(compact, "logit_feature_edges")
    if feature_matrix.shape != (len(feature_rows), len(selected_indices)):
        raise _EvidenceRefusal("invalid_compact_field", "feature_feature_edges")
    if logit_matrix.shape != (len(graph.logit_token_ids), len(selected_indices)):
        raise _EvidenceRefusal("invalid_compact_field", "logit_feature_edges")
    target_rows = np.flatnonzero(graph.logit_token_ids == candidate.target_token_id)
    if target_rows.size != 1:
        raise _EvidenceRefusal("ambiguous_target_logit_row")
    target_row = int(target_rows[0])
    target_influences = logit_matrix[target_row]
    if not np.isfinite(target_influences).all():
        raise _EvidenceRefusal("invalid_compact_field", "logit_feature_edges[target]")
    row_nodes = tuple(_feature_node(active[index]) for index in feature_rows)

    selected_features_without_closure = tuple(
        FeatureEvidence(
            node=_to_sibling_node(key),
            baseline_preactivation=activation_by_key[key],
            target_influence=float(target_influences[column]),
            selected=True,
            selection_rank=frontier_selected[key].rank,
        )
        for column, key in enumerate(selected_keys)
    )
    selected_only_view = AcceptedGraphView(
        graph_fingerprint=graph.graph_fingerprint,
        features=selected_features_without_closure,
    )
    selection_kwargs: dict[str, float] = {}
    if candidate.policy.calibration is not None:
        selection_kwargs["min_abs_predicted_target_delta"] = (
            candidate.policy.calibration.direct_min_abs_predicted_target_delta
        )
    direct_features = select_direct_features(
        selected_only_view,
        sample_count=candidate.policy.direct_sample_count,
        **selection_kwargs,
    )
    necessity_sample_count = min(
        candidate.policy.necessity_sample_count,
        MAX_NECESSITY_CONTROLS,
    )
    required_necessity_nodes: list[FeatureNode] = []
    for alias in candidate.qualified_aliases[: candidate.policy.alias_sample_count]:
        source_node = _to_sibling_node(alias.source)
        if (
            source_node not in required_necessity_nodes
            and len(required_necessity_nodes) < necessity_sample_count
        ):
            required_necessity_nodes.append(source_node)
    necessity_features = select_necessity_features(
        selected_only_view,
        sample_count=necessity_sample_count,
        required_nodes=tuple(required_necessity_nodes),
        **selection_kwargs,
    )
    alias_records = _alias_frontier_control_records(candidate)
    reserved_alias_keys = {record.key for record in alias_records}
    matched_controls = _matched_necessity_controls(
        necessity_features=necessity_features,
        active_keys=active_keys,
        activation_by_key=activation_by_key,
        selected_keys=set(selected_keys),
        reserved_keys=reserved_alias_keys,
    )
    alias_controls = tuple(
        FeatureEvidence(
            node=_to_sibling_node(record.key),
            baseline_preactivation=activation_by_key[record.key],
            target_influence=record.signed_target_effect,
            selected=False,
        )
        for record in alias_records
    )
    controls = matched_controls + alias_controls
    if len(controls) > MAX_UNSELECTED_CONTROLS:
        raise _EvidenceRefusal("bounded_unselected_control_budget_exceeded")
    if not controls:
        raise _EvidenceRefusal("missing_same_layer_unselected_control")
    direct_nodes = {feature.node for feature in direct_features}
    selected_features = tuple(
        replace(
            feature,
            predicted_downstream_feature_deltas=(
                _downstream_deltas(
                    source=selected_keys[column],
                    column=column,
                    row_nodes=row_nodes,
                    matrix=feature_matrix,
                )
                if feature.node in direct_nodes
                else ()
            ),
        )
        for column, feature in enumerate(selected_features_without_closure)
    )
    return AcceptedGraphView(
        graph_fingerprint=graph.graph_fingerprint,
        features=selected_features + controls,
    )


def _alias_substitutions(
    candidate: LiveBehavioralCandidate,
    view: AcceptedGraphView,
    graph: TypedCompactGraph,
) -> tuple[AliasSubstitution, ...]:
    if not candidate.qualified_aliases:
        return ()
    source = candidate.decoder_source
    assert source is not None
    if source.decoder_fingerprint != candidate.decoder_fingerprint:
        raise _EvidenceRefusal("alias_decoder_identity_mismatch")
    if source.decoder_output_topology not in {
        "complete_downstream_block",
        "same_layer",
    }:
        raise _EvidenceRefusal("unsupported_alias_decoder_topology")
    features = {item.node: item for item in view.features}
    aliases: list[AliasSubstitution] = []
    for qualified in candidate.qualified_aliases:
        if qualified.candidate_graph_fingerprint != graph.graph_fingerprint:
            raise _EvidenceRefusal("alias_candidate_graph_identity_mismatch")
        source_node = _to_sibling_node(qualified.source)
        substitute_node = _to_sibling_node(qualified.substitute)
        source_feature = features.get(source_node)
        substitute_feature = features.get(substitute_node)
        if source_feature is None or substitute_feature is None:
            raise _EvidenceRefusal("qualified_alias_not_in_accepted_graph")
        if not source_feature.selected:
            raise _EvidenceRefusal("qualified_alias_source_not_selected")

        source_vector = _decoder_vector(source, qualified.source)
        substitute_vector = _decoder_vector(source, qualified.substitute)
        if source_vector.shape != substitute_vector.shape:
            raise _EvidenceRefusal("alias_decoder_shape_mismatch")
        source_norm = float(np.linalg.norm(source_vector))
        substitute_norm = float(np.linalg.norm(substitute_vector))
        if source_norm == 0.0 or substitute_norm == 0.0:
            raise _EvidenceRefusal("zero_norm_alias_decoder")
        observed_cosine = float(
            np.clip(
                np.dot(source_vector, substitute_vector)
                / (source_norm * substitute_norm),
                -1.0,
                1.0,
            )
        )
        if not np.isclose(
            observed_cosine,
            qualified.qualified_decoder_cosine,
            rtol=0.0,
            atol=ALIAS_COSINE_ABS_TOLERANCE,
        ):
            raise _EvidenceRefusal("qualified_alias_decoder_cosine_mismatch")
        coefficient = float(
            np.dot(source_vector, substitute_vector)
            / np.dot(substitute_vector, substitute_vector)
        )
        substitute_delta = source_feature.baseline_preactivation * coefficient
        substitute_absolute = (
            substitute_feature.baseline_preactivation + substitute_delta
        )
        predicted_target_delta = _alias_predicted_target_delta(
            source=source_feature,
            substitute=substitute_feature,
            substitute_delta=substitute_delta,
        )
        controls = _alias_controls(
            source=qualified.source,
            substitute=qualified.substitute,
            source_vector=source_vector,
            features=features,
            decoder_source=source,
        )
        if not controls:
            raise _EvidenceRefusal("missing_low_cosine_alias_control")
        aliases.append(
            AliasSubstitution(
                source=source_node,
                substitute=substitute_node,
                substitute_absolute_preactivation=substitute_absolute,
                selection_evidence=AliasSelectionEvidence(
                    selection_policy_id=qualified.selection_policy_id,
                    calibration_fingerprint=qualified.calibration_fingerprint,
                    comparison_evidence_fingerprint=(
                        qualified.comparison_evidence_fingerprint
                    ),
                    baseline_graph_fingerprint=qualified.baseline_graph_fingerprint,
                    candidate_graph_fingerprint=qualified.candidate_graph_fingerprint,
                    decoder_fingerprint=candidate.decoder_fingerprint,
                    decoder_output_topology=source.decoder_output_topology,
                    qualified_decoder_cosine=qualified.qualified_decoder_cosine,
                    observed_decoder_cosine=observed_cosine,
                    source_decoder_norm=source_norm,
                    substitute_decoder_norm=substitute_norm,
                    least_squares_coefficient=coefficient,
                ),
                predicted_target_delta=predicted_target_delta,
                control_candidates=controls,
            )
        )
    return tuple(aliases)


def _alias_controls(
    *,
    source: FeatureKey,
    substitute: FeatureKey,
    source_vector: np.ndarray,
    features: Mapping[FeatureNode, FeatureEvidence],
    decoder_source: DecoderVectorSource,
) -> tuple[AliasControlCandidate, ...]:
    source_norm = float(np.linalg.norm(source_vector))
    candidates: list[AliasControlCandidate] = []
    for node, feature in features.items():
        key = FeatureKey(node.layer, node.position, node.feature)
        if (
            feature.selected
            or feature.necessity_control_for is not None
            or key.layer != source.layer
            or key in {source, substitute}
        ):
            continue
        vector = _decoder_vector(decoder_source, key)
        norm = float(np.linalg.norm(vector))
        cosine = (
            0.0
            if norm == 0.0
            else float(np.clip(np.dot(source_vector, vector) / (source_norm * norm), -1.0, 1.0))
        )
        candidates.append(AliasControlCandidate(node=node, similarity_to_source=cosine))
    return tuple(sorted(candidates, key=lambda item: (item.similarity_to_source, item.node))[:1])


def _decoder_vector(source: DecoderVectorSource, feature: FeatureKey) -> np.ndarray:
    vector = np.asarray(source.decoder_vector(feature), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise _EvidenceRefusal("invalid_authoritative_decoder_vector")
    return vector


def _exact_cpu_vector(value: torch.Tensor) -> np.ndarray:
    vector = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("authoritative decoder vector must be non-empty and finite")
    vector.setflags(write=False)
    return vector


def _alias_predicted_target_delta(
    *,
    source: FeatureEvidence,
    substitute: FeatureEvidence,
    substitute_delta: float,
) -> float | None:
    if substitute.baseline_preactivation == 0.0:
        return None
    return -source.target_influence + (
        substitute_delta
        / substitute.baseline_preactivation
        * substitute.target_influence
    )


def _downstream_deltas(
    *,
    source: FeatureKey,
    column: int,
    row_nodes: tuple[FeatureNode, ...],
    matrix: np.ndarray,
) -> tuple[FeatureValue, ...]:
    candidates = [
        FeatureValue(node=node, preactivation=float(matrix[row, column]))
        for row, node in enumerate(row_nodes)
        if node.layer > source.layer and float(matrix[row, column]) != 0.0
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-abs(item.preactivation), item.node),
        )[:MAX_DOWNSTREAM_DELTAS_PER_SELECTED]
    )


def _matched_necessity_controls(
    *,
    necessity_features: tuple[FeatureEvidence, ...],
    active_keys: tuple[FeatureKey, ...],
    activation_by_key: Mapping[FeatureKey, float],
    selected_keys: set[FeatureKey],
    reserved_keys: set[FeatureKey],
) -> tuple[FeatureEvidence, ...]:
    """Choose one explicit, unique null control for each necessity anchor."""

    available = set(active_keys) - selected_keys - reserved_keys
    controls: list[FeatureEvidence] = []
    for high in necessity_features:
        high_key = FeatureKey(high.node.layer, high.node.position, high.node.feature)
        candidates = (key for key in available if key.layer == high_key.layer)
        control = min(
            candidates,
            key=lambda key: (
                key.position != high_key.position,
                abs(
                    abs(activation_by_key[key])
                    - abs(high.baseline_preactivation)
                ),
                key,
            ),
            default=None,
        )
        if control is None:
            raise _EvidenceRefusal("missing_matched_necessity_control")
        available.remove(control)
        controls.append(
            FeatureEvidence(
                node=_to_sibling_node(control),
                baseline_preactivation=activation_by_key[control],
                # This is an intentionally declared null prediction. Unselected
                # active features have no accepted target-logit graph column.
                target_influence=0.0,
                selected=False,
                necessity_control_for=high.node,
            )
        )
    return tuple(controls)


def _alias_frontier_control_records(
    candidate: LiveBehavioralCandidate,
) -> tuple[FrontierRecord, ...]:
    """Retain frontier controls only when a pre-qualified alias needs them."""

    if not candidate.qualified_aliases:
        return ()
    records = {record.key: record for record in candidate.frontier.near_cutoff}
    substitutes = {alias.substitute for alias in candidate.qualified_aliases}
    chosen: list[FrontierRecord] = []
    for alias in candidate.qualified_aliases:
        substitute = records.get(alias.substitute)
        if substitute is None:
            raise _EvidenceRefusal("qualified_alias_substitute_not_in_frontier")
        if substitute not in chosen:
            chosen.append(substitute)
        comparator = min(
            (
                record
                for record in records.values()
                if record.key.layer == alias.source.layer
                and record.key not in substitutes
                and record not in chosen
            ),
            key=_control_order,
            default=None,
        )
        if comparator is None:
            raise _EvidenceRefusal("missing_alias_frontier_control")
        chosen.append(comparator)
    return tuple(chosen)


def _validate_frontier_active_values(
    records: tuple[FrontierRecord, ...],
    *,
    activation_by_key: Mapping[FeatureKey, float],
) -> None:
    for record in records:
        active_value = activation_by_key.get(record.key)
        if active_value is None:
            raise _EvidenceRefusal("frontier_feature_not_in_active_features")
        if active_value != record.activation:
            raise _EvidenceRefusal("frontier_activation_value_mismatch")


def _integer_array(value: Mapping[str, Any], key: str, *, ndim: int) -> np.ndarray:
    array = _array(value, key)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise _EvidenceRefusal("invalid_compact_field", key)
    return array.astype(np.int64, copy=False)


def _floating_matrix(value: Mapping[str, Any], key: str) -> np.ndarray:
    array = _array(value, key)
    if (
        array.ndim != 2
        or not np.issubdtype(array.dtype, np.floating)
    ):
        raise _EvidenceRefusal("invalid_compact_field", key)
    return array


def _floating_vector(value: Mapping[str, Any], key: str) -> np.ndarray:
    array = _array(value, key)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.floating):
        raise _EvidenceRefusal("invalid_compact_field", key)
    return array


def _array(value: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in value:
        raise _EvidenceRefusal("missing_compact_field", key)
    item = value[key]
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    return np.asarray(item)


def _feature_key(row: np.ndarray) -> FeatureKey:
    return FeatureKey(*(int(value) for value in row.tolist()))


def _feature_node(row: np.ndarray) -> FeatureNode:
    return FeatureNode(*(int(value) for value in row.tolist()))


def _to_sibling_node(key: FeatureKey) -> FeatureNode:
    return FeatureNode(key.layer, key.position, key.feature_id)


def _control_order(record: FrontierRecord) -> tuple[int, FeatureKey]:
    return record.rank, record.key


def _unknown(
    code: str, *, detail: Exception | None = None
) -> BehavioralRequestPreparation:
    metrics: dict[str, bool | int | float | str | None] = {}
    if detail is not None:
        metrics["refusal_type"] = type(detail).__name__
    return BehavioralRequestPreparation(
        status=BehavioralRequestStatus.UNKNOWN,
        request=None,
        assessment=BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.UNKNOWN,
            metrics=metrics,
            reason_codes=(code,),
        ),
    )


def _not_applicable(code: str) -> BehavioralRequestPreparation:
    return BehavioralRequestPreparation(
        status=BehavioralRequestStatus.NOT_APPLICABLE,
        request=None,
        assessment=BehavioralFaithfulnessReport(
            status=BehavioralFaithfulnessStatus.NOT_APPLICABLE,
            reason_codes=(code,),
        ),
    )


def _reason_code(value: str) -> str:
    return f"sibling_{_slug(value)}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unspecified"


def _require_fingerprint(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a sha256 fingerprint")


def _project_fingerprint(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value):
        return f"sha256:{value}"
    _require_fingerprint(value, name="evidence_fingerprint")
    return value


class _EvidenceRefusal(ValueError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(code if detail is None else f"{code}: {detail}")
