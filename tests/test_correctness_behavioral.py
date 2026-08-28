from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from circuit_tracer.verification import (
    BehavioralAggregateMetrics,
    BehavioralFaithfulnessReport as SiblingBehavioralReport,
    BehavioralProbePolicy,
    EvidenceCompleteness,
    FaithfulnessVerdict,
    FeatureNode,
    FrozenBehavioralCalibration,
    RuntimeExecutionStatus,
    VariantKind,
    plan_behavioral_variants,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities

from nlp_research_project.exact_trace_bench.correctness.behavioral import (
    BehavioralProbeMode,
    BehavioralRequestStatus,
    LiveProviderDecoderSource,
    LiveBehavioralCandidate,
    QualifiedAliasPair,
    _accepted_graph_view,
    map_sibling_behavioral_report,
    persist_sibling_behavioral_report,
    prepare_live_behavioral_request,
)
from nlp_research_project.exact_trace_bench.correctness.contracts import (
    BehavioralFaithfulnessStatus,
)
from nlp_research_project.exact_trace_bench.correctness.frontier import (
    FeatureKey,
    FrontierBounds,
    FrontierEvidence,
    FrontierRecord,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    TypedCompactGraph,
    load_typed_compact_graph,
)
from typed_graph_fixtures import write_typed_graph


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _record(
    key: FeatureKey,
    *,
    selected: bool,
    rank: int,
    activation: float,
    effect: float,
) -> FrontierRecord:
    return FrontierRecord(
        key=key,
        selected=selected,
        rank=rank,
        selection_score=10.0 - rank,
        cutoff_distance=-0.1 if selected else 0.1,
        activation=activation,
        signed_target_effect=effect,
        influence=abs(effect),
    )


def _candidate(tmp_path: Path, *, controls: int = 1) -> LiveBehavioralCandidate:
    graph_path = tmp_path / "step_000.npz"
    write_typed_graph(graph_path)
    graph = load_typed_compact_graph(graph_path)
    selected_keys = (FeatureKey(0, 0, 4), FeatureKey(1, 1, 9))
    matched_keys = (FeatureKey(0, 0, 200), FeatureKey(1, 1, 201))
    control_keys = tuple(FeatureKey(0, 1, 100 + index) for index in range(controls))
    active = np.asarray(
        [
            *(
                tuple((key.layer, key.position, key.feature_id))
                for key in selected_keys
            ),
            *(tuple((key.layer, key.position, key.feature_id)) for key in matched_keys),
            *(tuple((key.layer, key.position, key.feature_id)) for key in control_keys),
        ],
        dtype=np.int64,
    )
    frontier = FrontierEvidence(
        graph_fingerprint=graph.graph_fingerprint,
        provider_fingerprint=graph.provider_fingerprint,
        decoder_fingerprint=_fingerprint("decoder"),
        cutoff_score=0.5,
        relative_cutoff_gap=0.01,
        cutoff_tie_count=0,
        near_cutoff_count=controls,
        selected=(
            _record(
                selected_keys[0],
                selected=True,
                rank=0,
                activation=2.0,
                effect=0.8,
            ),
            _record(
                selected_keys[1],
                selected=True,
                rank=1,
                activation=-3.0,
                effect=-0.4,
            ),
        ),
        near_cutoff=tuple(
            _record(
                key,
                selected=False,
                rank=2 + index,
                activation=float(np.float32(2.1 + index)),
                effect=0.01 + index / 100,
            )
            for index, key in enumerate(control_keys)
        ),
        bounds=FrontierBounds(max_selected=2, max_near_cutoff=max(controls, 1)),
    )
    compact_result = {
        "active_features": active,
        "activation_values": np.asarray(
            [
                2.0,
                -3.0,
                1.9,
                -2.9,
                *(2.1 + index for index in range(controls)),
            ],
            dtype=np.float32,
        ),
        "selected_features": np.asarray([0, 1], dtype=np.int64),
        "feature_row_node_indices": np.asarray([0, 1], dtype=np.int64),
        "feature_feature_edges": np.asarray([[1.0, 0.2], [0.5, 1.0]], dtype=np.float32),
        "logit_feature_edges": np.asarray([[0.8, -0.4]], dtype=np.float32),
        "semantic_fingerprint": "semantic-v1",
        "execution_fingerprint": "execution-v1",
    }
    return LiveBehavioralCandidate(
        graph_path=graph_path,
        compact_result=compact_result,
        frontier=frontier,
        trace_id="trace-0",
        trace_fingerprint=graph.trace_fingerprint,
        target_fingerprint=graph.target_fingerprint,
        step_fingerprint=graph.step_fingerprint,
        semantic_fingerprint="semantic-v1",
        execution_fingerprint="execution-v1",
        provider_fingerprint=graph.provider_fingerprint,
        decoder_fingerprint=_fingerprint("decoder"),
        prompt_token_ids=tuple(int(value) for value in graph.token_ids.tolist()),
        target_position=1,
        target_token_id=42,
        policy_mode=BehavioralProbeMode.SMOKE,
        policy=BehavioralProbePolicy(),
    )


def test_builds_bounded_accepted_view_from_strictly_bound_evidence(
    tmp_path: Path,
) -> None:
    initial = _candidate(tmp_path, controls=12)
    compact_without_runtime_identity = {
        key: value
        for key, value in initial.compact_result.items()
        if key not in {"semantic_fingerprint", "execution_fingerprint"}
    }
    candidate = replace(
        initial,
        compact_result=compact_without_runtime_identity,
        target_position=len(initial.prompt_token_ids),
    )

    prepared = prepare_live_behavioral_request(candidate)

    assert prepared.status is BehavioralRequestStatus.READY
    assert prepared.request is not None
    features = prepared.request.graph.features
    selected = tuple(item for item in features if item.selected)
    controls = tuple(item for item in features if not item.selected)
    assert len(selected) == 2
    assert len(controls) == 2
    assert selected[0].baseline_preactivation == 2.0
    assert selected[0].target_influence == np.float32(0.8)
    assert selected[0].predicted_downstream_feature_deltas[0].node.layer == 1
    assert selected[0].predicted_downstream_feature_deltas[0].preactivation == 0.5
    assert selected[1].predicted_downstream_feature_deltas == ()
    assert tuple(item.node.feature for item in controls) == (200, 201)
    assert all(item.target_influence == 0.0 for item in controls)
    assert tuple(item.necessity_control_for for item in controls) == tuple(
        item.node for item in selected
    )


def test_matched_controls_prefer_same_position_then_activation_and_are_unique(
    tmp_path: Path,
) -> None:
    initial = _candidate(tmp_path, controls=1)
    compact = dict(initial.compact_result)
    compact["active_features"] = np.concatenate(
        (
            np.asarray(compact["active_features"]),
            np.asarray([[0, 9, 202], [1, 9, 203]], dtype=np.int64),
        ),
        axis=0,
    )
    compact["activation_values"] = np.concatenate(
        (
            np.asarray(compact["activation_values"]),
            np.asarray([2.0, -3.0], dtype=np.float32),
        )
    )

    prepared = prepare_live_behavioral_request(
        replace(initial, compact_result=compact)
    )

    assert prepared.status is BehavioralRequestStatus.READY
    assert prepared.request is not None
    controls = tuple(
        feature
        for feature in prepared.request.graph.features
        if feature.necessity_control_for is not None
    )
    # Exact activation matches exist at another position, but same-position
    # controls are the first matching dimension and each anchor owns one control.
    assert tuple(feature.node.feature for feature in controls) == (200, 201)
    assert len({feature.node for feature in controls}) == len(controls)


def test_active_activation_mismatch_refuses_before_planning(tmp_path: Path) -> None:
    initial = _candidate(tmp_path)
    compact = dict(initial.compact_result)
    activation_values = np.asarray(compact["activation_values"]).copy()
    activation_values[0] = 2.5
    compact["activation_values"] = activation_values

    prepared = prepare_live_behavioral_request(
        replace(initial, compact_result=compact)
    )

    assert prepared.status is BehavioralRequestStatus.UNKNOWN
    assert prepared.assessment.reason_codes == ("frontier_activation_value_mismatch",)


def test_calibrated_eligibility_is_shared_by_anchor_and_control_selection(
    tmp_path: Path,
) -> None:
    initial = _candidate(tmp_path)
    calibration = FrozenBehavioralCalibration(
        calibration_id="correctness-calibration-v1",
        policy_id=initial.policy.policy_id,
        direct_min_abs_predicted_target_delta=0.5,
        direct_max_mean_relative_closure=0.05,
    )
    candidate = replace(
        initial,
        policy=replace(initial.policy, calibration=calibration),
    )

    prepared = prepare_live_behavioral_request(candidate)

    assert prepared.status is BehavioralRequestStatus.READY
    assert prepared.request is not None
    controls = tuple(
        feature
        for feature in prepared.request.graph.features
        if feature.necessity_control_for is not None
    )
    assert len(controls) == 1
    assert controls[0].necessity_control_for == prepared.request.graph.features[0].node


def test_8192_selected_view_reads_only_policy_bounded_downstream_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_count = 8192
    selected_keys = tuple(FeatureKey(0, 0, index) for index in range(selected_count))
    control_keys = tuple(
        FeatureKey(0, 0, selected_count + index) for index in range(3)
    )
    selected_rows = np.asarray(
        [(key.layer, key.position, key.feature_id) for key in selected_keys],
        dtype=np.int64,
    )
    active = np.concatenate(
        (
            selected_rows,
            np.asarray(
                [(key.layer, key.position, key.feature_id) for key in control_keys],
                dtype=np.int64,
            ),
        ),
        axis=0,
    )
    matrix_storage = np.zeros((selected_count, 1), dtype=np.float32)
    feature_matrix = np.broadcast_to(
        matrix_storage,
        (selected_count, selected_count),
    )
    frontier = FrontierEvidence(
        graph_fingerprint=_fingerprint("large-graph"),
        provider_fingerprint=_fingerprint("large-provider"),
        decoder_fingerprint=_fingerprint("large-decoder"),
        cutoff_score=0.5,
        relative_cutoff_gap=0.01,
        cutoff_tie_count=0,
        near_cutoff_count=1,
        selected=tuple(
            _record(
                key,
                selected=True,
                rank=index,
                activation=1.0,
                effect=float(selected_count - index),
            )
            for index, key in enumerate(selected_keys)
        ),
        near_cutoff=(
            _record(
                control_keys[0],
                selected=False,
                rank=selected_count,
                activation=1.0,
                effect=0.0,
            ),
        ),
        bounds=FrontierBounds(max_selected=selected_count, max_near_cutoff=1),
    )
    candidate = LiveBehavioralCandidate(
        graph_path=tmp_path / "unused.npz",
        compact_result={
            "active_features": active,
            "activation_values": np.ones(len(active), dtype=np.float32),
            "selected_features": np.arange(selected_count, dtype=np.int64),
            "feature_row_node_indices": np.arange(selected_count, dtype=np.int64),
            "feature_feature_edges": feature_matrix,
            "logit_feature_edges": np.arange(
                selected_count,
                dtype=np.float32,
            ).reshape(1, -1),
        },
        frontier=frontier,
        trace_id="large-trace",
        trace_fingerprint=_fingerprint("large-trace"),
        target_fingerprint=_fingerprint("large-target"),
        step_fingerprint=_fingerprint("large-step"),
        semantic_fingerprint="semantic-v1",
        execution_fingerprint="execution-v1",
        provider_fingerprint=frontier.provider_fingerprint,
        decoder_fingerprint=frontier.decoder_fingerprint,
        prompt_token_ids=(1,),
        target_position=1,
        target_token_id=42,
        policy_mode=BehavioralProbeMode.SMOKE,
        policy=BehavioralProbePolicy(),
    )
    graph = SimpleNamespace(
        graph_fingerprint=frontier.graph_fingerprint,
        feature_ids=selected_rows,
        logit_token_ids=np.asarray([42], dtype=np.int64),
    )
    from nlp_research_project.exact_trace_bench.correctness import behavioral as module

    original = module._downstream_deltas
    columns: list[int] = []

    def recording_downstream_deltas(**kwargs):
        columns.append(int(kwargs["column"]))
        return original(**kwargs)

    monkeypatch.setattr(module, "_downstream_deltas", recording_downstream_deltas)

    view = _accepted_graph_view(candidate, cast(TypedCompactGraph, graph))

    assert len(view.features) == selected_count + 3
    assert len(columns) == candidate.policy.direct_sample_count
    assert len(set(columns)) == candidate.policy.direct_sample_count
    assert np.shares_memory(feature_matrix, matrix_storage)
    assert matrix_storage.nbytes == selected_count * np.dtype(np.float32).itemsize


class _FakeDecoderSource:
    decoder_fingerprint = _fingerprint("decoder")
    decoder_output_topology = "complete_downstream_block"

    def __init__(self, vectors: dict[FeatureKey, tuple[float, ...]]) -> None:
        self._vectors = vectors

    def decoder_vector(self, feature: FeatureKey) -> np.ndarray:
        return np.asarray(self._vectors[feature], dtype=np.float32)


def test_qualified_alias_uses_exact_decoder_geometry_and_reuses_necessity_pair(
    tmp_path: Path,
) -> None:
    initial = _candidate(tmp_path, controls=2)
    graph = load_typed_compact_graph(initial.graph_path)
    source = FeatureKey(0, 0, 4)
    substitute = FeatureKey(0, 1, 100)
    control = FeatureKey(0, 1, 101)
    candidate = replace(
        initial,
        qualified_aliases=(
            QualifiedAliasPair(
                source=source,
                substitute=substitute,
                selection_policy_id="frontier_alias_calibration_v1",
                calibration_fingerprint=_fingerprint("calibration"),
                comparison_evidence_fingerprint=_fingerprint("comparison"),
                baseline_graph_fingerprint=_fingerprint("baseline-graph"),
                candidate_graph_fingerprint=graph.graph_fingerprint,
                qualified_decoder_cosine=1.0,
            ),
        ),
        decoder_source=_FakeDecoderSource(
            {
                source: (1.0, 0.0),
                substitute: (1.0, 0.0),
                control: (0.0, 1.0),
            }
        ),
    )

    prepared = prepare_live_behavioral_request(candidate)

    assert prepared.status is BehavioralRequestStatus.READY
    assert prepared.request is not None
    assert len(prepared.request.aliases) == 1
    alias = prepared.request.aliases[0]
    assert alias.substitute_absolute_preactivation == pytest.approx(4.1)
    assert alias.selection_evidence.decoder_fingerprint == _fingerprint("decoder")
    assert alias.selection_evidence.least_squares_coefficient == 1.0
    assert alias.control_candidates[0].node.feature == 101
    assert alias.control_candidates[0].similarity_to_source == 0.0
    kinds = {item.kind for item in plan_behavioral_variants(prepared.request)}
    assert {
        VariantKind.NECESSITY_HIGH,
        VariantKind.NECESSITY_CONTROL,
        VariantKind.ALIAS_SUBSTITUTION,
    }.issubset(kinds)


def test_alias_source_outside_ordinary_cohort_is_forced_into_necessity_pair(
    tmp_path: Path,
) -> None:
    initial = _candidate(tmp_path, controls=2)
    graph = load_typed_compact_graph(initial.graph_path)
    source = FeatureKey(1, 1, 9)
    substitute = FeatureKey(1, 1, 201)
    matched_control = FeatureKey(1, 1, 202)
    alias_control = FeatureKey(1, 2, 203)
    compact = dict(initial.compact_result)
    compact["active_features"] = np.concatenate(
        (
            np.asarray(compact["active_features"]),
            np.asarray(
                [
                    (
                        matched_control.layer,
                        matched_control.position,
                        matched_control.feature_id,
                    ),
                    (
                        alias_control.layer,
                        alias_control.position,
                        alias_control.feature_id,
                    ),
                ],
                dtype=np.int64,
            ),
        ),
        axis=0,
    )
    compact["activation_values"] = np.concatenate(
        (
            np.asarray(compact["activation_values"]),
            np.asarray([-2.8, -2.7], dtype=np.float32),
        )
    )
    alias_frontier = replace(
        initial.frontier,
        near_cutoff=(
            _record(
                substitute,
                selected=False,
                rank=2,
                activation=float(np.float32(-2.9)),
                effect=0.01,
            ),
            _record(
                alias_control,
                selected=False,
                rank=3,
                activation=float(np.float32(-2.7)),
                effect=0.0,
            ),
        ),
        near_cutoff_count=2,
    )
    candidate = replace(
        initial,
        compact_result=compact,
        frontier=alias_frontier,
        policy=replace(initial.policy, necessity_sample_count=1),
        qualified_aliases=(
            QualifiedAliasPair(
                source=source,
                substitute=substitute,
                selection_policy_id="frontier_alias_calibration_v1",
                calibration_fingerprint=_fingerprint("calibration"),
                comparison_evidence_fingerprint=_fingerprint("comparison"),
                baseline_graph_fingerprint=_fingerprint("baseline-graph"),
                candidate_graph_fingerprint=graph.graph_fingerprint,
                qualified_decoder_cosine=1.0,
            ),
        ),
        decoder_source=_FakeDecoderSource(
            {
                source: (1.0, 0.0),
                substitute: (1.0, 0.0),
                alias_control: (0.0, 1.0),
            }
        ),
    )

    prepared = prepare_live_behavioral_request(candidate)

    assert prepared.status is BehavioralRequestStatus.READY
    assert prepared.request is not None
    owned_controls = tuple(
        feature
        for feature in prepared.request.graph.features
        if feature.necessity_control_for is not None
    )
    assert len(owned_controls) == 1
    assert owned_controls[0].node.feature == matched_control.feature_id
    assert owned_controls[0].necessity_control_for == FeatureNode(
        source.layer,
        source.position,
        source.feature_id,
    )
    kinds = {item.kind for item in plan_behavioral_variants(prepared.request)}
    assert VariantKind.ALIAS_SUBSTITUTION in kinds


def test_live_provider_decoder_source_uses_clt_and_plt_authoritative_rows() -> None:
    class _CltProvider:
        capabilities = TranscoderCapabilities(
            architecture="clt",
            checkpoint_format="fake",
            decoder_output_topology="cross_layer",
        )

        def get_decoder_block(self, layer: int, feature_ids: torch.Tensor) -> torch.Tensor:
            assert layer == 1
            assert feature_ids.tolist() == [7]
            return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    class _RowSource:
        released = False

        def materialize(self, keys: object, **kwargs: object) -> object:
            return SimpleNamespace(rows=torch.tensor([[5.0, 6.0]]))

        def release(self, reason: str) -> None:
            assert reason == "behavioral_alias_decoder_capture_complete"
            self.released = True

    row_source = _RowSource()

    class _PltProvider:
        capabilities = TranscoderCapabilities(
            architecture="plt",
            checkpoint_format="fake",
            decoder_output_topology="same_layer",
        )

        def create_decoder_row_source(self, layer: int, **kwargs: object) -> _RowSource:
            assert layer == 2
            return row_source

    clt = LiveProviderDecoderSource(
        SimpleNamespace(transcoders=_CltProvider()),
        decoder_fingerprint=_fingerprint("clt"),
    )
    plt = LiveProviderDecoderSource(
        SimpleNamespace(transcoders=_PltProvider()),
        decoder_fingerprint=_fingerprint("plt"),
    )

    assert clt.decoder_output_topology == "complete_downstream_block"
    assert clt.decoder_vector(FeatureKey(1, 0, 7)).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert plt.decoder_output_topology == "same_layer"
    assert plt.decoder_vector(FeatureKey(2, 0, 8)).tolist() == [5.0, 6.0]
    assert row_source.released is True


def test_missing_compact_or_mismatched_frontier_refuses_as_typed_unknown(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    missing = dict(candidate.compact_result)
    missing.pop("logit_feature_edges")

    absent = prepare_live_behavioral_request(replace(candidate, compact_result=missing))
    mismatch = prepare_live_behavioral_request(
        replace(
            candidate,
            frontier=replace(
                candidate.frontier,
                graph_fingerprint=_fingerprint("other-graph"),
            ),
        )
    )

    assert absent.status is BehavioralRequestStatus.UNKNOWN
    assert absent.assessment.status is BehavioralFaithfulnessStatus.UNKNOWN
    assert absent.assessment.reason_codes == ("missing_compact_field",)
    assert mismatch.status is BehavioralRequestStatus.UNKNOWN
    assert mismatch.assessment.reason_codes == ("frontier_identity_mismatch",)


def test_off_mode_is_not_applicable_without_touching_graph(tmp_path: Path) -> None:
    candidate = replace(
        _candidate(tmp_path),
        graph_path=tmp_path / "missing.npz",
        policy_mode=BehavioralProbeMode.OFF,
    )

    prepared = prepare_live_behavioral_request(candidate)

    assert prepared.status is BehavioralRequestStatus.NOT_APPLICABLE
    assert prepared.assessment.status is BehavioralFaithfulnessStatus.NOT_APPLICABLE
    assert prepared.assessment.reason_codes == ("behavioral_probe_mode_off",)


@dataclass(frozen=True)
class _FakeSerializedReport:
    evidence_fingerprint: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": "behavioral_faithfulness_report_v1",
                "evidence_fingerprint": self.evidence_fingerprint,
            },
            sort_keys=True,
        )


def test_persistence_uses_one_sibling_owned_json_seam(tmp_path: Path) -> None:
    report = _FakeSerializedReport(_fingerprint("behavioral-evidence"))
    path = tmp_path / "behavioral.json"

    persist_sibling_behavioral_report(report, path)

    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(report.to_json())


def test_maps_sibling_verdict_metrics_reasons_and_evidence_identity() -> None:
    raw_evidence_fingerprint = hashlib.sha256(b"behavioral-evidence").hexdigest()
    sibling = cast(
        SiblingBehavioralReport,
        SimpleNamespace(
            verdict=FaithfulnessVerdict.SUPPORTED,
            policy_id="behavioral_closure_v1",
            calibration_id="calibration-a",
            evidence_completeness=EvidenceCompleteness.COMPLETE,
            runtime_status=RuntimeExecutionStatus.COMPLETE,
            variants_planned=4,
            variants_completed=4,
            no_op_required=True,
            no_op_passed=True,
            metrics=BehavioralAggregateMetrics(
                direct_mean_abs_closure=0.01,
                direct_mean_relative_closure=0.02,
                direct_max_relative_closure=0.04,
                direct_sign_agreement=1.0,
                necessity_high_vs_control_separation=0.5,
                necessity_predicted_realized_spearman=0.9,
                necessity_median_high_control_effect_ratio=2.5,
                alias_mean_abs_target_delta=None,
                alias_mean_abs_closure=None,
                alias_relative_effect_error=None,
                alias_substitution_vs_source_ablation=None,
                alias_control_vs_source_ablation=None,
                alias_substitution_advantage=None,
                downstream_mean_abs_closure=0.02,
                downstream_mean_relative_closure=0.03,
                downstream_p95_relative_closure=0.05,
            ),
            reasons=("Frozen threshold satisfied",),
            refusal=None,
            evidence_fingerprint=raw_evidence_fingerprint,
        ),
    )

    mapped = map_sibling_behavioral_report(sibling)

    assert mapped.status is BehavioralFaithfulnessStatus.SUPPORTED
    assert mapped.metrics["variants_completed"] == 4
    assert mapped.metrics["direct_mean_relative_closure"] == 0.02
    assert mapped.metrics["necessity_predicted_realized_spearman"] == 0.9
    assert mapped.metrics["alias_substitution_advantage"] is None
    assert mapped.reason_codes == ("sibling_frozen_threshold_satisfied",)
    assert mapped.evidence_fingerprint == _fingerprint("behavioral-evidence")
