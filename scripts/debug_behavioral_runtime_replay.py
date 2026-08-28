#!/usr/bin/env python3
"""Replay a persisted behavioral probe without rerunning the full trace.

This is a deliberately narrow GPU debugging harness.  It treats the persisted
sibling behavioral report as the intervention recipe authority, while binding
that recipe back to the prepared trace spec, typed graph, and frontier before
loading a model.  Use ``--validate-only`` for the CPU-only half of the loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from circuit_tracer.verification import (
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionSemantics,
    InterventionVariant,
    NNSightInterventionRuntime,
    OrderingAdmissionMode,
    PreactivationIntervention,
    RuntimeExecutionStatus,
    TargetFunctional,
    TargetState,
    TraceIdentity,
    VariantKind,
)

from nlp_research_project.exact_trace_bench.correctness.frontier_artifact import (
    load_bounded_frontier_artifact,
)
from nlp_research_project.exact_trace_bench.trace_runtime.provider import (
    ProviderLoadPolicy,
    get_model_transcoder_metadata,
)
from nlp_research_project.exact_trace_bench.typed_compact_graph import (
    _fingerprint_json,
    load_typed_compact_graph,
)


REPORT_SCHEMA = "behavioral_faithfulness_report"
CURRENT_REPORT_SCHEMA_VERSION = 2
HISTORICAL_REPORT_SCHEMA_VERSIONS = frozenset({1})
SUPPORTED_REPORT_SCHEMA_VERSIONS = frozenset(
    {CURRENT_REPORT_SCHEMA_VERSION, *HISTORICAL_REPORT_SCHEMA_VERSIONS}
)
DEFAULT_DEADLINE_SECONDS = 120.0
DEFAULT_CLEANUP_RESERVE_SECONDS = 1.0
DEFAULT_PREDICTED_BASELINE_SECONDS = 1.0
DEFAULT_PREDICTED_VARIANT_SECONDS = 1.0
DEFAULT_NO_OP_TOLERANCE = 1e-5


class ReplayRefusal(ValueError):
    """Persisted evidence is insufficient or inconsistent for a safe replay."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayRefusal(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ReplayRefusal(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayRefusal(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayRefusal(f"{label} must be an integer")
    return value


def _number(value: object, label: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayRefusal(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayRefusal(f"{label} must be a finite number")
    return result


def _required_number(value: object, label: str) -> float:
    result = _number(value, label)
    assert result is not None
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_behavioral_payload(path: Path) -> Mapping[str, Any]:
    try:
        document = _mapping(json.loads(path.read_text(encoding="utf-8")), "report")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayRefusal(f"cannot load behavioral report: {path}") from exc
    if document.get("schema") != REPORT_SCHEMA:
        raise ReplayRefusal("unsupported behavioral report schema")
    schema_version = document.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS
    ):
        raise ReplayRefusal("unsupported behavioral report schema_version")
    payload = _mapping(document.get("report"), "report.report")
    expected = _string(document.get("evidence_fingerprint"), "evidence_fingerprint")
    observed = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    if observed != expected:
        raise ReplayRefusal("behavioral report evidence_fingerprint mismatch")
    return payload


def load_trace_spec(path: Path, trace_id: str) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReplayRefusal(f"cannot load trace specs: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = _mapping(json.loads(line), f"trace spec line {line_number}")
        except json.JSONDecodeError as exc:
            raise ReplayRefusal(
                f"invalid trace spec JSON at line {line_number}"
            ) from exc
        if item.get("trace_id") == trace_id:
            matches.append(item)
    if len(matches) != 1:
        raise ReplayRefusal(
            f"expected exactly one trace spec for {trace_id!r}; found {len(matches)}"
        )
    return matches[0]


def _feature_node(value: object, label: str) -> FeatureNode:
    node = _mapping(value, label)
    return FeatureNode(
        layer=_integer(node.get("layer"), f"{label}.layer"),
        position=_integer(node.get("position"), f"{label}.position"),
        feature=_integer(node.get("feature"), f"{label}.feature"),
    )


def _feature_value(value: object, label: str) -> FeatureValue:
    item = _mapping(value, label)
    preactivation = _number(item.get("preactivation"), f"{label}.preactivation")
    assert preactivation is not None
    return FeatureValue(
        node=_feature_node(item.get("node"), f"{label}.node"),
        preactivation=preactivation,
    )


def _variant(value: object, label: str) -> InterventionVariant:
    item = _mapping(value, label)
    interventions = tuple(
        PreactivationIntervention(
            node=_feature_node(raw.get("node"), f"{label}.interventions[{index}].node"),
            absolute_value=_required_number(
                raw.get("absolute_value"),
                f"{label}.interventions[{index}].absolute_value",
            ),
            graph_baseline_value=_number(
                raw.get("graph_baseline_value"),
                f"{label}.interventions[{index}].graph_baseline_value",
                optional=True,
            ),
            graph_delta=_number(
                raw.get("graph_delta"),
                f"{label}.interventions[{index}].graph_delta",
                optional=True,
            ),
        )
        for index, raw_value in enumerate(
            _sequence(item.get("interventions"), f"{label}.interventions")
        )
        for raw in (_mapping(raw_value, f"{label}.interventions[{index}]"),)
    )
    downstream = tuple(
        _feature_value(raw, f"{label}.predicted_downstream_feature_deltas[{index}]")
        for index, raw in enumerate(
            _sequence(
                item.get("predicted_downstream_feature_deltas"),
                f"{label}.predicted_downstream_feature_deltas",
            )
        )
    )
    return InterventionVariant(
        variant_id=_string(item.get("variant_id"), f"{label}.variant_id"),
        kind=VariantKind(_string(item.get("kind"), f"{label}.kind")),
        semantics=InterventionSemantics(
            _string(item.get("semantics"), f"{label}.semantics")
        ),
        interventions=interventions,
        predicted_target_delta=_number(
            item.get("predicted_target_delta"),
            f"{label}.predicted_target_delta",
            optional=True,
        ),
        predicted_downstream_feature_deltas=downstream,
    )


def reconstruct_request(
    payload: Mapping[str, Any],
    *,
    variant_selector: str,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> InterventionExecutionRequest:
    identity_payload = _mapping(payload.get("trace_identity"), "trace_identity")
    prompt_token_ids = tuple(
        _integer(value, f"trace_identity.prompt_token_ids[{index}]")
        for index, value in enumerate(
            _sequence(identity_payload.get("prompt_token_ids"), "prompt_token_ids")
        )
    )
    identity = TraceIdentity(
        trace_id=_string(identity_payload.get("trace_id"), "trace_identity.trace_id"),
        graph_fingerprint=_string(
            identity_payload.get("graph_fingerprint"),
            "trace_identity.graph_fingerprint",
        ),
        provider_fingerprint=_string(
            identity_payload.get("provider_fingerprint"),
            "trace_identity.provider_fingerprint",
        ),
        semantic_fingerprint=_string(
            identity_payload.get("semantic_fingerprint"),
            "trace_identity.semantic_fingerprint",
        ),
        execution_fingerprint=_string(
            identity_payload.get("execution_fingerprint"),
            "trace_identity.execution_fingerprint",
        ),
        prompt_token_ids=prompt_token_ids,
        target_position=_integer(
            identity_payload.get("target_position"), "trace_identity.target_position"
        ),
        target_token_id=_integer(
            identity_payload.get("target_token_id"), "trace_identity.target_token_id"
        ),
    )
    recipes = tuple(
        _variant(value, f"variant_recipes[{index}]")
        for index, value in enumerate(
            _sequence(payload.get("variant_recipes"), "variant_recipes")
        )
    )
    no_ops = tuple(item for item in recipes if item.kind is VariantKind.NO_OP)
    if len(no_ops) != 1 or recipes[0] is not no_ops[0]:
        raise ReplayRefusal("persisted recipes must contain one leading no-op")
    if variant_selector == "all":
        selected = recipes
    else:
        matches = tuple(item for item in recipes if item.variant_id == variant_selector)
        if len(matches) != 1:
            raise ReplayRefusal(f"unknown or ambiguous variant: {variant_selector!r}")
        if matches[0].kind is not VariantKind.DIRECT_DOUBLE:
            raise ReplayRefusal("a named replay variant must be direct_double")
        selected = (no_ops[0], matches[0])
    observed_nodes = tuple(
        sorted(
            {
                value.node
                for variant in selected
                for value in variant.predicted_downstream_feature_deltas
            }
        )
    )
    return InterventionExecutionRequest(
        identity=identity,
        target=TargetState(TargetFunctional(_string(payload.get("target"), "target"))),
        variants=selected,
        observed_downstream_nodes=observed_nodes,
        deadline_seconds=deadline_seconds,
        cleanup_reserve_seconds=DEFAULT_CLEANUP_RESERVE_SECONDS,
        predicted_baseline_seconds=DEFAULT_PREDICTED_BASELINE_SECONDS,
        predicted_variant_seconds=DEFAULT_PREDICTED_VARIANT_SECONDS,
    )


def validate_bound_evidence(
    request: InterventionExecutionRequest,
    *,
    trace_spec: Mapping[str, Any],
    graph_path: Path,
    frontier_path: Path,
) -> Mapping[str, Any]:
    identity = request.identity
    graph = load_typed_compact_graph(graph_path)
    frontier = load_bounded_frontier_artifact(frontier_path)
    checks = {
        "spec.trace_id": (trace_spec.get("trace_id"), identity.trace_id),
        "spec.target_position": (
            trace_spec.get("target_position"),
            identity.target_position,
        ),
        "spec.target_token_id": (
            trace_spec.get("target_token_id"),
            identity.target_token_id,
        ),
        "spec.prefix_token_count": (
            trace_spec.get("prefix_token_count"),
            len(identity.prompt_token_ids),
        ),
        "spec.generated_index": (trace_spec.get("generated_index"), graph.step_idx),
        "graph.graph_fingerprint": (
            graph.graph_fingerprint,
            identity.graph_fingerprint,
        ),
        "graph.provider_fingerprint": (
            graph.provider_fingerprint,
            identity.provider_fingerprint,
        ),
        "graph.prompt_token_ids": (
            tuple(int(value) for value in graph.token_ids.tolist()),
            identity.prompt_token_ids,
        ),
        "frontier.graph_fingerprint": (
            frontier.evidence.graph_fingerprint,
            graph.graph_fingerprint,
        ),
        "frontier.provider_fingerprint": (
            frontier.evidence.provider_fingerprint,
            graph.provider_fingerprint,
        ),
    }
    mismatches = [
        name for name, (observed, expected) in checks.items() if observed != expected
    ]
    if mismatches:
        raise ReplayRefusal("identity binding failed: " + ", ".join(mismatches))
    graph_knobs = _mapping(trace_spec.get("graph_knobs"), "trace_spec.graph_knobs")
    return {
        "trace_id": identity.trace_id,
        "graph_fingerprint": graph.graph_fingerprint,
        "provider_fingerprint": graph.provider_fingerprint,
        "frontier_fingerprint": frontier.content_fingerprint,
        "model_name": _string(graph_knobs.get("model_name"), "graph_knobs.model_name"),
        "provider_family": _string(
            graph_knobs.get("transcoder_provider_family"),
            "graph_knobs.transcoder_provider_family",
        ),
        "variants": [item.variant_id for item in request.variants],
    }


def validate_loaded_provider_binding(
    request: InterventionExecutionRequest,
    model: Any,
    *,
    graph_path: Path,
) -> str:
    """Bind the loaded provider to the graph identity and full trace receipt."""

    trace_path = graph_path.with_name("trace.json")
    try:
        trace = _mapping(json.loads(trace_path.read_text(encoding="utf-8")), "trace")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayRefusal(f"cannot load trace receipt: {trace_path}") from exc
    persisted_trace_id = _string(trace.get("trace_id"), "trace.trace_id")
    if persisted_trace_id != request.identity.trace_id:
        raise ReplayRefusal(
            "trace receipt identity mismatch: "
            f"expected={request.identity.trace_id}, observed={persisted_trace_id}"
        )
    persisted = _mapping(trace.get("transcoder"), "trace.transcoder")
    requested = _mapping(persisted.get("requested"), "trace.transcoder.requested")
    detected = _mapping(persisted.get("detected"), "trace.transcoder.detected")
    _mapping(
        detected.get("provider_fingerprint"),
        "trace.transcoder.detected.provider_fingerprint",
    )
    try:
        persisted_requested_fingerprint = _fingerprint_json(requested)
        persisted_full_fingerprint = _fingerprint_json(persisted)
    except (TypeError, ValueError) as exc:
        raise ReplayRefusal("persisted provider metadata is not canonical") from exc
    expected = request.identity.provider_fingerprint
    if persisted_requested_fingerprint != expected:
        raise ReplayRefusal(
            "persisted requested provider fingerprint mismatch: "
            f"expected={expected}, observed={persisted_requested_fingerprint}"
        )

    metadata = get_model_transcoder_metadata(model)
    if not isinstance(metadata, Mapping) or not metadata:
        raise ReplayRefusal("loaded provider metadata unavailable")
    try:
        loaded_full_fingerprint = _fingerprint_json(metadata)
    except (TypeError, ValueError) as exc:
        raise ReplayRefusal("loaded provider metadata is not canonical") from exc
    if loaded_full_fingerprint != persisted_full_fingerprint:
        raise ReplayRefusal(
            "loaded provider metadata mismatch: "
            f"expected={persisted_full_fingerprint}, "
            f"observed={loaded_full_fingerprint}"
        )
    return loaded_full_fingerprint


def require_complete_result(
    request: InterventionExecutionRequest,
    result: Any,
    *,
    no_op_tolerance: float,
) -> Mapping[str, Any]:
    failures: list[str] = []
    if result.status is not RuntimeExecutionStatus.COMPLETE:
        failures.append(f"status={result.status.value}")
    if result.refusal is not None:
        failures.append(f"refusal={result.refusal.code}: {result.refusal.detail}")
    if not result.cleanup_completed:
        failures.append("cleanup_completed=false")
    if result.ordering_admission_mode is not OrderingAdmissionMode.CANDIDATE_SMOKE:
        failures.append("ordering_admission_mode is not candidate_smoke")
    expected_ids = tuple(variant.variant_id for variant in request.variants)
    observed_ids = tuple(observation.variant_id for observation in result.observations)
    if observed_ids != expected_ids:
        failures.append(
            f"observation order mismatch: expected={expected_ids!r}, observed={observed_ids!r}"
        )
    no_op_delta: float | None = None
    if result.baseline is None:
        failures.append("baseline is missing")
    elif observed_ids and observed_ids[0] == "no_op":
        no_op_delta = abs(
            result.observations[0].target_value - result.baseline.target_value
        )
        if no_op_delta > no_op_tolerance:
            failures.append(
                f"no-op target parity delta {no_op_delta:.9g} exceeds {no_op_tolerance:.9g}"
            )
    else:
        failures.append("ordered no-op observation is missing")
    if failures:
        raise ReplayRefusal("runtime replay failed closed: " + "; ".join(failures))
    return {
        "status": result.status.value,
        "ordering_admission_mode": result.ordering_admission_mode.value,
        "variants": list(observed_ids),
        "baseline_target_value": result.baseline.target_value,
        "no_op_target_delta": no_op_delta,
        "elapsed_seconds": result.elapsed_seconds,
        "cleanup_completed": result.cleanup_completed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-report", required=True, type=Path)
    parser.add_argument("--trace-specs", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        help="One persisted direct_double variant ID, or 'all'",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="Cooperative replay deadline (default: 120, contract maximum)",
    )
    parser.add_argument(
        "--no-op-tolerance",
        type=float,
        default=DEFAULT_NO_OP_TOLERANCE,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and reconstruct on CPU without loading the model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 < args.deadline_seconds <= DEFAULT_DEADLINE_SECONDS:
        raise ReplayRefusal("deadline-seconds must be in (0, 120]")
    if not math.isfinite(args.no_op_tolerance) or args.no_op_tolerance < 0:
        raise ReplayRefusal("no-op-tolerance must be a finite non-negative number")
    payload = load_behavioral_payload(args.behavioral_report)
    request = reconstruct_request(
        payload,
        variant_selector=args.variant,
        deadline_seconds=args.deadline_seconds,
    )
    spec = load_trace_spec(args.trace_specs, request.identity.trace_id)
    validation = validate_bound_evidence(
        request,
        trace_spec=spec,
        graph_path=args.graph,
        frontier_path=args.frontier,
    )
    if args.validate_only:
        print(_canonical({"status": "validated", **validation}))
        return 0

    graph_knobs = _mapping(spec.get("graph_knobs"), "trace_spec.graph_knobs")
    model = ProviderLoadPolicy.from_scenario(graph_knobs).load()
    validate_loaded_provider_binding(request, model, graph_path=args.graph)
    result = NNSightInterventionRuntime(
        model,
        ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE,
    ).evaluate(request)
    outcome = require_complete_result(
        request,
        result,
        no_op_tolerance=args.no_op_tolerance,
    )
    print(_canonical({"status": "complete", **validation, "runtime": outcome}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
