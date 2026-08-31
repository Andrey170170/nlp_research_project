#!/usr/bin/env python3
"""Run the targeted model-backed NNSight ordering qualification on a GPU.

This is intentionally not a trace launcher.  It binds the qualification cases
to persisted production evidence, loads that exact provider stack, delegates
the independent ordering experiment to the sibling library, and writes a
project-owned immutable-workspace receipt.  It never promotes the capability.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scripts.debug_behavioral_runtime_replay import (
    ReplayRefusal,
    load_behavioral_payload,
    load_trace_spec,
    reconstruct_request,
    validate_bound_evidence,
    validate_loaded_provider_binding,
)

from nlp_research_project.exact_trace_bench.correctness.ordering_qualification_runner import (
    QualificationGateError,
    build_gate_receipt,
    collect_input_artifacts,
    collect_workspace_provenance,
    invoke_sibling_qualification,
    write_gate_receipt,
)
from nlp_research_project.exact_trace_bench.full_answer.runtime_environment import (
    capture_runtime_environment,
)
from nlp_research_project.exact_trace_bench.trace_runtime.provider import (
    ProviderLoadPolicy,
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationGateError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationGateError(f"{label} must be a non-empty string")
    return value


def _scope_from_graph_knobs(graph_knobs: Mapping[str, Any]) -> dict[str, Any]:
    architecture = _string(
        graph_knobs.get("transcoder_architecture"),
        "graph_knobs.transcoder_architecture",
    )
    if architecture != "plt":
        raise QualificationGateError(
            "this qualification lane is scoped to same-layer PLT providers"
        )
    model_name = _string(graph_knobs.get("model_name"), "graph_knobs.model_name")
    if "gemma-3" not in model_name.lower():
        raise QualificationGateError(
            "this qualification lane is scoped to Gemma 3"
        )
    return {
        "model_family": "gemma3",
        "provider_architecture": architecture,
        "decoder_output_topology": "same_layer",
        "dtype": "bfloat16",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-report", required=True, type=Path)
    parser.add_argument("--trace-specs", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=120.0,
        help="Per-request cooperative deadline forwarded to the sibling gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_raw = os.environ.get("WORKSPACE_ROOT")
    library_raw = os.environ.get("LIB_WORKSPACE_ROOT")
    if not workspace_raw or not library_raw:
        raise QualificationGateError(
            "WORKSPACE_ROOT and LIB_WORKSPACE_ROOT are required"
        )
    workspace = collect_workspace_provenance(
        workspace_root=Path(workspace_raw),
        library_root=Path(library_raw),
    )
    inputs = collect_input_artifacts(
        {
            "behavioral_report": args.behavioral_report,
            "trace_specs": args.trace_specs,
            "graph": args.graph,
            "trace_receipt": args.graph.with_name("trace.json"),
            "frontier": args.frontier,
        }
    )

    payload = load_behavioral_payload(args.behavioral_report)
    request = reconstruct_request(
        payload,
        variant_selector="all",
        deadline_seconds=args.deadline_seconds,
    )
    spec = load_trace_spec(args.trace_specs, request.identity.trace_id)
    binding = dict(
        validate_bound_evidence(
            request,
            trace_spec=spec,
            graph_path=args.graph,
            frontier_path=args.frontier,
        )
    )
    graph_knobs = _mapping(spec.get("graph_knobs"), "trace_spec.graph_knobs")
    scope = _scope_from_graph_knobs(graph_knobs)
    binding["qualification_scope"] = scope
    provenance = {
        "request_binding": binding,
        "input_artifacts": inputs,
        "workspace": workspace,
        "repeat_index": args.repeat_index,
    }

    model = ProviderLoadPolicy.from_scenario(graph_knobs).load()
    binding["loaded_provider_fingerprint"] = validate_loaded_provider_binding(
        request,
        model,
        graph_path=args.graph,
    )

    import circuit_tracer.verification as sibling_api

    sibling_receipt = invoke_sibling_qualification(
        sibling_api,
        model=model,
        execution_requests=(request,),
        scope=scope,
        provenance=provenance,
    )
    gate_receipt = build_gate_receipt(
        sibling_receipt=sibling_receipt,
        request_binding=binding,
        input_artifacts=inputs,
        workspace_provenance=workspace,
        runtime_environment=capture_runtime_environment(),
        repeat_index=args.repeat_index,
    )
    write_gate_receipt(args.output, gate_receipt)
    print(json.dumps(gate_receipt, sort_keys=True))
    return 0 if gate_receipt["status"] == "qualified" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationGateError, ReplayRefusal) as exc:
        raise SystemExit(f"ordering qualification refused: {exc}") from exc
