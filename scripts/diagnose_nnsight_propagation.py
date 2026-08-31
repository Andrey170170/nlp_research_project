#!/usr/bin/env python3
"""Run the focused, non-promoting propagated-ordering diagnostic on a GPU.

This is neither a trace launcher nor a qualification gate.  It binds the
diagnostic to persisted production evidence, loads that exact provider stack,
delegates boundary localization to the sibling library, and writes one
project-owned immutable-workspace receipt.  The result can explain an ordering
qualification failure but cannot qualify or promote a runtime.
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
    collect_input_artifacts,
    collect_workspace_provenance,
)
from nlp_research_project.exact_trace_bench.correctness.propagated_ordering_diagnostic_runner import (
    PropagatedOrderingDiagnosticError,
    build_diagnostic_receipt,
    diagnostic_scope_from_graph_knobs,
    invoke_sibling_diagnostic,
    write_diagnostic_receipt,
)
from nlp_research_project.exact_trace_bench.full_answer.runtime_environment import (
    capture_runtime_environment,
)
from nlp_research_project.exact_trace_bench.trace_runtime.provider import (
    ProviderLoadPolicy,
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PropagatedOrderingDiagnosticError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-report", required=True, type=Path)
    parser.add_argument("--trace-specs", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=300.0,
        help="Per-request cooperative deadline forwarded to the sibling diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_raw = os.environ.get("WORKSPACE_ROOT")
    library_raw = os.environ.get("LIB_WORKSPACE_ROOT")
    if not workspace_raw or not library_raw:
        raise PropagatedOrderingDiagnosticError(
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
    scope = diagnostic_scope_from_graph_knobs(graph_knobs)
    binding["diagnostic_scope"] = scope
    binding["diagnostic_kind"] = "propagated_ordering_boundary_localization"
    provenance = {
        "request_binding": binding,
        "input_artifacts": inputs,
        "workspace": workspace,
        "diagnostic_kind": "propagated_ordering_boundary_localization",
        "diagnostic_only_no_qualification_claim": True,
    }

    model = ProviderLoadPolicy.from_scenario(graph_knobs).load()
    binding["loaded_provider_fingerprint"] = validate_loaded_provider_binding(
        request,
        model,
        graph_path=args.graph,
    )

    import circuit_tracer.verification as sibling_api

    sibling_receipt = invoke_sibling_diagnostic(
        sibling_api,
        model=model,
        execution_requests=(request,),
        scope=scope,
        provenance=provenance,
    )
    diagnostic_receipt = build_diagnostic_receipt(
        sibling_receipt=sibling_receipt,
        request_binding=binding,
        input_artifacts=inputs,
        workspace_provenance=workspace,
        runtime_environment=capture_runtime_environment(),
    )
    write_diagnostic_receipt(args.output, diagnostic_receipt)
    print(json.dumps(diagnostic_receipt, sort_keys=True))
    return 0 if diagnostic_receipt["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PropagatedOrderingDiagnosticError,
        QualificationGateError,
        ReplayRefusal,
    ) as exc:
        raise SystemExit(f"propagated-ordering diagnostic refused: {exc}") from exc
