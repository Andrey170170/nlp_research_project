from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph_compare import compare_artifact_dirs
from .io_utils import read_json, write_json


EXPECTED_CASES = ("execution_b128", "execution_b256", "execution_b512")


def _telemetry_evidence(artifacts_dir: Path) -> dict[str, Any]:
    completion_rows: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("prompt_*/completion_*/telemetry.live.jsonl")):
        refreshes: list[dict[str, Any]] = []
        semantic_schedule: list[dict[str, int]] = []
        execution_batch_count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                name = record.get("name")
                attrs = record.get("attrs") or {}
                if name == "phase4.refresh":
                    refreshes.append(
                        {
                            "refresh_index": attrs.get("refresh_index"),
                            "pending_hash": attrs.get("pending_hash"),
                            "order_hash": attrs.get(
                                "ranker_frontier_selected_order_hash"
                            ),
                            "membership_hash": attrs.get(
                                "ranker_frontier_selected_membership_hash"
                            ),
                            "selected_count": attrs.get(
                                "ranker_frontier_selected_count"
                            ),
                        }
                    )
                elif name == "phase4.feature_batch":
                    start = attrs.get("phase4_semantic_batch_index_start")
                    rows = attrs.get("phase4_semantic_batch_rows")
                    refresh_index = attrs.get("scheduler_refresh_index")
                    if not isinstance(start, int) or not isinstance(rows, (list, tuple)):
                        raise ValueError(
                            f"missing semantic batch telemetry in {path}"
                        )
                    semantic_schedule.extend(
                        {
                            "semantic_batch_index": start + offset,
                            "rows": int(row_count),
                            "refresh_index": int(refresh_index),
                        }
                        for offset, row_count in enumerate(rows)
                    )
                    execution_batch_count = max(
                        execution_batch_count,
                        int(attrs.get("phase4_execution_batch_count") or 0),
                    )
        if not refreshes or not semantic_schedule:
            raise ValueError(f"incomplete Phase 4 telemetry in {path}")
        if any(
            refresh[key] is None
            for refresh in refreshes
            for key in ("refresh_index", "pending_hash", "order_hash", "membership_hash")
        ):
            raise ValueError(f"missing refresh identity telemetry in {path}")
        completion_rows.append(
            {
                "completion": str(path.parent.relative_to(artifacts_dir)),
                "refreshes": refreshes,
                "semantic_schedule": semantic_schedule,
                "semantic_batch_count": len(semantic_schedule),
                "execution_batch_count": execution_batch_count,
            }
        )
    if not completion_rows:
        raise ValueError(f"no incremental telemetry found under {artifacts_dir}")
    return {"completions": completion_rows}


def _discover_cases(run_root: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for scenario_path in sorted(run_root.glob("*/scenario.json")):
        scenario = read_json(scenario_path)
        case = scenario.get("validation_case")
        if case in EXPECTED_CASES:
            if case in cases:
                raise ValueError(f"duplicate validation case {case}")
            cases[str(case)] = scenario_path.parent
    missing = [case for case in EXPECTED_CASES if case not in cases]
    if missing:
        raise ValueError(f"missing validation cases: {', '.join(missing)}")
    return cases


def validate_phase4_static_coalescing_run(
    run_root: Path,
    *,
    weighted_edge_jaccard_min: float = 0.999999,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        case_roots = _discover_cases(run_root)
    except ValueError as exc:
        return {"status": "fail", "passed": False, "failures": [str(exc)], "cases": {}}

    cases: dict[str, dict[str, Any]] = {}
    for case, scenario_root in case_roots.items():
        try:
            result = read_json(scenario_root / "result.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{case}: cannot read result: {exc}")
            result = {}
        if result.get("status") != "success":
            failures.append(f"{case}: trace status is {result.get('status')!r}")
        baseline = result.get("baseline_check") or {}
        if baseline.get("status") != "gate_pass" or baseline.get("passed") is not True:
            failures.append(f"{case}: pinned baseline gate did not pass")
        try:
            evidence = _telemetry_evidence(scenario_root / "artifacts")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{case}: {exc}")
            evidence = {"completions": []}
        cases[case] = {
            "scenario_root": str(scenario_root),
            "baseline_gate": baseline.get("status"),
            **evidence,
        }

    reference_case = EXPECTED_CASES[0]
    reference = cases[reference_case]
    comparisons: list[dict[str, Any]] = []
    for case in EXPECTED_CASES[1:]:
        candidate = cases[case]
        # Execution counts intentionally differ, so compare only semantic and refresh evidence.
        reference_contract = [
            {
                "completion": row["completion"],
                "refreshes": row["refreshes"],
                "semantic_schedule": row["semantic_schedule"],
                "semantic_batch_count": row["semantic_batch_count"],
            }
            for row in reference["completions"]
        ]
        candidate_contract = [
            {
                "completion": row["completion"],
                "refreshes": row["refreshes"],
                "semantic_schedule": row["semantic_schedule"],
                "semantic_batch_count": row["semantic_batch_count"],
            }
            for row in candidate["completions"]
        ]
        semantic_equal = candidate_contract == reference_contract
        if not semantic_equal:
            failures.append(f"{case}: semantic schedule or refresh identity diverged")

        try:
            graph = compare_artifact_dirs(
                Path(reference["scenario_root"]) / "artifacts",
                Path(candidate["scenario_root"]) / "artifacts",
            )
        except Exception as exc:  # noqa: BLE001 - report analysis failure as gate failure
            graph = {"comparison_complete": False, "error": str(exc)}
        graph_requirements = {
            "comparison_complete": graph.get("comparison_complete") is True,
            "feature_topology_exact": graph.get("overall_mean_feature_jaccard") == 1.0,
            "edge_topology_exact": graph.get("overall_mean_edge_jaccard") == 1.0,
            "top256_exact": graph.get("overall_mean_top256_edge_jaccard") == 1.0,
            "weighted_edge_within_tolerance": float(
                graph.get("overall_mean_weighted_edge_jaccard") or 0.0
            )
            >= weighted_edge_jaccard_min,
        }
        if not all(graph_requirements.values()):
            failures.append(f"{case}: compact graph parity gate failed")
        comparisons.append(
            {
                "reference_case": reference_case,
                "candidate_case": case,
                "semantic_and_refresh_contract_equal": semantic_equal,
                "graph_requirements": graph_requirements,
                "graph_metrics": {
                    key: graph.get(key)
                    for key in (
                        "overall_mean_feature_jaccard",
                        "overall_mean_edge_jaccard",
                        "overall_mean_weighted_edge_jaccard",
                        "overall_mean_top256_edge_jaccard",
                    )
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "weighted_edge_jaccard_min": weighted_edge_jaccard_min,
        "failures": failures,
        "cases": cases,
        "comparisons": comparisons,
    }


def write_phase4_static_validation_report(run_root: Path, report: dict[str, Any]) -> Path:
    output_path = run_root / "phase4_static_coalescing_validation.json"
    write_json(output_path, report)
    return output_path
