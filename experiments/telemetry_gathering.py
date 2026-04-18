from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from extract_utils import ensure_dir, iter_jsonl, read_json, write_csv


DEFAULT_INPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _sanitize_column_fragment(value: str) -> str:
    fragment = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return fragment or "unknown"


def _infer_cluster(scenario_root: Path | None, scenario: dict[str, Any]) -> str | None:
    if scenario.get("cluster"):
        return str(scenario["cluster"])
    if scenario_root is None:
        return None
    for part in scenario_root.parts:
        if part in {"ascend", "cardinal"}:
            return part
    return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _relative_to_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _looks_like_completion_dir(path: Path) -> bool:
    return path.name.startswith("completion_") and path.parent.name.startswith(
        "prompt_"
    )


def _load_summary_run_metadata(scenario_root: Path | None) -> dict[str, Any]:
    if scenario_root is None:
        return {}
    candidates = [scenario_root / "summary.json", scenario_root.parent / "summary.json"]
    for summary_path in candidates:
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        run_metadata = summary.get("run_metadata")
        if isinstance(run_metadata, dict):
            return run_metadata
        if any(
            key in summary
            for key in (
                "run_id",
                "launch_id",
                "run_name",
                "run_description",
                "run_goal",
            )
        ):
            return {
                "run_id": summary.get("run_id") or summary.get("launch_id"),
                "run_name": summary.get("run_name"),
                "run_description": summary.get("run_description"),
                "run_goal": summary.get("run_goal"),
            }
    return {}


def _find_scenario_root(completion_dir: Path, input_root: Path) -> Path | None:
    resolved_completion_dir = completion_dir.resolve()
    resolved_input_root = input_root.resolve()
    for candidate in [resolved_completion_dir, *resolved_completion_dir.parents]:
        if (candidate / "result.json").exists():
            return candidate
        if candidate == resolved_input_root:
            break
    return None


def _event_scope(event: dict[str, Any]) -> str | None:
    scope = event.get("scope")
    if not isinstance(scope, str):
        return None
    normalized = scope.strip().lower()
    return normalized or None


def _event_step_index(event: dict[str, Any]) -> int | None:
    for key in ("trace_step_index", "step_index"):
        step_index = _to_int(event.get(key))
        if step_index is not None:
            return step_index
    return None


def _flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    core_keys = (
        "ts_ns",
        "scope",
        "name",
        "phase",
        "step_index",
        "trace_step_index",
        "batch_index",
        "elapsed_ms",
        "count",
        "event_index",
        "prompt_id",
        "completion_id",
        "phase4_feature_batch_size",
        "phase4_feature_batch_planner_status",
    )
    for key in core_keys:
        value = event.get(key)
        if _is_scalar(value):
            row[key] = value

    attrs = event.get("attrs")
    if isinstance(attrs, dict):
        for attr_key, attr_value in attrs.items():
            if not isinstance(attr_key, str):
                continue
            column_name = f"attr_{_sanitize_column_fragment(attr_key)}"
            if _is_scalar(attr_value):
                row[column_name] = attr_value
            else:
                row[column_name] = json.dumps(attr_value, sort_keys=True)

    for key, value in event.items():
        if key in core_keys or key == "attrs":
            continue
        if not isinstance(key, str):
            continue
        column_name = f"event_{_sanitize_column_fragment(key)}"
        if _is_scalar(value):
            row[column_name] = value
        else:
            row[column_name] = json.dumps(value, sort_keys=True)
    return row


def _iter_completion_dirs(input_root: Path) -> list[Path]:
    completion_manifest_dirs = {
        path.parent
        for path in input_root.glob("**/completion.json")
        if _looks_like_completion_dir(path.parent)
    }
    telemetry_dirs = {
        path.parent
        for path in input_root.glob("**/telemetry.jsonl")
        if _looks_like_completion_dir(path.parent)
    }
    return sorted(completion_manifest_dirs | telemetry_dirs)


def _timing_summary_scalar_fields(timing_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "completion_timing_summary_present": True,
        "completion_timing_step_count": _to_int(timing_summary.get("step_count")),
        "completion_timing_completion_end_to_end_seconds": _to_float(
            timing_summary.get("completion_end_to_end_seconds")
        ),
    }

    totals = timing_summary.get("totals")
    if isinstance(totals, dict):
        for key in (
            "step_end_to_end_seconds",
            "attribution_seconds",
            "token_generation_seconds",
            "artifact_save_seconds",
        ):
            row[f"completion_timing_totals_{key}"] = _to_float(totals.get(key))

    averages_per_step = timing_summary.get("averages_per_step")
    if isinstance(averages_per_step, dict):
        for key in (
            "step_end_to_end_seconds",
            "attribution_seconds",
            "token_generation_seconds",
            "artifact_save_seconds",
        ):
            row[f"completion_timing_averages_per_step_{key}"] = _to_float(
                averages_per_step.get(key)
            )

    phase_elapsed_aggregate = timing_summary.get(
        "attribution_phase_elapsed_seconds_total_aggregate",
        timing_summary.get("attribution_phase_elapsed_seconds_total"),
    )
    if isinstance(phase_elapsed_aggregate, dict):
        phase_elapsed_total_aggregate = 0.0
        for phase_name, elapsed_seconds in phase_elapsed_aggregate.items():
            if not isinstance(phase_name, str):
                continue
            elapsed_seconds_float = _to_float(elapsed_seconds)
            if elapsed_seconds_float is None:
                continue
            column_name = (
                "completion_timing_attribution_phase_elapsed_seconds_total_"
                f"{_sanitize_column_fragment(phase_name)}"
            )
            row[column_name] = elapsed_seconds_float
            phase_elapsed_total_aggregate += elapsed_seconds_float
        row["completion_timing_attribution_phase_elapsed_seconds_total_all_phases"] = (
            round(phase_elapsed_total_aggregate, 6)
        )

    phase_elapsed_wall_clock = timing_summary.get(
        "attribution_phase_wall_clock_elapsed_seconds_total"
    )
    if isinstance(phase_elapsed_wall_clock, dict):
        phase_elapsed_total_wall_clock = 0.0
        for phase_name, elapsed_seconds in phase_elapsed_wall_clock.items():
            if not isinstance(phase_name, str):
                continue
            elapsed_seconds_float = _to_float(elapsed_seconds)
            if elapsed_seconds_float is None:
                continue
            column_name = (
                "completion_timing_attribution_phase_wall_clock_elapsed_seconds_total_"
                f"{_sanitize_column_fragment(phase_name)}"
            )
            row[column_name] = elapsed_seconds_float
            phase_elapsed_total_wall_clock += elapsed_seconds_float
        row[
            "completion_timing_attribution_phase_wall_clock_elapsed_seconds_total_all_phases"
        ] = round(phase_elapsed_total_wall_clock, 6)

    return row


def gather_telemetry(
    *, input_root: Path, output_dir: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    run_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []

    completion_dirs = _iter_completion_dirs(input_root)

    for completion_dir in completion_dirs:
        completion_manifest_path = completion_dir / "completion.json"
        completion_manifest = (
            read_json(completion_manifest_path)
            if completion_manifest_path.exists()
            else {}
        )

        telemetry_ref = completion_manifest.get("telemetry_events_path")
        telemetry_path: Path | None = None
        declared_telemetry_path: Path | None = None
        if isinstance(telemetry_ref, str) and telemetry_ref.strip():
            declared_telemetry_path = completion_dir / telemetry_ref.strip()
            if declared_telemetry_path.exists():
                telemetry_path = declared_telemetry_path
        default_telemetry_path = completion_dir / "telemetry.jsonl"
        if telemetry_path is None and default_telemetry_path.exists():
            telemetry_path = default_telemetry_path

        telemetry_events = list(iter_jsonl(telemetry_path)) if telemetry_path else []
        scenario_root = _find_scenario_root(completion_dir, input_root)
        result_path = scenario_root / "result.json" if scenario_root else None
        scenario_path = scenario_root / "scenario.json" if scenario_root else None
        result = read_json(result_path) if result_path and result_path.exists() else {}
        scenario = (
            read_json(scenario_path) if scenario_path and scenario_path.exists() else {}
        )

        summary_run_metadata = _load_summary_run_metadata(scenario_root)
        scenario_run_metadata_raw = scenario.get("run_metadata")
        scenario_run_metadata = (
            scenario_run_metadata_raw
            if isinstance(scenario_run_metadata_raw, dict)
            else {}
        )
        result_run_metadata_raw = result.get("run_metadata")
        result_run_metadata = (
            result_run_metadata_raw if isinstance(result_run_metadata_raw, dict) else {}
        )

        run_id = _first_non_null(
            result.get("run_id"),
            result.get("launch_id"),
            result_run_metadata.get("run_id"),
            result_run_metadata.get("launch_id"),
            scenario_run_metadata.get("run_id"),
            scenario_run_metadata.get("launch_id"),
            summary_run_metadata.get("run_id"),
            summary_run_metadata.get("launch_id"),
        )
        run_name = _first_non_null(
            result.get("run_name"),
            result_run_metadata.get("run_name"),
            scenario_run_metadata.get("run_name"),
            summary_run_metadata.get("run_name"),
        )
        run_description = _first_non_null(
            result.get("run_description"),
            result_run_metadata.get("run_description"),
            scenario_run_metadata.get("run_description"),
            summary_run_metadata.get("run_description"),
        )
        run_goal = _first_non_null(
            result.get("run_goal"),
            result_run_metadata.get("run_goal"),
            scenario_run_metadata.get("run_goal"),
            summary_run_metadata.get("run_goal"),
        )

        prompt_id = (
            completion_manifest.get("prompt_id")
            if isinstance(completion_manifest.get("prompt_id"), str)
            else completion_dir.parent.name
        )
        completion_id = (
            completion_manifest.get("completion_id")
            if isinstance(completion_manifest.get("completion_id"), str)
            else completion_dir.name
        )
        scenario_name = (
            result.get("name")
            or scenario.get("name")
            or (scenario_root.name if scenario_root else completion_dir.parent.name)
        )

        base_context = {
            "scenario_root": str(scenario_root) if scenario_root else None,
            "scenario_name": scenario_name,
            "stage": result.get("stage") or scenario.get("stage"),
            "cluster": _infer_cluster(scenario_root, scenario),
            "method": result.get("method") or scenario.get("method"),
            "status": result.get("status"),
            "run_id": run_id,
            "launch_id": run_id,
            "run_name": run_name,
            "run_description": run_description,
            "run_goal": run_goal,
            "prompt_id": prompt_id,
            "completion_id": completion_id,
            "completion_dir": str(completion_dir),
            "completion_manifest_path": (
                str(completion_manifest_path)
                if completion_manifest_path.exists()
                else None
            ),
            "telemetry_file_path": str(telemetry_path) if telemetry_path else None,
            "telemetry_manifest_declared_path": (
                str(declared_telemetry_path) if declared_telemetry_path else None
            ),
            "telemetry_file_path_relative": (
                _relative_to_or_str(telemetry_path, input_root)
                if telemetry_path
                else None
            ),
            "telemetry_manifest_declared_path_relative": (
                _relative_to_or_str(declared_telemetry_path, input_root)
                if declared_telemetry_path
                else None
            ),
        }

        telemetry_scope_counts: dict[str, int] = defaultdict(int)
        telemetry_elapsed_ms_totals: dict[str, float] = defaultdict(float)
        total_elapsed_ms = 0.0
        step_event_aggs: dict[int, dict[str, float | int]] = defaultdict(
            lambda: {
                "event_count": 0,
                "batch_event_count": 0,
                "op_event_count": 0,
                "elapsed_ms_total": 0.0,
                "batch_elapsed_ms_total": 0.0,
                "op_elapsed_ms_total": 0.0,
            }
        )

        for event_idx, event in enumerate(telemetry_events):
            if not isinstance(event, dict):
                continue
            scope = _event_scope(event)
            if scope:
                telemetry_scope_counts[scope] += 1

            elapsed_ms = _to_float(event.get("elapsed_ms"))
            if elapsed_ms is not None:
                total_elapsed_ms += elapsed_ms
                if scope:
                    telemetry_elapsed_ms_totals[scope] += elapsed_ms

            step_index = _event_step_index(event)
            if step_index is not None:
                step_agg = step_event_aggs[step_index]
                step_agg["event_count"] += 1
                if elapsed_ms is not None:
                    step_agg["elapsed_ms_total"] += elapsed_ms
                if scope == "batch":
                    step_agg["batch_event_count"] += 1
                    if elapsed_ms is not None:
                        step_agg["batch_elapsed_ms_total"] += elapsed_ms
                elif scope == "op":
                    step_agg["op_event_count"] += 1
                    if elapsed_ms is not None:
                        step_agg["op_elapsed_ms_total"] += elapsed_ms

            event_row = {
                **base_context,
                "telemetry_event_file_index": event_idx,
                "telemetry_step_index": step_index,
                "telemetry_scope": scope,
                **_flatten_event(event),
            }
            if elapsed_ms is not None:
                event_row["elapsed_seconds"] = round(elapsed_ms / 1000.0, 6)

            if scope == "batch":
                batch_rows.append(event_row)
            elif scope == "op":
                op_rows.append(event_row)

        run_row = {
            **base_context,
            "result_file": str(result_path)
            if result_path and result_path.exists()
            else None,
            "scenario_file": (
                str(scenario_path) if scenario_path and scenario_path.exists() else None
            ),
            "telemetry_file_exists": telemetry_path is not None,
            "telemetry_event_count_file": len(telemetry_events),
            "telemetry_event_count_manifest": _to_int(
                completion_manifest.get("telemetry_event_count")
            ),
            "telemetry_scope_batch_count": telemetry_scope_counts.get("batch", 0),
            "telemetry_scope_op_count": telemetry_scope_counts.get("op", 0),
            "telemetry_scope_phase_count": telemetry_scope_counts.get("phase", 0),
            "telemetry_scope_step_count": telemetry_scope_counts.get("step", 0),
            "telemetry_scope_completion_count": telemetry_scope_counts.get(
                "completion", 0
            ),
            "telemetry_scope_run_count": telemetry_scope_counts.get("run", 0),
            "telemetry_elapsed_ms_total": round(total_elapsed_ms, 6),
            "telemetry_elapsed_seconds_total": round(total_elapsed_ms / 1000.0, 6),
            "telemetry_elapsed_ms_batch_total": round(
                telemetry_elapsed_ms_totals.get("batch", 0.0),
                6,
            ),
            "telemetry_elapsed_ms_op_total": round(
                telemetry_elapsed_ms_totals.get("op", 0.0),
                6,
            ),
            "manifest_n_steps_traced": _to_int(
                completion_manifest.get("n_steps_traced")
            ),
            "manifest_generated_token_count": _to_int(
                completion_manifest.get("generated_token_count")
            ),
            "manifest_phase4_feature_batch_size_effective": _to_int(
                completion_manifest.get("phase4_feature_batch_size_effective")
            ),
            "manifest_phase4_feature_batch_planner_status": completion_manifest.get(
                "phase4_feature_batch_planner_status"
            ),
            "manifest_phase4_feature_batch_planner_skip_reason": completion_manifest.get(
                "phase4_feature_batch_planner_skip_reason"
            ),
        }
        manifest_event_count = run_row.get("telemetry_event_count_manifest")
        run_row["telemetry_event_count_delta_file_minus_manifest"] = (
            run_row["telemetry_event_count_file"] - manifest_event_count
            if isinstance(manifest_event_count, int)
            else None
        )

        timing_summary = completion_manifest.get("timing_summary")
        if isinstance(timing_summary, dict):
            run_row.update(_timing_summary_scalar_fields(timing_summary))
        else:
            run_row["completion_timing_summary_present"] = False

        run_rows.append(run_row)

        manifest_steps = completion_manifest.get("steps")
        step_records = (
            [step for step in manifest_steps if isinstance(step, dict)]
            if isinstance(manifest_steps, list)
            else []
        )
        seen_step_indices: set[int] = set()

        for step_row_index, step_record in enumerate(step_records):
            step_index = _to_int(step_record.get("step_index"))
            if step_index is None:
                step_index = step_row_index
            seen_step_indices.add(step_index)

            step_row = {
                **base_context,
                "step_index": step_index,
                "step_row_source": "manifest_step",
                "prefix_token_count": _to_int(step_record.get("prefix_token_count")),
                "generated_token_count": _to_int(
                    step_record.get("generated_token_count")
                ),
                "next_token_id": _to_int(step_record.get("next_token_id")),
                "next_token_text": step_record.get("next_token_text"),
                "n_active_features": _to_int(step_record.get("n_active_features")),
                "n_edges_retained": _to_int(step_record.get("n_edges_retained")),
                "stop_reason": step_record.get("stop_reason"),
                "step_end_to_end_seconds": _to_float(
                    step_record.get("step_end_to_end_seconds")
                ),
                "attribution_seconds": _to_float(
                    step_record.get("attribution_seconds")
                ),
                "token_generation_seconds": _to_float(
                    step_record.get("token_generation_seconds")
                ),
                "artifact_save_seconds": _to_float(
                    step_record.get("artifact_save_seconds")
                ),
                "phase4_feature_batch_size": _to_int(
                    step_record.get("phase4_feature_batch_size")
                ),
                "phase4_feature_batch_planner_status": step_record.get(
                    "phase4_feature_batch_planner_status"
                ),
                "phase4_feature_batch_planner_skip_reason": step_record.get(
                    "phase4_feature_batch_planner_skip_reason"
                ),
                "telemetry_event_count_manifest_step": _to_int(
                    step_record.get("telemetry_event_count")
                ),
            }

            step_phase_elapsed_aggregate = step_record.get(
                "attribution_phase_elapsed_seconds_aggregate",
                step_record.get("attribution_phase_elapsed_seconds"),
            )
            phase_elapsed_total_aggregate = 0.0
            if isinstance(step_phase_elapsed_aggregate, dict):
                for phase_name, elapsed_seconds in step_phase_elapsed_aggregate.items():
                    if not isinstance(phase_name, str):
                        continue
                    elapsed_seconds_float = _to_float(elapsed_seconds)
                    if elapsed_seconds_float is None:
                        continue
                    column_name = (
                        "attribution_phase_elapsed_seconds_"
                        f"{_sanitize_column_fragment(phase_name)}"
                    )
                    step_row[column_name] = elapsed_seconds_float
                    phase_elapsed_total_aggregate += elapsed_seconds_float
            step_row["attribution_phase_elapsed_seconds_all_phases"] = round(
                phase_elapsed_total_aggregate,
                6,
            )

            step_phase_elapsed_wall_clock = step_record.get(
                "attribution_phase_wall_clock_elapsed_seconds"
            )
            phase_elapsed_total_wall_clock = 0.0
            if isinstance(step_phase_elapsed_wall_clock, dict):
                for (
                    phase_name,
                    elapsed_seconds,
                ) in step_phase_elapsed_wall_clock.items():
                    if not isinstance(phase_name, str):
                        continue
                    elapsed_seconds_float = _to_float(elapsed_seconds)
                    if elapsed_seconds_float is None:
                        continue
                    column_name = (
                        "attribution_phase_wall_clock_elapsed_seconds_"
                        f"{_sanitize_column_fragment(phase_name)}"
                    )
                    step_row[column_name] = elapsed_seconds_float
                    phase_elapsed_total_wall_clock += elapsed_seconds_float
            step_row["attribution_phase_wall_clock_elapsed_seconds_all_phases"] = round(
                phase_elapsed_total_wall_clock,
                6,
            )

            step_event_agg = step_event_aggs.get(step_index)
            if step_event_agg:
                elapsed_ms_total = float(step_event_agg["elapsed_ms_total"])
                batch_elapsed_ms_total = float(step_event_agg["batch_elapsed_ms_total"])
                op_elapsed_ms_total = float(step_event_agg["op_elapsed_ms_total"])
                step_row.update(
                    {
                        "telemetry_event_count_file_step": int(
                            step_event_agg["event_count"]
                        ),
                        "telemetry_batch_event_count_file_step": int(
                            step_event_agg["batch_event_count"]
                        ),
                        "telemetry_op_event_count_file_step": int(
                            step_event_agg["op_event_count"]
                        ),
                        "telemetry_elapsed_ms_total_step": round(elapsed_ms_total, 6),
                        "telemetry_elapsed_seconds_total_step": round(
                            elapsed_ms_total / 1000.0,
                            6,
                        ),
                        "telemetry_batch_elapsed_ms_total_step": round(
                            batch_elapsed_ms_total,
                            6,
                        ),
                        "telemetry_op_elapsed_ms_total_step": round(
                            op_elapsed_ms_total,
                            6,
                        ),
                    }
                )

            step_rows.append(step_row)

        for step_index in sorted(step_event_aggs):
            if step_index in seen_step_indices:
                continue
            step_event_agg = step_event_aggs[step_index]
            elapsed_ms_total = float(step_event_agg["elapsed_ms_total"])
            batch_elapsed_ms_total = float(step_event_agg["batch_elapsed_ms_total"])
            op_elapsed_ms_total = float(step_event_agg["op_elapsed_ms_total"])
            step_rows.append(
                {
                    **base_context,
                    "step_index": step_index,
                    "step_row_source": "telemetry_only",
                    "telemetry_event_count_file_step": int(
                        step_event_agg["event_count"]
                    ),
                    "telemetry_batch_event_count_file_step": int(
                        step_event_agg["batch_event_count"]
                    ),
                    "telemetry_op_event_count_file_step": int(
                        step_event_agg["op_event_count"]
                    ),
                    "telemetry_elapsed_ms_total_step": round(elapsed_ms_total, 6),
                    "telemetry_elapsed_seconds_total_step": round(
                        elapsed_ms_total / 1000.0,
                        6,
                    ),
                    "telemetry_batch_elapsed_ms_total_step": round(
                        batch_elapsed_ms_total,
                        6,
                    ),
                    "telemetry_op_elapsed_ms_total_step": round(op_elapsed_ms_total, 6),
                }
            )

    ensure_dir(output_dir)
    return run_rows, step_rows, batch_rows, op_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gather per-completion telemetry.jsonl and completion/result metadata into "
            "flat analysis-friendly CSV tables"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing benchmark scenario outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where telemetry summary CSVs are written",
    )
    args = parser.parse_args()

    run_rows, step_rows, batch_rows, op_rows = gather_telemetry(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )

    write_csv(
        args.output_dir / "telemetry_runs.csv",
        run_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "method",
            "status",
            "run_id",
            "run_name",
            "prompt_id",
            "completion_id",
            "completion_manifest_path",
            "telemetry_file_path",
            "telemetry_manifest_declared_path",
            "telemetry_file_exists",
            "telemetry_event_count_file",
            "telemetry_event_count_manifest",
            "telemetry_event_count_delta_file_minus_manifest",
            "telemetry_scope_batch_count",
            "telemetry_scope_op_count",
            "telemetry_elapsed_seconds_total",
            "manifest_phase4_feature_batch_size_effective",
            "manifest_phase4_feature_batch_planner_status",
            "manifest_phase4_feature_batch_planner_skip_reason",
            "completion_timing_summary_present",
            "completion_timing_completion_end_to_end_seconds",
            "completion_timing_totals_attribution_seconds",
            "completion_timing_attribution_phase_elapsed_seconds_total_all_phases",
            "completion_timing_attribution_phase_wall_clock_elapsed_seconds_total_all_phases",
        ],
    )
    write_csv(
        args.output_dir / "telemetry_steps.csv",
        step_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "method",
            "status",
            "run_id",
            "prompt_id",
            "completion_id",
            "step_index",
            "step_row_source",
            "step_end_to_end_seconds",
            "attribution_seconds",
            "token_generation_seconds",
            "artifact_save_seconds",
            "phase4_feature_batch_size",
            "phase4_feature_batch_planner_status",
            "phase4_feature_batch_planner_skip_reason",
            "telemetry_event_count_manifest_step",
            "telemetry_event_count_file_step",
            "telemetry_elapsed_seconds_total_step",
            "attribution_phase_elapsed_seconds_all_phases",
            "attribution_phase_wall_clock_elapsed_seconds_all_phases",
        ],
    )
    write_csv(
        args.output_dir / "telemetry_batches.csv",
        batch_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "method",
            "status",
            "run_id",
            "prompt_id",
            "completion_id",
            "telemetry_event_file_index",
            "telemetry_step_index",
            "scope",
            "name",
            "phase",
            "batch_index",
            "elapsed_ms",
            "elapsed_seconds",
        ],
    )
    write_csv(
        args.output_dir / "telemetry_ops.csv",
        op_rows,
        preferred_headers=[
            "scenario_root",
            "scenario_name",
            "stage",
            "cluster",
            "method",
            "status",
            "run_id",
            "prompt_id",
            "completion_id",
            "telemetry_event_file_index",
            "telemetry_step_index",
            "scope",
            "name",
            "phase",
            "elapsed_ms",
            "elapsed_seconds",
        ],
    )

    print(
        "Wrote telemetry tables: "
        f"runs={len(run_rows)}, "
        f"steps={len(step_rows)}, "
        f"batches={len(batch_rows)}, "
        f"ops={len(op_rows)} "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
