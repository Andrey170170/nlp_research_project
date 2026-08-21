"""Resolve immutable mechanism expectations from a prepared launch record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from .prepared_workload import PreparedWorkloadError, validate_prepared_workload


class PreparedExpectationError(ValueError):
    """Raised when one prepared workload has no unambiguous mechanism identity."""


def resolve_prepared_mechanism_expectations(
    prepared_workload_path: Path,
) -> dict[str, str | int | bool]:
    """Return the single fingerprinted mechanism selected by a prepared bundle."""

    try:
        workload = validate_prepared_workload(prepared_workload_path)
    except PreparedWorkloadError as error:
        raise PreparedExpectationError(str(error)) from error
    launch_spec = workload.launch_spec

    selections = launch_spec.get("mechanism_selections")
    if not isinstance(selections, list) or len(selections) != 1:
        raise PreparedExpectationError(
            "prepared diagnostic requires exactly one mechanism selection"
        )
    selection = selections[0]
    if not isinstance(selection, Mapping):
        raise PreparedExpectationError("mechanism selection must be an object")

    required_string_fields = (
        "feature_row_influence_mode",
        "backward_engine_mode",
        "forward_graph_mode",
        "vjp_kernel_mode",
    )
    result: dict[str, str | int | bool] = {}
    for field in required_string_fields:
        value = selection.get(field)
        if not isinstance(value, str) or not value:
            raise PreparedExpectationError(
                f"mechanism selection lacks non-empty {field}"
            )
        result[field] = value
    lane_count = selection.get("forward_lane_count")
    if (
        not isinstance(lane_count, int)
        or isinstance(lane_count, bool)
        or lane_count <= 0
    ):
        raise PreparedExpectationError(
            "mechanism selection lacks a positive forward_lane_count"
        )
    result["forward_lane_count"] = lane_count

    configs = launch_spec.get("selected_configs")
    if not isinstance(configs, list) or len(configs) != 1:
        raise PreparedExpectationError(
            "prepared workload requires exactly one selected configuration"
        )
    config_entry = configs[0]
    if not isinstance(config_entry, Mapping):
        raise PreparedExpectationError("selected configuration must be an object")
    if config_entry.get("config_fingerprint") != selection.get("config_fingerprint"):
        raise PreparedExpectationError(
            "mechanism and selected configuration fingerprints disagree"
        )
    config = config_entry.get("selected_config")
    if not isinstance(config, Mapping):
        raise PreparedExpectationError("selected_config must be an object")

    def positive_int(field: str) -> int:
        value = config.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PreparedExpectationError(f"selected_config lacks a positive {field}")
        return value

    def nonnegative_int(field: str, default: int) -> int:
        value = config.get(field, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PreparedExpectationError(
                f"selected_config lacks a non-negative {field}"
            )
        return value

    session_capacity = positive_int("nnsight_session_capacity")
    result.update(
        {
            "session_capacity": session_capacity,
            # Backward engines are constructed from the resolved session capacity.
            "backward_batch_capacity": session_capacity,
            "phase1_trace_batch_size_max": positive_int("phase1_trace_batch_size_max"),
            # Single-forward modes keep logical capacity but execute Phase 1 with
            # the physical forward graph width. Duplicated lanes use that width too.
            "actual_phase1_forward_trace_width": lane_count,
            "phase3_batch_size": positive_int("phase3_compute_microbatch_max_rows"),
            "phase4_batch_size": positive_int("phase4_execution_batch_max_rows"),
        }
    )
    active_rows = config.get("decoder_active_row_residency", False)
    if not isinstance(active_rows, bool):
        raise PreparedExpectationError(
            "selected_config.decoder_active_row_residency must be a bool"
        )
    active_row_requirement = config.get(
        "decoder_active_row_residency_requirement", "preferred"
    )
    if active_row_requirement not in {"preferred", "required"}:
        raise PreparedExpectationError(
            "selected_config.decoder_active_row_residency_requirement must be "
            "preferred or required"
        )
    result.update(
        {
            "decoder_active_row_residency": active_rows,
            "decoder_active_row_residency_requirement": active_row_requirement,
            "decoder_active_row_max_bytes": nonnegative_int(
                "decoder_active_row_max_bytes", 0
            ),
            "decoder_active_row_safety_margin_bytes": nonnegative_int(
                "decoder_active_row_safety_margin_bytes", 0
            ),
        }
    )
    stop_mode = config.get("diagnostic_stop_mode")
    if not isinstance(stop_mode, str) or not stop_mode:
        raise PreparedExpectationError(
            "selected_config lacks a non-empty diagnostic_stop_mode"
        )
    execution_mode = workload.record.get("execution_mode")
    expected_execution_mode = {
        "none": "full",
        "phase0_probe": "phase0-probe",
        "phase3_probe": "phase3-probe",
        "transition_probe": "transition-probe",
    }.get(stop_mode)
    if expected_execution_mode is None:
        raise PreparedExpectationError(
            f"selected_config has unsupported diagnostic_stop_mode {stop_mode!r}"
        )
    if execution_mode != expected_execution_mode:
        raise PreparedExpectationError(
            "prepared execution_mode disagrees with selected diagnostic stop mode"
        )
    result["require_full_completion"] = stop_mode == "none"
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_workload", type=Path)
    parser.add_argument(
        "--format",
        choices=("json", "lines"),
        default="json",
        help="lines emits a stable sixteen-line record for the Slurm wrapper",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = resolve_prepared_mechanism_expectations(args.prepared_workload)
    except (OSError, json.JSONDecodeError, PreparedExpectationError) as error:
        raise SystemExit(str(error)) from error
    if args.format == "lines":
        for field in (
            "feature_row_influence_mode",
            "backward_engine_mode",
            "forward_graph_mode",
            "vjp_kernel_mode",
            "forward_lane_count",
            "session_capacity",
            "backward_batch_capacity",
            "phase1_trace_batch_size_max",
            "actual_phase1_forward_trace_width",
            "phase3_batch_size",
            "phase4_batch_size",
            "require_full_completion",
            "decoder_active_row_residency",
            "decoder_active_row_residency_requirement",
            "decoder_active_row_max_bytes",
            "decoder_active_row_safety_margin_bytes",
        ):
            value = result[field]
            print(int(value) if isinstance(value, bool) else value)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
