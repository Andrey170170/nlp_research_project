"""Canonical project-side selection for backward execution topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from circuit_tracer import BackwardPlan


BACKWARD_ENGINE_PRESETS: dict[str, tuple[str, str]] = {
    "duplicated_lanes": ("logical_capacity", "nnsight_injected"),
    "single_forward_batched_vjp": ("single_lane", "autograd_batched"),
    "single_forward_serial_vjp": ("single_lane", "autograd_serial"),
}
FORWARD_GRAPH_MODES = frozenset(mode for mode, _ in BACKWARD_ENGINE_PRESETS.values())
VJP_KERNEL_MODES = frozenset(kernel for _, kernel in BACKWARD_ENGINE_PRESETS.values())
_AXES_TO_PRESET = {axes: preset for preset, axes in BACKWARD_ENGINE_PRESETS.items()}


@dataclass(frozen=True)
class BackwardExecutionSelection:
    """Resolved backward identity plus the representation selected by the user."""

    backward_engine_mode: str
    forward_graph_mode: str
    vjp_kernel_mode: str
    selection_source: str

    def forward_lane_count(self, knobs: Mapping[str, Any]) -> int | None:
        if self.forward_graph_mode == "single_lane":
            return 1
        for key in ("nnsight_session_capacity", "attribution_batch_size"):
            value = knobs.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None


def resolve_backward_execution_selection(
    knobs: Mapping[str, Any],
) -> BackwardExecutionSelection:
    """Resolve a preset or a complete explicit axis pair, failing closed.

    A missing selection preserves the historical ``duplicated_lanes`` default.
    Explicit axes are deliberately mutually exclusive with a preset so provenance
    never has two potentially conflicting sources of truth.
    """

    preset_present = (
        "backward_engine_mode" in knobs
        and knobs.get("backward_engine_mode") is not None
    )
    graph_present = (
        "forward_graph_mode" in knobs and knobs.get("forward_graph_mode") is not None
    )
    kernel_present = (
        "vjp_kernel_mode" in knobs and knobs.get("vjp_kernel_mode") is not None
    )

    if graph_present != kernel_present:
        raise ValueError(
            "forward_graph_mode and vjp_kernel_mode must be selected together"
        )
    if preset_present and graph_present:
        raise ValueError(
            "backward_engine_mode cannot be combined with forward_graph_mode and "
            "vjp_kernel_mode"
        )

    if graph_present:
        graph_mode = str(knobs["forward_graph_mode"])
        kernel_mode = str(knobs["vjp_kernel_mode"])
        if graph_mode not in FORWARD_GRAPH_MODES:
            raise ValueError(
                f"forward_graph_mode must be one of {sorted(FORWARD_GRAPH_MODES)!r}"
            )
        if kernel_mode not in VJP_KERNEL_MODES:
            raise ValueError(
                f"vjp_kernel_mode must be one of {sorted(VJP_KERNEL_MODES)!r}"
            )
        try:
            preset = _AXES_TO_PRESET[(graph_mode, kernel_mode)]
        except KeyError as error:
            raise ValueError(
                "unsupported backward execution combination: "
                f"forward_graph_mode={graph_mode!r}, "
                f"vjp_kernel_mode={kernel_mode!r}"
            ) from error
        return BackwardExecutionSelection(
            backward_engine_mode=preset,
            forward_graph_mode=graph_mode,
            vjp_kernel_mode=kernel_mode,
            selection_source="explicit_pair",
        )

    preset = str(knobs.get("backward_engine_mode", "duplicated_lanes"))
    try:
        graph_mode, kernel_mode = BACKWARD_ENGINE_PRESETS[preset]
    except KeyError as error:
        raise ValueError(
            f"backward_engine_mode must be one of {sorted(BACKWARD_ENGINE_PRESETS)!r}"
        ) from error
    return BackwardExecutionSelection(
        backward_engine_mode=preset,
        forward_graph_mode=graph_mode,
        vjp_kernel_mode=kernel_mode,
        selection_source="preset" if preset_present else "default",
    )


def normalize_backward_overrides(
    knobs: dict[str, Any], overrides: Mapping[str, Any]
) -> None:
    """Prevent the inherited legacy default from fabricating a pair conflict."""

    explicit_pair = (
        overrides.get("forward_graph_mode") is not None
        and overrides.get("vjp_kernel_mode") is not None
    )
    explicit_preset = overrides.get("backward_engine_mode") is not None
    if explicit_pair and not explicit_preset:
        knobs.pop("backward_engine_mode", None)


def backward_mechanism_record(knobs: Mapping[str, Any]) -> dict[str, Any]:
    selection = resolve_backward_execution_selection(knobs)
    return {
        "backward_engine_mode": selection.backward_engine_mode,
        "forward_graph_mode": selection.forward_graph_mode,
        "vjp_kernel_mode": selection.vjp_kernel_mode,
        "forward_lane_count": selection.forward_lane_count(knobs),
        "backward_selection_source": selection.selection_source,
    }


def build_backward_plan(knobs: Mapping[str, Any]) -> BackwardPlan:
    """Build the sibling typed plan from exactly one selection representation."""

    from circuit_tracer import (
        BackwardEngineMode,
        BackwardPlan,
        ForwardGraphMode,
        VjpKernelMode,
    )

    selection = resolve_backward_execution_selection(knobs)
    if selection.selection_source == "explicit_pair":
        return BackwardPlan(
            forward_graph_mode=cast(ForwardGraphMode, selection.forward_graph_mode),
            vjp_kernel_mode=cast(VjpKernelMode, selection.vjp_kernel_mode),
        )
    return BackwardPlan(mode=cast(BackwardEngineMode, selection.backward_engine_mode))
