"""The exact-trace path must pin float32 matmul precision, not inherit it.

TF32 silently truncates float32 matmul inputs to a 10-bit mantissa. The PyTorch
default has changed across releases, and a bounded parity gate would not detect
the resulting drift, so the policy is pinned at process entry and recorded as
run provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_research_project.exact_trace_bench.trace_runtime.numeric_precision import (  # noqa: E402
    FLOAT32_MATMUL_PRECISION,
    pin_float32_matmul_precision,
    resolved_float32_precision,
)


def test_pin_forces_highest_float32_matmul_precision() -> None:
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        state = pin_float32_matmul_precision()
        assert torch.get_float32_matmul_precision() == FLOAT32_MATMUL_PRECISION
        assert state["float32_matmul_precision"] == FLOAT32_MATMUL_PRECISION
    finally:
        torch.set_float32_matmul_precision(original)


def test_pin_disables_tf32_on_every_exposed_switch() -> None:
    original = torch.get_float32_matmul_precision()
    try:
        state = pin_float32_matmul_precision()
        for key, value in state.items():
            if key.endswith("allow_tf32"):
                assert value is False, f"{key} must be disabled on the exact path"
    finally:
        torch.set_float32_matmul_precision(original)


def test_pin_is_idempotent() -> None:
    original = torch.get_float32_matmul_precision()
    try:
        first = pin_float32_matmul_precision()
        second = pin_float32_matmul_precision()
        assert first == second
    finally:
        torch.set_float32_matmul_precision(original)


def test_resolved_state_is_json_safe_and_records_torch_version() -> None:
    import json

    state = resolved_float32_precision()
    assert state["torch_version"] == str(torch.__version__)
    json.dumps(state)


def test_campaign_records_numeric_precision_in_run_config() -> None:
    """`run_config.json` is the provenance record; the key must be wired in."""

    import inspect

    from nlp_research_project.exact_trace_bench.trace_runtime import campaign

    source = inspect.getsource(campaign)
    assert "pin_float32_matmul_precision()" in source
    assert '"numeric_precision"' in source

    write_run_config = inspect.signature(campaign._write_run_config)
    assert "numeric_precision" in write_run_config.parameters
