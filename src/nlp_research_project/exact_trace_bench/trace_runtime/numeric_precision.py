"""Explicit float32 matmul precision policy for the exact-trace critical path.

``exact_trace_internal_dtype`` selects fp32 or fp64 accumulation for attribution
rows, but it does not control how PyTorch executes a float32 matmul. TF32
silently truncates float32 matmul inputs to a 10-bit mantissa, and the PyTorch
default for that behaviour has changed across releases. Inheriting the default
makes an exactness-critical property depend on the installed torch version,
where a bounded parity gate would not detect the change.

This module pins the policy at process entry and reports the resolved backend
state so it can be recorded as run provenance rather than assumed.
"""

from __future__ import annotations

from typing import Any

import torch


FLOAT32_MATMUL_PRECISION = "highest"


def _plain(value: object) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


def resolved_float32_precision() -> dict[str, Any]:
    """Read the backend precision state that affects float32 matmuls."""

    state: dict[str, Any] = {
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "torch_version": str(torch.__version__),
    }
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None:
        for name in ("allow_tf32", "fp32_precision"):
            if hasattr(matmul, name):
                state[f"cuda_matmul_{name}"] = _plain(getattr(matmul, name))
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        state["cudnn_allow_tf32"] = _plain(cudnn.allow_tf32)
    return state


def pin_float32_matmul_precision() -> dict[str, Any]:
    """Force IEEE float32 matmuls and return the resolved backend state.

    Idempotent. ``torch.set_float32_matmul_precision`` is the supported control
    for matmul TF32; ``cudnn.allow_tf32`` is set separately because it is a
    distinct switch. The current models use no convolutions, so the cuDNN switch
    is defensive rather than load-bearing.
    """

    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = False
    return resolved_float32_precision()
