"""Small runtime support functions independent of legacy tracing entrypoints."""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


def generate_next_token(
    model: Any, input_ids: torch.Tensor, *, temperature: float
) -> dict[str, Any]:
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_dict_in_generate=True,
            output_scores=True,
        )
    token_id = int(outputs.sequences[0, -1].item())
    token_text = model.tokenizer.decode([token_id], skip_special_tokens=False)
    logprob = None
    if outputs.scores:
        scores = outputs.scores[0][0].float()
        logprob = float(torch.log_softmax(scores, dim=-1)[token_id].item())
    return {
        "next_input_ids": outputs.sequences,
        "token_id": token_id,
        "token_text": token_text,
        "token_logprob": logprob,
    }


def capture_resource_snapshot() -> dict[str, float | None]:
    rss_gib = None
    if resource is not None:
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    snapshot = {
        "rss_gib": rss_gib,
        "cuda_allocated_gib": None,
        "cuda_reserved_gib": None,
        "cuda_peak_allocated_gib": None,
        "cuda_peak_reserved_gib": None,
    }
    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                "cuda_reserved_gib": torch.cuda.memory_reserved() / (1024**3),
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()
                / (1024**3),
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved()
                / (1024**3),
            }
        )
    return snapshot


def capture_transcoder_diagnostics(model: Any) -> dict[str, Any] | None:
    getter = getattr(
        getattr(model, "transcoders", None), "get_diagnostic_snapshot", None
    )
    if not callable(getter):
        return None
    snapshot = getter()
    if not isinstance(snapshot, dict):
        return None
    keys = (
        "encoder_load_count",
        "encoder_load_seconds",
        "decoder_load_count",
        "decoder_load_seconds",
        "decoder_cache_hit_count",
        "decoder_cache_miss_count",
        "decoder_cache_eviction_count",
        "decoder_cache_skip_count",
        "decoder_cache_auto_disable_count",
        "decoder_cache_bytes_resident",
        "decoder_cache_max_bytes",
        "encode_sparse_seconds",
        "reconstruction_chunk_count",
        "reconstruction_seconds",
    )
    return {key: snapshot[key] for key in keys if key in snapshot}


def build_completion_timing_summary(
    *, completion_end_to_end_seconds: float, step_records: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = (
        "step_end_to_end_seconds",
        "attribution_seconds",
        "token_generation_seconds",
        "artifact_save_seconds",
    )
    totals = {field: 0.0 for field in fields}
    for record in step_records:
        for field in fields:
            value = record.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    totals[field] += numeric
    count = len(step_records)
    return {
        "completion_end_to_end_seconds": round(completion_end_to_end_seconds, 6),
        "totals": {key: round(value, 6) for key, value in totals.items()},
        "averages_per_step": {
            key: round(value / count if count else 0.0, 6)
            for key, value in totals.items()
        },
        "step_count": count,
    }
