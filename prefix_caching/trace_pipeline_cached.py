"""Tracing pipeline with prefix-cache validation.

This is a standalone entrypoint that runs the same exact tracing as
``trace_pipeline_chunked.py`` but wraps each step with a PrefixCache
to measure how much work is redundant across consecutive steps.

It does NOT modify the fork or skip any computation.  It runs attribute()
normally, then compares the results against what was cached from the
previous step.  The output is a ``cache_validation.json`` alongside the
normal trace artifacts.

Usage::

    uv run python -m prefix_caching.trace_pipeline_cached \\
        --gsm8k-indices 94 --completions 1 --temperature 0 \\
        --max-steps 20 --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

import trace_pipeline as base
from circuit_utils import save_compact
from prefix_caching.cache import PrefixCache
from trace_pipeline_chunked import (
    build_sparsification_config,
    compact_result_to_step_data,
    extract_compact_chunked_attribution,
)

try:
    from circuit_tracer.attribution.prefix_cache import (
        PrefixActivationCache as LibraryPrefixCache,
    )
except ImportError:  # library version predates the forward cache
    LibraryPrefixCache = None  # type: ignore[assignment,misc]


def trace_completion_with_cache_validation(
    model,
    prompt: str,
    *,
    output_dir: Path,
    prompt_idx: int,
    completion_idx: int,
    temperature: float = 0.0,
    max_steps: int = 20,
    max_feature_nodes: int = 32768,
    max_edges: int = 10_000,
    attribution_batch_size: int = 256,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_n_logits: int = 5,
    desired_logit_prob: float = 0.9,
    offload: str | None = "cpu",
    verbose_attribution: bool = False,
    attribution_update_interval: int = 4,
    profile_attribution: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: Any | None = None,
    prompt_token_count: int | None = None,
    prompt_source: str = "gsm8k",
    fixture_name: str | None = None,
    fixture_kind: str | None = None,
    use_library_prefix_cache: bool = False,
    use_decoder_chunk_session: bool = False,
) -> dict[str, Any]:
    """Trace a completion with prefix-cache validation at each step.

    Runs the same tracing as ``trace_completion_compact_chunked`` but after
    each attribution call, compares the active features against what was
    cached from the previous step.  Writes per-step comparison stats to
    ``cache_validation.json``.

    Parameters
    ----------
    use_library_prefix_cache:
        When True, instantiates a ``circuit_tracer.attribution.PrefixActivationCache``
        and passes it into every ``attribute()`` call so the library can
        populate and consult it across consecutive steps.  The in-process
        ``PrefixCache`` validator is still recorded (features side).
    """
    tokenizer = model.tokenizer
    prompt_id = f"prompt_{prompt_idx:03d}"
    completion_id = f"completion_{completion_idx:03d}"
    completion_dir = output_dir / prompt_id / completion_id
    completion_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trace_start = time.time()
    input_ids = model.ensure_tokenized(prompt).unsqueeze(0)
    initial_input_token_count = int(input_ids.shape[1])
    resolved_prompt_token_count = (
        initial_input_token_count
        if prompt_token_count is None
        else int(prompt_token_count)
    )
    generated_token_ids: list[int] = []
    step_records: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []

    # Stop tokens.
    candidate_stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn != tokenizer.unk_token_id:
        candidate_stop_ids.append(end_of_turn)
    stop_token_ids = {
        token_id for token_id in candidate_stop_ids if token_id is not None
    }

    cache = PrefixCache()

    library_cache = None
    if use_library_prefix_cache:
        if LibraryPrefixCache is None:
            raise RuntimeError(
                "use_library_prefix_cache=True but the installed circuit_tracer "
                "build has no PrefixActivationCache.  Pull the jay/prefix-caching "
                "branch of circuit-tracer_chunked (or newer)."
            )
        library_cache = LibraryPrefixCache()
        print(
            f"  [{prompt_id}/{completion_id}] Library-side forward cache is ENABLED"
        )

    # Strategy D: session-scoped decoder chunk cache.  When enabled, we
    # build the cache once per completion (via the transcoder's own
    # constructor) and pass the same instance into every attribute() call.
    # The library branch we pull treats it as externally owned, so it is
    # not cleared at the end of setup_attribution and its 12 GiB of
    # resident chunks survive across steps.
    decoder_chunk_session = None
    if use_decoder_chunk_session:
        transcoders = getattr(model, "transcoders", None)
        create_cache = getattr(transcoders, "create_decoder_block_cache", None)
        if not callable(create_cache):
            raise RuntimeError(
                "use_decoder_chunk_session=True but the transcoder does not "
                "expose create_decoder_block_cache().  Ensure the library "
                "has exact-chunked-decoder support and the configured "
                "cross_batch_decoder_cache_bytes > 0."
            )
        decoder_chunk_session = create_cache()
        if decoder_chunk_session is None:
            raise RuntimeError(
                "create_decoder_block_cache() returned None; confirm "
                "lazy_decoder=True and cross_batch_decoder_cache_bytes > 0."
            )
        print(
            f"  [{prompt_id}/{completion_id}] Session decoder-chunk cache "
            f"is ENABLED (max_bytes={decoder_chunk_session.max_bytes})"
        )

    print(f"\n  [{prompt_id}/{completion_id}] Starting cached trace (temp={temperature})")

    for step_idx in range(max_steps):
        # ── Time the attribution call ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attribution_start = time.time()

        compact_result = extract_compact_chunked_attribution(
            model,
            input_ids[0],
            max_feature_nodes=max_feature_nodes,
            batch_size=attribution_batch_size,
            feature_batch_size=feature_batch_size,
            logit_batch_size=logit_batch_size,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            offload=offload,
            verbose=verbose_attribution,
            update_interval=attribution_update_interval,
            profile=profile_attribution,
            profile_log_interval=profile_log_interval,
            diagnostic_feature_cap=diagnostic_feature_cap,
            sparsification=sparsification,
            prefix_cache=library_cache,
            decoder_chunk_cache=decoder_chunk_session,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        attribution_seconds = time.time() - attribution_start

        # ── Cache comparison ──
        active_features = compact_result["active_features"]

        cache_record: dict[str, Any] = {
            "step_index": step_idx,
            "prefix_token_count": int(input_ids.shape[1]),
            "total_active_features": int(active_features.shape[0]),
            "attribution_seconds": round(attribution_seconds, 4),
        }

        if library_cache is not None:
            cache_record["library_cache"] = {
                "enabled": True,
                "hit_count": int(library_cache.hit_count),
                "miss_count": int(library_cache.miss_count),
                "cached_prefix_len": int(library_cache.cached_prefix_len),
            }
        else:
            cache_record["library_cache"] = {"enabled": False}

        if decoder_chunk_session is not None:
            cache_record["decoder_chunk_session"] = {
                "enabled": True,
                "max_bytes": int(decoder_chunk_session.max_bytes),
                "bytes_resident": int(decoder_chunk_session.bytes_resident),
                "entries_resident": len(decoder_chunk_session._entries),
            }
        else:
            cache_record["decoder_chunk_session"] = {"enabled": False}

        if cache.has_data:
            compare_result = cache.compare(active_features)
            cache_record["comparison"] = compare_result.to_dict()
            print(
                f"    Step {step_idx:03d}: "
                f"cached_pos={compare_result.cached_positions} "
                f"matched_pos={compare_result.matched_positions} "
                f"pos_rate={compare_result.position_match_rate:.4f} "
                f"feat_rate={compare_result.feature_match_rate:.4f} "
                f"attrib={attribution_seconds:.1f}s"
            )
        else:
            cache_record["comparison"] = None
            print(
                f"    Step {step_idx:03d}: first step (no cache yet) "
                f"features={active_features.shape[0]} "
                f"attrib={attribution_seconds:.1f}s"
            )

        # Print library-side cache counters to stdout every step so they
        # survive SLURM timeout even if completion.json never flushes.
        if library_cache is not None:
            print(
                f"        [library_cache] hit={library_cache.hit_count} "
                f"miss={library_cache.miss_count} "
                f"cached_prefix_len={library_cache.cached_prefix_len}",
                flush=True,
            )
        if decoder_chunk_session is not None:
            print(
                f"        [decoder_chunk_session] "
                f"bytes_resident={decoder_chunk_session.bytes_resident} "
                f"entries={len(decoder_chunk_session._entries)} "
                f"max_bytes={decoder_chunk_session.max_bytes}",
                flush=True,
            )

        # ── Store current step's features for next comparison ──
        cache.store(input_ids[0], active_features)

        cache_records.append(cache_record)

        # ── Normal step processing (same as trace_pipeline_chunked) ──
        token_result = base.generate_next_token(
            model, input_ids, temperature=temperature
        )
        next_token_id = token_result["token_id"]
        next_token_text = token_result["token_text"]
        generated_token_ids.append(next_token_id)

        step_data = compact_result_to_step_data(
            compact_result,
            step_idx,
            token_text=next_token_text,
            logprob=token_result["token_logprob"],
            max_edges=max_edges,
        )
        save_compact(step_data, completion_dir / f"step_{step_idx:03d}.npz")

        step_records.append({
            "step_index": step_idx,
            "prefix_token_count": int(input_ids.shape[1]),
            "generated_token_count": len(generated_token_ids),
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "next_token_logprob": token_result["token_logprob"],
            "n_active_features": step_data.n_features,
            "n_edges_retained": len(step_data.weights),
            "stop_reason": "eos" if next_token_id in stop_token_ids else None,
            "attribution_seconds": round(attribution_seconds, 4),
            "resource_snapshot": base.capture_resource_snapshot(),
        })

        del compact_result, step_data
        gc.collect()
        input_ids = token_result["next_input_ids"]

        if next_token_id in stop_token_ids:
            print(f"    Stop token at step {step_idx}")
            break

    # ── Save the normal completion manifest ──
    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    manifest = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "prompt_source": prompt_source,
        "completion_text": completion_text,
        "n_steps_traced": len(step_records),
        "duration_seconds": round(time.time() - trace_start, 2),
        "prompt_token_count": resolved_prompt_token_count,
        "initial_input_token_count": initial_input_token_count,
        "generated_token_count": len(generated_token_ids),
        "temperature": temperature,
        "max_feature_nodes": max_feature_nodes,
        "attribution_batch_size": attribution_batch_size,
        "resource_snapshot": base.capture_resource_snapshot(),
        "steps": step_records,
    }
    manifest_path = completion_dir / "completion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # ── Save cache validation results ──
    cache_validation = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "n_steps": len(cache_records),
        "temperature": temperature,
        "steps": cache_records,
    }
    cache_path = completion_dir / "cache_validation.json"
    cache_path.write_text(json.dumps(cache_validation, indent=2))

    print(f"    Saved manifest: {manifest_path}")
    print(f"    Saved cache validation: {cache_path}")
    print(f"    Answer (first 200 chars): {completion_text[:200]}")
    return manifest


def run_pipeline(args: argparse.Namespace) -> None:
    base.validate_attribution_batch_sizes(
        args.attribution_batch_size,
        args.feature_batch_size,
        args.logit_batch_size,
    )
    sparsification = build_sparsification_config(args)
    model = base.load_model(
        lazy_encoder=not args.no_lazy_encoder,
        lazy_decoder=not args.no_lazy_decoder,
        decoder_chunk_size=args.decoder_chunk_size,
        cross_batch_decoder_cache_bytes=args.cross_batch_decoder_cache_bytes,
    )
    examples = base.load_prompt_examples(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload = None if args.no_offload else "cpu"

    # Save run config.
    run_config = {
        "experiment": "prefix_cache_validation",
        "prompts": args.prompts,
        "completions_per_prompt": args.completions,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_feature_nodes": args.max_feature_nodes,
        "max_edges": args.max_edges,
        "attribution_batch_size": args.attribution_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "logit_batch_size": args.logit_batch_size,
        "decoder_chunk_size": args.decoder_chunk_size,
        "cross_batch_decoder_cache_bytes": args.cross_batch_decoder_cache_bytes,
        "output_dir": str(output_dir),
        "use_library_prefix_cache": bool(args.use_library_prefix_cache),
        "use_decoder_chunk_session": bool(args.use_decoder_chunk_session),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    total = len(examples) * args.completions
    done = 0

    for prompt_idx, example in enumerate(examples):
        prompt = base.resolve_prompt_text(model.tokenizer, example)
        initial_input_token_count = int(model.ensure_tokenized(prompt).shape[0])

        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = base.build_prompt_meta_record(
            example,
            prompt_text=prompt,
            initial_input_token_count=initial_input_token_count,
        )
        (prompt_dir / "prompt_meta.json").write_text(json.dumps(prompt_meta, indent=2))

        for comp_idx in range(args.completions):
            done += 1
            print(f"\n{'=' * 60}")
            print(
                f"Completion {done}/{total}: prompt {prompt_idx}, completion {comp_idx}"
            )
            print(f"{'=' * 60}")

            trace_completion_with_cache_validation(
                model,
                prompt,
                output_dir=output_dir,
                prompt_idx=prompt_idx,
                completion_idx=comp_idx,
                temperature=args.temperature,
                max_steps=args.max_steps,
                max_feature_nodes=args.max_feature_nodes,
                max_edges=args.max_edges,
                attribution_batch_size=args.attribution_batch_size,
                feature_batch_size=args.feature_batch_size,
                logit_batch_size=args.logit_batch_size,
                max_n_logits=args.max_n_logits,
                desired_logit_prob=args.desired_logit_prob,
                offload=offload,
                verbose_attribution=args.verbose_attribution,
                attribution_update_interval=args.attribution_update_interval,
                profile_attribution=args.profile_attribution,
                profile_log_interval=args.profile_log_interval,
                diagnostic_feature_cap=args.diagnostic_feature_cap,
                sparsification=sparsification,
                prompt_token_count=prompt_meta["prompt_token_count"],
                prompt_source=prompt_meta["prompt_source"],
                fixture_name=prompt_meta.get("fixture_name"),
                fixture_kind=prompt_meta.get("fixture_kind"),
                use_library_prefix_cache=args.use_library_prefix_cache,
                use_decoder_chunk_session=args.use_decoder_chunk_session,
            )

    print(f"\nPipeline complete! {done} completions traced to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracing pipeline with prefix-cache validation"
    )
    parser.add_argument(
        "--prompts", type=int, default=1, help="Number of GSM8K prompts"
    )
    parser.add_argument(
        "--gsm8k-indices", default=None,
        help="Comma-separated GSM8K test indices to trace",
    )
    parser.add_argument(
        "--gsm8k-indices-file", default=None,
        help="Path to JSON/newline file with GSM8K test indices",
    )
    parser.add_argument(
        "--prepared-prompt-file", default=None,
        help="Path to a prepared prompt text file",
    )
    parser.add_argument(
        "--prepared-prompt-meta-file", default=None,
        help="Optional JSON metadata for the prepared prompt",
    )
    parser.add_argument(
        "--completions", type=int, default=1, help="Completions per prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default 0 for deterministic)",
    )
    parser.add_argument(
        "--output-dir",
        default="/fs/scratch/PAS3272/kopanev.1/prefix_cache_bench",
        help="Output directory",
    )
    parser.add_argument(
        "--no-offload", action="store_true",
        help="Keep attribution on GPU (faster but may OOM)",
    )
    parser.add_argument(
        "--max-feature-nodes", type=int, default=32768,
        help="Max feature nodes for attribution",
    )
    parser.add_argument(
        "--max-edges", type=int, default=10_000,
        help="Edges to retain per step",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Max generation steps per completion",
    )
    parser.add_argument(
        "--attribution-batch-size", type=int, default=256,
        help="Backward batch size for attribution",
    )
    parser.add_argument(
        "--feature-batch-size", type=int, default=None,
        help="Optional Phase-4 feature microbatch override",
    )
    parser.add_argument(
        "--logit-batch-size", type=int, default=None,
        help="Optional Phase-3 logit microbatch override",
    )
    parser.add_argument(
        "--max-n-logits", type=int, default=5,
        help="Max logit targets to attribute",
    )
    parser.add_argument(
        "--desired-logit-prob", type=float, default=0.9,
        help="Cumulative probability threshold for logit targets",
    )
    parser.add_argument(
        "--verbose-attribution", action="store_true",
        help="Enable fork attribution logging",
    )
    parser.add_argument(
        "--attribution-update-interval", type=int, default=4,
        help="Feature ranking refresh interval",
    )
    parser.add_argument(
        "--profile-attribution", action="store_true",
        help="Enable attribution profiling logs",
    )
    parser.add_argument(
        "--profile-log-interval", type=int, default=1,
        help="Profiling log interval (batches)",
    )
    parser.add_argument(
        "--diagnostic-feature-cap", type=int, default=None,
        help="Debug-only early active-feature cap",
    )
    parser.add_argument(
        "--sparsify-per-layer-position-topk", type=int, default=None,
        help="Retain top-K features per (layer, position) bucket",
    )
    parser.add_argument(
        "--sparsify-global-cap", type=int, default=None,
        help="Optional global cap after per-bucket sparsification",
    )
    parser.add_argument(
        "--decoder-chunk-size", type=int, default=256,
        help="Fork-native decoder chunk size",
    )
    parser.add_argument(
        "--cross-batch-decoder-cache-bytes", type=int, default=None,
        help="Optional Phase-4 cross-batch decoder cache budget (bytes)",
    )
    parser.add_argument(
        "--no-lazy-encoder", action="store_true",
        help="Eagerly load encoder weights",
    )
    parser.add_argument(
        "--no-lazy-decoder", action="store_true",
        help="Eagerly load decoder weights",
    )
    parser.add_argument(
        "--use-library-prefix-cache", action="store_true",
        help=(
            "Pass a circuit_tracer.attribution.PrefixActivationCache into "
            "every attribute() call.  The library populates it after each "
            "step and emits phase0.setup.prefix_cache_{lookup,store} trace "
            "events.  This infrastructure run does not yet skip forward-pass "
            "compute; it records diagnostic evidence of reuse potential."
        ),
    )
    parser.add_argument(
        "--use-decoder-chunk-session", action="store_true",
        help=(
            "Strategy D: build one DecoderChunkCache per completion and "
            "pass it into every attribute() call so the 12 GiB chunk "
            "budget persists across consecutive tracing steps instead of "
            "being rebuilt.  Decoder chunks depend only on frozen model "
            "weights, so this is safe by construction and targets the "
            "dominant Phase-4 cost."
        ),
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)
