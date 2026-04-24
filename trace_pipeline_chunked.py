"""Fork-native exact chunked-decoder tracing pipeline.

This variant relies on the chunked `circuit-tracer` fork directly instead of
installing runtime monkey patches. For GemmaScope-2 CLTs, the fork handles the
exact chunked decoder path internally and preserves the full active feature set
until normal Phase-4 feature selection.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch

import trace_pipeline as base
from circuit_utils import StepData, save_compact


def extract_compact_chunked_attribution(
    model,
    prompt: str | torch.Tensor | list[int],
    *,
    max_feature_nodes: int = 32768,
    batch_size: int = 256,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_n_logits: int = 5,
    desired_logit_prob: float = 0.9,
    offload: str | None = "cpu",
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: Any | None = None,
    prefix_cache: Any | None = None,
    decoder_chunk_cache: Any | None = None,
) -> dict[str, Any]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import importlib

    attribute_module = importlib.import_module(
        "circuit_tracer.attribution.attribute_nnsight"
    )
    attribute_nnsight = getattr(attribute_module, "attribute")
    return attribute_nnsight(
        prompt=prompt,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        feature_batch_size=feature_batch_size,
        logit_batch_size=logit_batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
        update_interval=update_interval,
        profile=profile,
        profile_log_interval=profile_log_interval,
        diagnostic_feature_cap=diagnostic_feature_cap,
        sparsification=sparsification,
        compact_output=True,
        prefix_cache=prefix_cache,
        decoder_chunk_cache=decoder_chunk_cache,
    )


def compact_result_to_step_data(
    compact_result: dict[str, Any],
    step_idx: int,
    *,
    token_text: str = "",
    logprob: float | None = None,
    max_edges: int = 10_000,
) -> StepData:
    feature_feature_edges = compact_result["feature_feature_edges"]
    logit_feature_edges = compact_result["logit_feature_edges"]
    feature_row_node_indices = compact_result["feature_row_node_indices"].to(
        dtype=torch.int64
    )
    logit_row_node_indices = compact_result["logit_row_node_indices"].to(
        dtype=torch.int64
    )
    selected_features = compact_result["selected_features"].to(dtype=torch.int64)
    feature_ids = compact_result["active_features"].numpy().astype(base.np.int64)
    n_features = feature_ids.shape[0]

    ff_flat = feature_feature_edges.abs().float().reshape(-1)
    lf_flat = logit_feature_edges.abs().float().reshape(-1)
    combined = torch.cat([ff_flat, lf_flat])
    k = min(max_edges, int((combined != 0).sum().item()))
    if k == 0:
        return StepData(
            step_idx=step_idx,
            row_idx=base.np.empty(0, dtype=base.np.int32),
            col_idx=base.np.empty(0, dtype=base.np.int32),
            weights=base.np.empty(0, dtype=base.np.float32),
            feature_ids=feature_ids,
            token_text=token_text,
            logprob=logprob,
            n_features=n_features,
        )

    topk_vals, topk_idx = torch.topk(combined, k, sorted=False)
    topk64 = topk_vals.double()
    kept_mass = float(topk64.sum().item())
    if kept_mass == 0:
        return StepData(
            step_idx=step_idx,
            row_idx=base.np.empty(0, dtype=base.np.int32),
            col_idx=base.np.empty(0, dtype=base.np.int32),
            weights=base.np.empty(0, dtype=base.np.float32),
            feature_ids=feature_ids,
            token_text=token_text,
            logprob=logprob,
            n_features=n_features,
        )

    ff_size = ff_flat.numel()
    n_selected = int(selected_features.numel())
    rows: list[int] = []
    cols: list[int] = []

    in_ff = topk_idx < ff_size
    ff_idx = topk_idx[in_ff]
    for flat_idx in ff_idx.tolist():
        local_row = flat_idx // n_selected
        local_col = flat_idx % n_selected
        rows.append(int(feature_row_node_indices[local_row].item()))
        cols.append(int(selected_features[local_col].item()))

    lf_idx = (topk_idx[~in_ff] - ff_size).tolist()
    for flat_idx in lf_idx:
        local_row = flat_idx // n_selected
        local_col = flat_idx % n_selected
        rows.append(int(logit_row_node_indices[local_row].item()))
        cols.append(int(selected_features[local_col].item()))

    return StepData(
        step_idx=step_idx,
        row_idx=base.np.array(rows, dtype=base.np.int32),
        col_idx=base.np.array(cols, dtype=base.np.int32),
        weights=(topk64 / kept_mass).float().numpy(),
        feature_ids=feature_ids,
        token_text=token_text,
        logprob=logprob,
        n_features=n_features,
    )


def trace_completion_compact_chunked(
    model,
    prompt: str,
    *,
    output_dir: Path,
    prompt_idx: int,
    completion_idx: int,
    temperature: float = 0.7,
    max_steps: int = 256,
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
) -> dict[str, Any]:
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

    candidate_stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn != tokenizer.unk_token_id:
        candidate_stop_ids.append(end_of_turn)
    stop_token_ids = {
        token_id for token_id in candidate_stop_ids if token_id is not None
    }

    print(f"\n  [{prompt_id}/{completion_id}] Starting trace (temp={temperature})")

    for step_idx in range(max_steps):
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
        )

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

        step_record = {
            "step_index": step_idx,
            "prefix_token_count": int(input_ids.shape[1]),
            "generated_token_count": len(generated_token_ids),
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "next_token_logprob": token_result["token_logprob"],
            "n_active_features": step_data.n_features,
            "n_edges_retained": len(step_data.weights),
            "stop_reason": "eos" if next_token_id in stop_token_ids else None,
            "resource_snapshot": base.capture_resource_snapshot(),
            "transcoder_diagnostics": base.capture_transcoder_diagnostics(model),
        }
        step_records.append(step_record)

        if step_idx % 10 == 0 or next_token_id in stop_token_ids:
            print(
                f"    Step {step_idx:03d}: tok={next_token_text!r} "
                f"feat={step_data.n_features} edges={len(step_data.weights)}"
            )

        del compact_result, step_data
        gc.collect()
        input_ids = token_result["next_input_ids"]

        if next_token_id in stop_token_ids:
            print(f"    Stop token at step {step_idx}")
            break

    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    manifest = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "fixture_name": fixture_name,
        "fixture_kind": fixture_kind,
        "completion_text": completion_text,
        "n_steps_traced": len(step_records),
        "duration_seconds": round(time.time() - trace_start, 2),
        "prompt_token_count": resolved_prompt_token_count,
        "initial_input_token_count": initial_input_token_count,
        "generated_token_count": len(generated_token_ids),
        "temperature": temperature,
        "max_feature_nodes": max_feature_nodes,
        "max_edges": max_edges,
        "attribution_batch_size": attribution_batch_size,
        "feature_batch_size": feature_batch_size,
        "logit_batch_size": logit_batch_size,
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "offload": offload,
        "verbose_attribution": verbose_attribution,
        "attribution_update_interval": attribution_update_interval,
        "profile_attribution": profile_attribution,
        "profile_log_interval": profile_log_interval,
        "diagnostic_feature_cap": diagnostic_feature_cap,
        "sparsification": (
            {
                "per_layer_position_topk": sparsification.per_layer_position_topk,
                "global_cap": sparsification.global_cap,
            }
            if sparsification is not None
            else None
        ),
        "save_raw": False,
        "graph_packaging_mode": "compact_chunked_no_full_graph",
        "resource_snapshot": base.capture_resource_snapshot(),
        "steps": step_records,
    }
    manifest_path = completion_dir / "completion.json"
    manifest_path.write_text(base.json.dumps(manifest, indent=2))
    print(f"    Saved manifest: {manifest_path}")
    print(f"    Answer (first 200 chars): {completion_text[:200]}")
    return manifest


def build_sparsification_config(args: argparse.Namespace):
    if (
        args.sparsify_per_layer_position_topk is None
        and args.sparsify_global_cap is None
    ):
        return None

    import importlib

    sparsification_module = importlib.import_module(
        "circuit_tracer.attribution.sparsification"
    )
    SparsificationConfig = getattr(sparsification_module, "SparsificationConfig")

    return SparsificationConfig(
        per_layer_position_topk=args.sparsify_per_layer_position_topk,
        global_cap=args.sparsify_global_cap,
    )


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
    gsm8k_indices = [
        example["gsm8k_index"]
        for example in examples
        if example.get("gsm8k_index") is not None
    ]
    if not gsm8k_indices:
        gsm8k_indices = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload = None if args.no_offload else "cpu"

    run_config = {
        "prompts": args.prompts,
        "gsm8k_indices": gsm8k_indices,
        "completions_per_prompt": args.completions,
        "temperature": args.temperature,
        "max_feature_nodes": args.max_feature_nodes,
        "max_edges": args.max_edges,
        "max_steps": args.max_steps,
        "attribution_batch_size": args.attribution_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "logit_batch_size": args.logit_batch_size,
        "max_n_logits": args.max_n_logits,
        "desired_logit_prob": args.desired_logit_prob,
        "offload": offload,
        "verbose_attribution": args.verbose_attribution,
        "attribution_update_interval": args.attribution_update_interval,
        "profile_attribution": args.profile_attribution,
        "profile_log_interval": args.profile_log_interval,
        "diagnostic_feature_cap": args.diagnostic_feature_cap,
        "save_raw": args.save_raw,
        "output_dir": str(output_dir),
        "patch_type": "fork_native_exact_chunked_decoder",
        "uses_monkeypatch": False,
        "lazy_encoder": not args.no_lazy_encoder,
        "lazy_decoder": not args.no_lazy_decoder,
        "decoder_chunk_size": args.decoder_chunk_size,
        "cross_batch_decoder_cache_bytes": args.cross_batch_decoder_cache_bytes,
        "prepared_prompt_file": args.prepared_prompt_file,
        "prepared_prompt_meta_file": args.prepared_prompt_meta_file,
        "graph_packaging_mode": (
            "full_graph" if args.save_raw else "compact_chunked_no_full_graph"
        ),
        "sparsification": (
            {
                "per_layer_position_topk": sparsification.per_layer_position_topk,
                "global_cap": sparsification.global_cap,
            }
            if sparsification is not None
            else None
        ),
    }
    (output_dir / "run_config.json").write_text(base.json.dumps(run_config, indent=2))

    total = len(examples) * args.completions
    done = 0

    for prompt_idx, example in enumerate(examples):
        prompt = base.resolve_prompt_text(model.tokenizer, example)  # type: ignore[unresolved-attribute]
        initial_input_token_count = int(
            model.ensure_tokenized(prompt).shape[0]  # type: ignore[unresolved-attribute]
        )

        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = base.build_prompt_meta_record(
            example,
            prompt_text=prompt,
            initial_input_token_count=initial_input_token_count,
        )
        (prompt_dir / "prompt_meta.json").write_text(
            base.json.dumps(prompt_meta, indent=2)
        )

        for comp_idx in range(args.completions):
            done += 1
            print(f"\n{'=' * 60}")
            print(
                f"Completion {done}/{total}: prompt {prompt_idx}, completion {comp_idx}"
            )
            print(f"{'=' * 60}")

            trace_fn = (
                base.trace_completion
                if args.save_raw
                else trace_completion_compact_chunked
            )
            trace_fn(
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
                **({"save_raw": True} if args.save_raw else {}),
            )

    print(f"\nPipeline complete! {done} completions traced to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fork-native exact chunked-decoder tracing pipeline"
    )
    parser.add_argument(
        "--prompts", type=int, default=10, help="Number of GSM8K prompts"
    )
    parser.add_argument(
        "--gsm8k-indices",
        default=None,
        help="Comma-separated GSM8K test indices to trace explicitly",
    )
    parser.add_argument(
        "--gsm8k-indices-file",
        default=None,
        help="Path to JSON/newline file containing explicit GSM8K test indices",
    )
    parser.add_argument(
        "--prepared-prompt-file",
        default=None,
        help="Path to a prepared prompt/prefix text file to trace instead of formatting GSM8K input",
    )
    parser.add_argument(
        "--prepared-prompt-meta-file",
        default=None,
        help="Optional JSON metadata file describing the prepared prompt fixture",
    )
    parser.add_argument(
        "--completions", type=int, default=3, help="Completions per prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--output-dir",
        default="/fs/scratch/PAS3272/kopanev.1/traces_chunked",
        help="Output directory",
    )
    parser.add_argument(
        "--save-raw", action="store_true", help="Also save raw .pt files (~460 MB each)"
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep attribution on GPU (faster but may OOM)",
    )
    parser.add_argument(
        "--max-feature-nodes",
        type=int,
        default=32768,
        help="Max feature nodes for attribution",
    )
    parser.add_argument(
        "--max-edges", type=int, default=10_000, help="Edges to retain per step"
    )
    parser.add_argument(
        "--max-steps", type=int, default=256, help="Max generation steps per completion"
    )
    parser.add_argument(
        "--attribution-batch-size",
        type=int,
        default=256,
        help="Backward batch size for attribution graph extraction",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=None,
        help="Optional Phase-4 feature microbatch override (<= attribution batch size)",
    )
    parser.add_argument(
        "--logit-batch-size",
        type=int,
        default=None,
        help="Optional Phase-3 logit microbatch override (<= attribution batch size)",
    )
    parser.add_argument(
        "--max-n-logits",
        type=int,
        default=5,
        help="Maximum number of logit targets to attribute",
    )
    parser.add_argument(
        "--desired-logit-prob",
        type=float,
        default=0.9,
        help="Cumulative probability threshold for auto-selected logit targets",
    )
    parser.add_argument(
        "--verbose-attribution",
        action="store_true",
        help="Enable fork attribution phase logging and tqdm progress",
    )
    parser.add_argument(
        "--attribution-update-interval",
        type=int,
        default=4,
        help="Feature ranking refresh interval used inside attribution",
    )
    parser.add_argument(
        "--profile-attribution",
        action="store_true",
        help="Enable batch-level attribution profiling logs from the fork",
    )
    parser.add_argument(
        "--profile-log-interval",
        type=int,
        default=1,
        help="Emit attribution profiling logs every N batches",
    )
    parser.add_argument(
        "--diagnostic-feature-cap",
        type=int,
        default=None,
        help="Debug-only early active-feature cap for profiling/scaling experiments",
    )
    parser.add_argument(
        "--sparsify-per-layer-position-topk",
        type=int,
        default=None,
        help="Retain top-K features per (layer, position) bucket before exact attribution",
    )
    parser.add_argument(
        "--sparsify-global-cap",
        type=int,
        default=None,
        help="Optional global cap applied after per-bucket sparsification",
    )
    parser.add_argument(
        "--decoder-chunk-size",
        type=int,
        default=256,
        help="Fork-native decoder chunk size for exact CLT attribution",
    )
    parser.add_argument(
        "--cross-batch-decoder-cache-bytes",
        type=int,
        default=None,
        help="Optional Phase-4 cross-batch decoder cache budget in bytes",
    )
    parser.add_argument(
        "--no-lazy-encoder",
        action="store_true",
        help="Eagerly load encoder weights instead of using lazy encoder reads",
    )
    parser.add_argument(
        "--no-lazy-decoder",
        action="store_true",
        help="Eagerly load decoder weights instead of using lazy decoder reads",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)
