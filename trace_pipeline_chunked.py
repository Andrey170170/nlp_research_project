"""Fork-native exact chunked-decoder tracing pipeline.

This variant relies on the chunked `circuit-tracer` fork directly instead of
installing runtime monkey patches. For GemmaScope-2 CLTs, the fork handles the
exact chunked decoder path internally and preserves the full active feature set
until normal Phase-4 feature selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import trace_pipeline as base


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
    gsm8k_indices = base.parse_gsm8k_indices(
        args.gsm8k_indices, args.gsm8k_indices_file
    )
    examples = base.load_gsm8k_examples(args.prompts, indices=gsm8k_indices)
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
        prompt = base.format_prompt(model.tokenizer, example["question"])  # type: ignore[unresolved-attribute]

        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = {
            "gsm8k_index": example["gsm8k_index"],
            "question": example["question"],
            "ground_truth_answer": example["answer"],
            "prompt_text": prompt,
        }
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

            base.trace_completion(
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
                save_raw=args.save_raw,
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
