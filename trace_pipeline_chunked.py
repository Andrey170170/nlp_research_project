"""
Exact chunked-decoder variant of the multi-prompt tracing pipeline.

This keeps circuit-tracer's original attribution semantics much more closely
than the activation top-K monkey patch in trace_pipeline.py by avoiding full
decoder materialization during setup_attribution while preserving the full
active feature set until circuit-tracer's normal Phase-4 pruning.

Key idea:
- compute CLT reconstruction exactly, but in chunks
- defer feature->decoder expansion until AttributionContext needs a specific
  destination layer during backward scoring
- keep circuit-tracer's normal max_feature_nodes behavior in Phase 4

This should let us compare:
- trace_pipeline.py          : approximate early feature-cap patch
- trace_pipeline_chunked.py  : exact chunked decoder patch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType
from typing import Any

import torch

import trace_pipeline as base


def _select_active_encoder_vectors(transcoder, features: torch.Tensor) -> torch.Tensor:
    """Return encoder vectors in the exact order of features.indices()."""
    layer_idx, _, feat_idx = features.indices()
    encoder_vectors = []

    for layer_id in range(transcoder.n_layers):
        current_layer = layer_idx == layer_id
        if bool(current_layer.any()):
            encoder_vectors.append(
                transcoder._get_encoder_weights(layer_id)[feat_idx[current_layer]]
            )

    if encoder_vectors:
        return torch.cat(encoder_vectors, dim=0)

    return torch.empty(
        (0, transcoder.d_model), device=features.device, dtype=transcoder.dtype
    )


def _get_decoder_slice(
    transcoder, source_layer: int, feat_ids: torch.Tensor, output_offset: int
) -> torch.Tensor:
    """Return decoder vectors for one destination-layer offset only."""
    decoder_block = transcoder._get_decoder_vectors(source_layer, feat_ids.cpu())
    return decoder_block[:, output_offset, :].to(
        device=feat_ids.device, dtype=transcoder.dtype
    )


def _compute_reconstruction_chunked(
    transcoder,
    features: torch.Tensor,
    inputs: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Compute exact reconstruction without materializing all decoder rows."""
    features = features.coalesce()
    layer_idx, pos_idx, feat_idx = features.indices()
    activations = features.values().to(dtype=transcoder.dtype)
    n_layers, n_pos, _ = features.shape

    reconstruction = torch.zeros(
        (n_layers, n_pos, transcoder.d_model),
        device=inputs.device,
        dtype=transcoder.dtype,
    )

    for source_layer in range(n_layers):
        current_layer = layer_idx == source_layer
        if not bool(current_layer.any()):
            continue

        source_indices = current_layer.nonzero(as_tuple=False).squeeze(-1)

        for start in range(0, source_indices.numel(), chunk_size):
            idx_chunk = source_indices[start : start + chunk_size]
            pos_chunk = pos_idx[idx_chunk]
            feat_chunk = feat_idx[idx_chunk]
            act_chunk = activations[idx_chunk]

            unique_feats, inv = feat_chunk.unique(sorted=True, return_inverse=True)
            unique_decoders = transcoder._get_decoder_vectors(
                source_layer, unique_feats.cpu()
            )

            for output_offset in range(unique_decoders.shape[1]):
                dest_layer = source_layer + output_offset
                output_vecs = (
                    unique_decoders[:, output_offset, :][inv] * act_chunk[:, None]
                )
                reconstruction[dest_layer].index_add_(0, pos_chunk, output_vecs)

    reconstruction = reconstruction + transcoder.b_dec[:, None]
    if transcoder.W_skip is not None:
        reconstruction = reconstruction + inputs @ transcoder.W_skip

    return reconstruction


def install_chunked_decoder_patch(
    transcoders,
    *,
    reconstruction_chunk_size: int = 1024,
    score_chunk_size: int = 128,
) -> None:
    """Install an exact chunked decoder patch for attribution setup."""
    from circuit_tracer.attribution.context_nnsight import AttributionContext
    from circuit_tracer.replacement_model.replacement_model_nnsight import (
        NNSightReplacementModel,
    )

    def patched_compute_attribution_components(
        self, inputs, zero_positions=slice(0, 1)
    ):
        features, _ = self.encode_sparse(inputs, zero_positions=zero_positions)
        features = features.coalesce()
        encoder_vectors = _select_active_encoder_vectors(self, features)
        reconstruction = _compute_reconstruction_chunked(
            self,
            features,
            inputs,
            chunk_size=reconstruction_chunk_size,
        )

        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values().to(dtype=self.dtype)

        self._chunked_attr_state = {
            "transcoder": self,
            "source_layers": layer_idx.detach(),
            "source_positions": pos_idx.detach(),
            "source_feature_ids": feat_idx.detach(),
            "source_activations": activations.detach(),
            "score_chunk_size": score_chunk_size,
        }

        empty_decoder_vecs = torch.empty(
            (0, self.d_model), device=inputs.device, dtype=self.dtype
        )
        empty_idx = torch.empty((0,), device=inputs.device, dtype=torch.long)
        empty_locations = torch.empty((2, 0), device=inputs.device, dtype=torch.long)

        return {
            "activation_matrix": features,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vectors,
            "decoder_vecs": empty_decoder_vecs,
            "encoder_to_decoder_map": empty_idx,
            "decoder_locations": empty_locations,
        }

    if not getattr(NNSightReplacementModel, "_chunked_decoder_setup_patch", False):
        original_setup_attribution = NNSightReplacementModel.setup_attribution

        @torch.no_grad()
        def patched_setup_attribution(self, inputs):
            ctx = original_setup_attribution(self, inputs)
            state = getattr(self.transcoders, "_chunked_attr_state", None)
            if state is not None:
                ctx._chunked_decoder_state = state
                ctx._chunked_transcoders = state["transcoder"]
                self.transcoders._chunked_attr_state = None
            return ctx

        setattr(NNSightReplacementModel, "setup_attribution", patched_setup_attribution)
        setattr(NNSightReplacementModel, "_chunked_decoder_setup_patch", True)

    if not getattr(AttributionContext, "_chunked_decoder_feature_patch", False):
        original_compute_feature_attributions = (
            AttributionContext.compute_feature_attributions
        )

        def patched_compute_feature_attributions(self, layer, grads):
            state = getattr(self, "_chunked_decoder_state", None)
            transcoder = getattr(self, "_chunked_transcoders", None)
            if state is None or transcoder is None:
                return original_compute_feature_attributions(self, layer, grads)

            source_layers = state["source_layers"]
            source_positions = state["source_positions"]
            source_feature_ids = state["source_feature_ids"]
            source_activations = state["source_activations"]
            score_chunk_size_local = state["score_chunk_size"]

            for source_layer in range(layer + 1):
                if source_layer >= transcoder.n_layers:
                    continue
                current_layer = source_layers == source_layer
                if not bool(current_layer.any()):
                    continue

                source_indices = current_layer.nonzero(as_tuple=False).squeeze(-1)
                output_offset = layer - source_layer
                if output_offset >= transcoder.n_layers - source_layer:
                    continue

                for start in range(0, source_indices.numel(), score_chunk_size_local):
                    idx_chunk = source_indices[start : start + score_chunk_size_local]
                    pos_chunk = source_positions[idx_chunk]
                    feat_chunk = source_feature_ids[idx_chunk]
                    act_chunk = source_activations[idx_chunk]

                    unique_feats, inv = feat_chunk.unique(
                        sorted=True, return_inverse=True
                    )
                    output_vecs = (
                        _get_decoder_slice(
                            transcoder,
                            source_layer,
                            unique_feats,
                            output_offset,
                        )[inv]
                        * act_chunk[:, None]
                    )

                    scores = torch.einsum(
                        "bkd,kd->kb",
                        grads[:, pos_chunk].to(output_vecs.dtype),
                        output_vecs,
                    )
                    self._batch_buffer[idx_chunk] += scores

        setattr(
            AttributionContext,
            "compute_feature_attributions",
            patched_compute_feature_attributions,
        )
        setattr(AttributionContext, "_chunked_decoder_feature_patch", True)

    transcoders.compute_attribution_components = MethodType(
        patched_compute_attribution_components, transcoders
    )
    print(
        "  Installed exact chunked decoder patch "
        f"(reconstruction_chunk_size={reconstruction_chunk_size}, "
        f"score_chunk_size={score_chunk_size})"
    )


def run_pipeline(args: argparse.Namespace) -> None:
    model = base.load_model()
    install_chunked_decoder_patch(
        model.transcoders,  # type: ignore[union-attr]
        reconstruction_chunk_size=args.reconstruction_chunk_size,
        score_chunk_size=args.score_chunk_size,
    )
    examples = base.load_gsm8k_examples(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload = None if args.no_offload else "cpu"

    run_config = {
        "prompts": args.prompts,
        "completions_per_prompt": args.completions,
        "temperature": args.temperature,
        "max_feature_nodes": args.max_feature_nodes,
        "max_edges": args.max_edges,
        "max_steps": args.max_steps,
        "offload": offload,
        "save_raw": args.save_raw,
        "output_dir": str(output_dir),
        "patch_type": "chunked_decoder_exact",
        "reconstruction_chunk_size": args.reconstruction_chunk_size,
        "score_chunk_size": args.score_chunk_size,
    }
    (output_dir / "run_config.json").write_text(base.json.dumps(run_config, indent=2))

    total = len(examples) * args.completions
    done = 0

    for prompt_idx, example in enumerate(examples):
        prompt = base.format_prompt(model.tokenizer, example["question"])  # type: ignore[unresolved-attribute]

        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = {
            "gsm8k_index": prompt_idx,
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
                offload=offload,
                save_raw=args.save_raw,
            )

    print(f"\nPipeline complete! {done} completions traced to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunked-decoder multi-prompt tracing pipeline"
    )
    parser.add_argument(
        "--prompts", type=int, default=10, help="Number of GSM8K prompts"
    )
    parser.add_argument(
        "--completions", type=int, default=3, help="Completions per prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--output-dir",
        default="/fs/scratch/PAS3272/kopanev.1/traces_1",
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
        "--reconstruction-chunk-size",
        type=int,
        default=1024,
        help="Active-feature chunk size for exact reconstruction",
    )
    parser.add_argument(
        "--score-chunk-size",
        type=int,
        default=128,
        help="Feature chunk size for decoder scoring during backward attribution",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)
