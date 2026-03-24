"""
Multi-prompt tracing pipeline for temporal circuit stability.

Runs N GSM8K prompts × M completions each, extracting per-step
attribution graphs and saving compact .npz representations
(optionally alongside raw .pt files).

Designed for OSC H100 SLURM jobs.

Usage:
    python trace_pipeline.py [OPTIONS]

    --prompts N               Number of GSM8K prompts (default: 10)
    --completions N           Completions per prompt (default: 3)
    --temperature T           Sampling temperature (default: 0.7)
    --output-dir DIR          Output directory (default: /fs/scratch/PAS3272/kopanev.1/traces)
    --save-raw                Also save raw .pt files (~460 MB each)
    --no-offload              Keep attribution on GPU (faster, may OOM)
    --max-feature-nodes N     Max feature nodes for attribution (default: 32768)
    --max-edges N             Edges to retain per step (default: 10000)
    --max-steps N             Max generation steps per completion (default: 256)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Literal

import numpy as np
import torch
from circuit_tracer import ReplacementModel, attribute
from datasets import load_dataset

from circuit_utils import (
    StepData,
    save_compact,
    sparsify_edges,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

HF_REPO = "google/gemma-scope-2-1b-it"
CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"


# ── feature cap patch ────────────────────────────────────────────────
# The 262K-width cross-layer transcoders produce ~100K+ active features
# per forward pass.  select_decoder_vectors() gathers decoder vectors
# for ALL of them, which needs ~86 GB on GPU — more than a single H100.
# This patch prunes the sparse feature tensor to the top-K by activation
# magnitude BEFORE decoder selection, keeping memory manageable.
# circuit-tracer's own max_feature_nodes only prunes later (Phase 4),
# after the OOM point.


def install_feature_cap_patch(transcoders, max_features: int) -> None:
    """Monkey-patch compute_attribution_components to cap features before
    decoder selection.  This is the only way to control peak GPU memory
    with large cross-layer transcoders on a single GPU.
    """
    from types import MethodType

    def patched(self, inputs, zero_positions=slice(0, 1)):
        features, encoder_vectors = self.encode_sparse(
            inputs, zero_positions=zero_positions
        )
        # Prune before the expensive decoder selection
        features = features.coalesce()
        nnz = features._nnz()
        if nnz > max_features:
            keep = torch.topk(
                features.values().abs(), k=max_features, sorted=False
            ).indices
            features = torch.sparse_coo_tensor(
                features.indices()[:, keep],
                features.values()[keep],
                size=features.shape,
                device=features.device,
                dtype=features.dtype,
            ).coalesce()
            encoder_vectors = encoder_vectors[keep]
            print(f"    Feature cap: {max_features}/{nnz} kept")

        pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_to_decoder_map = (
            self.select_decoder_vectors(features)
        )
        reconstruction = self.compute_reconstruction(
            pos_ids, layer_ids, decoder_vectors, inputs
        )
        return {
            "activation_matrix": features,
            "reconstruction": reconstruction,
            "encoder_vecs": encoder_vectors,
            "decoder_vecs": decoder_vectors,
            "encoder_to_decoder_map": encoder_to_decoder_map,
            "decoder_locations": torch.stack((layer_ids, pos_ids)),
        }

    transcoders.compute_attribution_components = MethodType(patched, transcoders)
    print(f"  Installed feature cap patch (max_features={max_features})")


# ── model loading ────────────────────────────────────────────────────


def _layer_file_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def load_gemma_scope_2_clt_compat(
    paths: dict[int, str],
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load GemmaScope-2 CLTs while working around the installed loader bug."""
    from safetensors.torch import load_file
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

    if device is None:
        device = torch.device(DEVICE)

    ordered_paths = [paths[idx] for idx in sorted(paths)]
    # Load and stack on CPU to avoid OOM on smaller GPUs (40 GB A100).
    # The per-layer files total ~30 GB; stacking doubles peak memory briefly.
    params_list = [load_file(path, device="cpu") for path in ordered_paths]
    state_dict_raw = {
        key: torch.stack([params[key] for params in params_list])
        for key in params_list[0].keys()
    }
    del params_list  # free the individual layer dicts

    W_enc = (
        state_dict_raw["w_enc"]
        .transpose(-1, -2)
        .contiguous()
        .to(device=device, dtype=dtype)
    )
    b_enc = state_dict_raw["b_enc"].to(device=device, dtype=dtype)
    b_dec = state_dict_raw["b_dec"].to(device=device, dtype=dtype)
    threshold = state_dict_raw["threshold"].unsqueeze(1).to(device=device, dtype=dtype)

    n_layers, d_transcoder, d_model = W_enc.shape
    w_dec_raw = state_dict_raw["w_dec"]
    if w_dec_raw.shape[:3] != (n_layers, d_transcoder, n_layers):
        raise ValueError(
            "Unexpected decoder shape for GemmaScope-2 CLT: "
            f"{tuple(w_dec_raw.shape)} with encoder-derived layer count {n_layers}"
        )

    state_dict = {
        "W_enc": W_enc,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "activation_function.threshold": threshold,
    }
    for i in range(n_layers):
        state_dict[f"W_dec.{i}"] = w_dec_raw[i, :, i:, :].to(device=device, dtype=dtype)

    if "affine_skip_connection" in state_dict_raw:
        state_dict["W_skip"] = state_dict_raw["affine_skip_connection"].to(
            device=device, dtype=dtype
        )

    with torch.device("meta"):
        instance = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function="jump_relu",
            skip_connection=("W_skip" in state_dict),
            lazy_decoder=False,
            lazy_encoder=False,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            dtype=dtype,
        )

    instance.load_state_dict(state_dict, assign=True)
    return instance


def load_model() -> ReplacementModel:
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    if torch.cuda.is_available():
        print(f"  GPU memory before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{CLT_SUBFOLDER}/params_layer_*.safetensors"],
    )
    clt_dir = Path(local_dir) / CLT_SUBFOLDER
    layer_files = sorted(
        clt_dir.glob("params_layer_*.safetensors"), key=_layer_file_index
    )
    paths = {i: str(path) for i, path in enumerate(layer_files)}
    print(f"  Found {len(paths)} transcoder layer files")

    transcoders = load_gemma_scope_2_clt_compat(
        paths=paths,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device(DEVICE),
        dtype=DTYPE,
    )

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name="google/gemma-3-1b-it",
        transcoders=transcoders,
        dtype=DTYPE,
        backend="nnsight",
    )

    if torch.cuda.is_available():
        print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


# ── data loading ─────────────────────────────────────────────────────


def load_gsm8k_examples(n: int = 10) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = [ds[i] for i in range(min(n, len(ds)))]
    print(f"Loaded {len(examples)} GSM8K examples")
    return examples


def format_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Please solve this step by step and end with 'Final answer: <number>'."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── generation ───────────────────────────────────────────────────────


def generate_next_token(
    model, input_ids: torch.Tensor, *, temperature: float = 0.0
) -> dict[str, Any]:
    tokenizer = model.tokenizer

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_dict_in_generate=True,
            output_scores=True,
        )

    next_token_id = int(outputs.sequences[0, -1].item())
    next_token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
    logprob = None
    if outputs.scores:
        token_scores = outputs.scores[0][0].float()
        logprob = float(torch.log_softmax(token_scores, dim=-1)[next_token_id].item())

    return {
        "next_input_ids": outputs.sequences,
        "token_id": next_token_id,
        "token_text": next_token_text,
        "token_logprob": logprob,
    }


def extract_graph(
    model,
    prompt: str | torch.Tensor | list[int],
    *,
    max_feature_nodes: int = 32768,
    batch_size: int = 256,
    offload: Literal["cpu", "disk"] | None = "cpu",
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return attribute(
        prompt=prompt,
        model=model,
        max_n_logits=5,
        desired_logit_prob=0.9,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=False,
    )


# ── compact save from live graph ─────────────────────────────────────


def graph_to_step_data(
    graph,
    step_idx: int,
    *,
    token_text: str = "",
    logprob: float | None = None,
    max_edges: int = 10_000,
) -> StepData:
    """Convert a live circuit_tracer.Graph to compact StepData."""
    adj = graph.adjacency_matrix
    af = graph.active_features
    n_features = af.shape[0]

    row_idx, col_idx, weights = sparsify_edges(adj, n_features, max_edges=max_edges)

    return StepData(
        step_idx=step_idx,
        row_idx=row_idx,
        col_idx=col_idx,
        weights=weights,
        feature_ids=af.cpu().numpy().astype(np.int64),
        token_text=token_text,
        logprob=logprob,
        n_features=n_features,
    )


# ── main tracing loop ────────────────────────────────────────────────


def trace_completion(
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
    offload: Literal["cpu", "disk"] | None = "cpu",
    save_raw: bool = False,
) -> dict:
    """Trace a single completion: generate token-by-token with attribution."""
    tokenizer = model.tokenizer
    prompt_id = f"prompt_{prompt_idx:03d}"
    completion_id = f"completion_{completion_idx:03d}"
    completion_dir = output_dir / prompt_id / completion_id
    completion_dir.mkdir(parents=True, exist_ok=True)

    input_ids = model.ensure_tokenized(prompt).unsqueeze(0)
    generated_token_ids: list[int] = []
    step_records: list[dict[str, Any]] = []

    # Stop tokens
    _candidate_stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    _eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(_eot, int) and _eot != tokenizer.unk_token_id:
        _candidate_stop_ids.append(_eot)
    stop_token_ids = {tid for tid in _candidate_stop_ids if tid is not None}

    print(f"\n  [{prompt_id}/{completion_id}] Starting trace (temp={temperature})")

    for step_idx in range(max_steps):
        graph = extract_graph(
            model,
            input_ids[0],
            max_feature_nodes=max_feature_nodes,
            offload=offload,
        )

        token_result = generate_next_token(model, input_ids, temperature=temperature)
        next_token_id = token_result["token_id"]
        next_token_text = token_result["token_text"]
        generated_token_ids.append(next_token_id)

        # Save compact .npz
        sd = graph_to_step_data(
            graph,
            step_idx,
            token_text=next_token_text,
            logprob=token_result["token_logprob"],
            max_edges=max_edges,
        )
        save_compact(sd, completion_dir / f"step_{step_idx:03d}.npz")

        # Optionally save raw .pt
        if save_raw:
            graph.to_pt(str(completion_dir / f"step_{step_idx:03d}.pt"))

        step_record = {
            "step_index": step_idx,
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "next_token_logprob": token_result["token_logprob"],
            "n_active_features": sd.n_features,
            "n_edges_retained": len(sd.weights),
            "stop_reason": "eos" if next_token_id in stop_token_ids else None,
        }
        step_records.append(step_record)

        # Progress (every 10 steps)
        if step_idx % 10 == 0 or next_token_id in stop_token_ids:
            print(
                f"    Step {step_idx:03d}: "
                f"tok={next_token_text!r} feat={sd.n_features} edges={len(sd.weights)}"
            )

        del graph, sd
        input_ids = token_result["next_input_ids"]

        if next_token_id in stop_token_ids:
            print(f"    Stop token at step {step_idx}")
            break

    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    manifest = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "prompt": prompt,
        "completion_text": completion_text,
        "n_steps_traced": len(step_records),
        "temperature": temperature,
        "max_feature_nodes": max_feature_nodes,
        "max_edges": max_edges,
        "offload": offload,
        "save_raw": save_raw,
        "steps": step_records,
    }
    manifest_path = completion_dir / "completion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"    Saved manifest: {manifest_path}")
    print(f"    Answer (first 200 chars): {completion_text[:200]}")
    return manifest


def run_pipeline(args: argparse.Namespace) -> None:
    model = load_model()
    install_feature_cap_patch(model.transcoders, args.max_feature_nodes)  # type: ignore[union-attr]
    examples = load_gsm8k_examples(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offload = None if args.no_offload else "cpu"

    # Save run config
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
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    total = len(examples) * args.completions
    done = 0

    for prompt_idx, example in enumerate(examples):
        prompt = format_prompt(model.tokenizer, example["question"])  # type: ignore[unresolved-attribute]

        # Save ground truth alongside prompt traces
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = {
            "gsm8k_index": prompt_idx,
            "question": example["question"],
            "ground_truth_answer": example["answer"],
            "prompt_text": prompt,
        }
        (prompt_dir / "prompt_meta.json").write_text(json.dumps(prompt_meta, indent=2))

        for comp_idx in range(args.completions):
            done += 1
            print(f"\n{'=' * 60}")
            print(
                f"Completion {done}/{total}: prompt {prompt_idx}, completion {comp_idx}"
            )
            print(f"{'=' * 60}")

            trace_completion(
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
        description="Multi-prompt tracing pipeline for temporal circuit stability"
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
        default="/fs/scratch/PAS3272/kopanev.1/traces",
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
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)
