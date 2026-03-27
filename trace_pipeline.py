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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Literal

import numpy as np
import torch
from circuit_tracer import ReplacementModel, attribute
from datasets import load_dataset

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

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


def load_gemma_scope_2_clt_native(
    paths: dict[int, str],
    feature_input_hook: str = "hook_resid_mid",
    feature_output_hook: str = "hook_mlp_out",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
    decoder_chunk_size: int = 256,
    cross_batch_decoder_cache_bytes: int | None = None,
):
    """Load GemmaScope-2 CLTs via the fork-native loader."""
    from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt

    if device is None:
        device = torch.device(DEVICE)

    loader_kwargs: dict[str, Any] = {
        "paths": paths,
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
        "device": device,
        "dtype": dtype,
        "lazy_encoder": lazy_encoder,
        "lazy_decoder": lazy_decoder,
        # Fork-only kwarg; kept dynamic until local env is synced to the fork.
        "decoder_chunk_size": decoder_chunk_size,
    }
    if cross_batch_decoder_cache_bytes is not None:
        loader_kwargs["cross_batch_decoder_cache_bytes"] = (
            cross_batch_decoder_cache_bytes
        )
    return load_gemma_scope_2_clt(**loader_kwargs)


def load_model(
    *,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
    decoder_chunk_size: int = 256,
    exact_chunked_decoder: bool = True,
    cross_batch_decoder_cache_bytes: int | None = None,
) -> ReplacementModel:
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(
        "  Transcoder loader: fork-native GemmaScope-2 CLT "
        f"(lazy_encoder={lazy_encoder}, lazy_decoder={lazy_decoder}, "
        f"decoder_chunk_size={decoder_chunk_size}, "
        f"exact_chunked_decoder={exact_chunked_decoder}, "
        f"cross_batch_decoder_cache_bytes={cross_batch_decoder_cache_bytes or 0})"
    )
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

    transcoders = load_gemma_scope_2_clt_native(
        paths=paths,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device(DEVICE),
        dtype=DTYPE,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
        decoder_chunk_size=decoder_chunk_size,
        cross_batch_decoder_cache_bytes=cross_batch_decoder_cache_bytes,
    )
    transcoders.exact_chunked_decoder = exact_chunked_decoder

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name="google/gemma-3-1b-it",
        transcoders=transcoders,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        backend="nnsight",
    )

    if torch.cuda.is_available():
        print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


# ── data loading ─────────────────────────────────────────────────────


def parse_gsm8k_indices(
    inline_indices: str | None = None,
    indices_file: str | None = None,
) -> list[int] | None:
    """Parse explicit GSM8K indices from CLI inputs."""
    values: list[int] = []

    if inline_indices:
        for chunk in inline_indices.split(","):
            chunk = chunk.strip()
            if chunk:
                values.append(int(chunk))

    if indices_file:
        raw = Path(indices_file).read_text().strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = [line.strip() for line in raw.splitlines() if line.strip()]

            if isinstance(parsed, list):
                values.extend(int(v) for v in parsed)
            else:
                raise ValueError(
                    "GSM8K indices file must contain a JSON list or newline-separated integers"
                )

    if not values:
        return None

    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def load_gsm8k_examples(n: int = 10, indices: list[int] | None = None) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if indices is not None:
        examples = [{**ds[i], "gsm8k_index": i} for i in indices]
    else:
        examples = [{**ds[i], "gsm8k_index": i} for i in range(min(n, len(ds)))]
    print(f"Loaded {len(examples)} GSM8K examples")
    return examples


def load_prepared_prompt_examples(
    prompt_text_file: str,
    prompt_meta_file: str | None = None,
) -> list[dict[str, Any]]:
    prompt_path = Path(prompt_text_file)
    prompt_text = prompt_path.read_text()
    prompt_meta: dict[str, Any] = {}
    if prompt_meta_file is not None:
        prompt_meta = json.loads(Path(prompt_meta_file).read_text())

    example = {
        "question": prompt_meta.get("question", ""),
        "answer": prompt_meta.get("ground_truth_answer", ""),
        "gsm8k_index": prompt_meta.get("gsm8k_index"),
        "prompt_text": prompt_text,
        "prompt_source": prompt_meta.get("prompt_source", "prepared_prompt"),
        "fixture_name": prompt_meta.get("fixture_name"),
        "fixture_kind": prompt_meta.get("fixture_kind", "prepared_prompt"),
        "prompt_token_count": prompt_meta.get("prompt_token_count"),
        "initial_input_token_count": prompt_meta.get("initial_input_token_count"),
        "prepared_prompt_meta": prompt_meta,
        "prepared_prompt_file": str(prompt_path),
        "prepared_prompt_meta_file": prompt_meta_file,
    }
    print(f"Loaded prepared prompt fixture from {prompt_path}")
    return [example]


def load_prompt_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prepared_prompt_file is not None:
        if args.gsm8k_indices is not None or args.gsm8k_indices_file is not None:
            raise ValueError(
                "prepared prompt inputs cannot be combined with GSM8K index selection"
            )
        return load_prepared_prompt_examples(
            args.prepared_prompt_file,
            args.prepared_prompt_meta_file,
        )

    gsm8k_indices = parse_gsm8k_indices(args.gsm8k_indices, args.gsm8k_indices_file)
    return load_gsm8k_examples(args.prompts, indices=gsm8k_indices)


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


def resolve_prompt_text(tokenizer, example: dict[str, Any]) -> str:
    prepared_prompt = example.get("prompt_text")
    if isinstance(prepared_prompt, str) and prepared_prompt:
        return prepared_prompt
    return format_prompt(tokenizer, example["question"])


def build_prompt_meta_record(
    example: dict[str, Any],
    *,
    prompt_text: str,
    initial_input_token_count: int,
) -> dict[str, Any]:
    prompt_token_count = example.get("prompt_token_count")
    if prompt_token_count is None:
        prompt_token_count = initial_input_token_count

    prompt_meta = {
        "gsm8k_index": example.get("gsm8k_index"),
        "question": example.get("question", ""),
        "ground_truth_answer": example.get("answer", ""),
        "prompt_text": prompt_text,
        "prompt_source": example.get("prompt_source", "gsm8k"),
        "fixture_name": example.get("fixture_name"),
        "fixture_kind": example.get("fixture_kind"),
        "prompt_token_count": int(prompt_token_count),
        "initial_input_token_count": int(initial_input_token_count),
    }

    if example.get("prepared_prompt_file") is not None:
        prompt_meta["prepared_prompt_file"] = example["prepared_prompt_file"]
    if example.get("prepared_prompt_meta_file") is not None:
        prompt_meta["prepared_prompt_meta_file"] = example["prepared_prompt_meta_file"]

    prepared_prompt_meta = example.get("prepared_prompt_meta")
    if isinstance(prepared_prompt_meta, dict) and prepared_prompt_meta:
        prompt_meta["prepared_prompt_metadata"] = prepared_prompt_meta

    return prompt_meta


def capture_resource_snapshot() -> dict[str, float | None]:
    rss_gib = None
    if resource is not None:
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)

    snapshot: dict[str, float | None] = {
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
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )

    return snapshot


def capture_transcoder_diagnostics(model) -> dict[str, Any] | None:
    getter = getattr(
        getattr(model, "transcoders", None), "get_diagnostic_snapshot", None
    )
    if not callable(getter):
        return None

    snapshot = getter()
    if not isinstance(snapshot, dict):
        return None

    keys_of_interest = [
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
    ]
    return {key: snapshot.get(key) for key in keys_of_interest if key in snapshot}


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
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_n_logits: int = 5,
    desired_logit_prob: float = 0.9,
    offload: Literal["cpu", "disk"] | None = "cpu",
    verbose: bool = False,
    update_interval: int = 4,
    profile: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: Any | None = None,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attribute_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "batch_size": batch_size,
        "feature_batch_size": feature_batch_size,
        "logit_batch_size": logit_batch_size,
        "max_feature_nodes": max_feature_nodes,
        "offload": offload,
        "verbose": verbose,
        "update_interval": update_interval,
        # These exist in the local fork; keep dynamic to avoid stale type hints.
        "profile": profile,
        "profile_log_interval": profile_log_interval,
        "diagnostic_feature_cap": diagnostic_feature_cap,
        "sparsification": sparsification,
    }
    return attribute(**attribute_kwargs)


def validate_attribution_batch_sizes(
    batch_size: int,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("attribution_batch_size must be > 0")
    if feature_batch_size is not None:
        if feature_batch_size <= 0:
            raise ValueError("feature_batch_size must be > 0 when provided")
        if feature_batch_size > batch_size:
            raise ValueError(
                "feature_batch_size must be <= attribution_batch_size to avoid widening trace VRAM"
            )
    if logit_batch_size is not None:
        if logit_batch_size <= 0:
            raise ValueError("logit_batch_size must be > 0 when provided")
        if logit_batch_size > batch_size:
            raise ValueError(
                "logit_batch_size must be <= attribution_batch_size to avoid widening trace VRAM"
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
    attribution_batch_size: int = 256,
    feature_batch_size: int | None = None,
    logit_batch_size: int | None = None,
    max_n_logits: int = 5,
    desired_logit_prob: float = 0.9,
    offload: Literal["cpu", "disk"] | None = "cpu",
    verbose_attribution: bool = False,
    attribution_update_interval: int = 4,
    profile_attribution: bool = False,
    profile_log_interval: int = 1,
    diagnostic_feature_cap: int | None = None,
    sparsification: Any | None = None,
    save_raw: bool = False,
    prompt_token_count: int | None = None,
    prompt_source: str = "gsm8k",
    fixture_name: str | None = None,
    fixture_kind: str | None = None,
) -> dict:
    """Trace a single completion: generate token-by-token with attribution."""
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
            "prefix_token_count": int(input_ids.shape[1]),
            "generated_token_count": len(generated_token_ids),
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "next_token_logprob": token_result["token_logprob"],
            "n_active_features": sd.n_features,
            "n_edges_retained": len(sd.weights),
            "stop_reason": "eos" if next_token_id in stop_token_ids else None,
            "resource_snapshot": capture_resource_snapshot(),
            "transcoder_diagnostics": capture_transcoder_diagnostics(model),
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
        "save_raw": save_raw,
        "resource_snapshot": capture_resource_snapshot(),
        "steps": step_records,
    }
    manifest_path = completion_dir / "completion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"    Saved manifest: {manifest_path}")
    print(f"    Answer (first 200 chars): {completion_text[:200]}")
    return manifest


def run_pipeline(args: argparse.Namespace) -> None:
    validate_attribution_batch_sizes(
        args.attribution_batch_size,
        args.feature_batch_size,
        args.logit_batch_size,
    )
    model = load_model(exact_chunked_decoder=False)
    install_feature_cap_patch(model.transcoders, args.max_feature_nodes)  # type: ignore[union-attr]
    examples = load_prompt_examples(args)
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

    # Save run config
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
        "prepared_prompt_file": args.prepared_prompt_file,
        "prepared_prompt_meta_file": args.prepared_prompt_meta_file,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    total = len(examples) * args.completions
    done = 0

    for prompt_idx, example in enumerate(examples):
        prompt = resolve_prompt_text(model.tokenizer, example)  # type: ignore[unresolved-attribute]
        initial_input_token_count = int(
            model.ensure_tokenized(prompt).shape[0]  # type: ignore[unresolved-attribute]
        )

        # Save ground truth alongside prompt traces
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta = build_prompt_meta_record(
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
                save_raw=args.save_raw,
                prompt_token_count=prompt_meta["prompt_token_count"],
                prompt_source=prompt_meta["prompt_source"],
                fixture_name=prompt_meta.get("fixture_name"),
                fixture_kind=prompt_meta.get("fixture_kind"),
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
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)
