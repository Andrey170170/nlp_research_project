"""
Exploration script for the circuit-tracer pipeline.
Designed for OSC (H100, 80GB VRAM).

Run: python explore_pipeline.py
"""

import json
import gc
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from circuit_tracer import ReplacementModel, attribute
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("experiments/explore")
TRACE_DIR = Path("experiments/traces")
ATTRIBUTION_BATCH_SIZE = 256
ATTRIBUTION_MAX_FEATURE_NODES = 32768
ATTRIBUTION_OFFLOAD = "cpu"
MAX_TRACE_STEPS = 256

HF_REPO = "google/gemma-scope-2-1b-it"
CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"


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
    params_list = [load_file(path, device=device.type) for path in ordered_paths]
    state_dict_raw = {
        key: torch.stack([params[key] for params in params_list])
        for key in params_list[0].keys()
    }

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


def _prune_sparse_features(
    features: torch.Tensor, max_active_features: int | None
) -> torch.Tensor:
    features = features.coalesce()
    if max_active_features is None or features._nnz() <= max_active_features:
        return features

    keep = torch.topk(
        features.values().abs(), k=max_active_features, sorted=False
    ).indices
    return torch.sparse_coo_tensor(
        features.indices()[:, keep],
        features.values()[keep],
        size=features.shape,
        device=features.device,
        dtype=features.dtype,
    ).coalesce()


def _select_active_encoder_vectors(transcoder, features: torch.Tensor) -> torch.Tensor:
    layer_idx, _, feat_idx = features.indices()
    encoder_vectors = []
    for layer_id in range(transcoder.n_layers):
        current_layer = layer_idx == layer_id
        if current_layer.any():
            encoder_vectors.append(
                transcoder._get_encoder_weights(layer_id)[feat_idx[current_layer]]
            )

    if encoder_vectors:
        return torch.cat(encoder_vectors, dim=0)
    return torch.empty(
        (0, transcoder.d_model), device=features.device, dtype=transcoder.dtype
    )


def load_gsm8k_example(idx: int = 0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    example = ds[idx]
    print(f"Question: {example['question'][:200]}...")
    print(f"Answer: {example['answer'][:200]}...")
    return example


def format_prompt(tokenizer, question: str) -> str:
    """Format using the model's chat template."""
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


def load_model():
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
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

    print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


def greedy_generate_next_token(model, input_ids: torch.Tensor) -> dict[str, Any]:
    tokenizer = model.tokenizer

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
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


def save_graph(graph, output_stem: Path, extra_meta: dict[str, Any] | None = None):
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    pt_path = output_stem.with_suffix(".pt")
    graph.to_pt(str(pt_path))

    meta = {
        "n_active_features": (
            graph.active_features.shape[0]
            if hasattr(graph, "active_features")
            else None
        ),
        "adjacency_shape": (
            list(graph.adjacency_matrix.shape)
            if hasattr(graph, "adjacency_matrix")
            else None
        ),
        "input_string": graph.input_string if hasattr(graph, "input_string") else None,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = output_stem.with_name(f"{output_stem.name}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved graph to {pt_path}")
    print(f"  Saved metadata to {meta_path}")


def extract_graph(model, prompt: str | torch.Tensor | list[int]):
    print("\nExtracting attribution graph...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"  GPU memory before attribution: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=5,
        desired_logit_prob=0.9,
        batch_size=ATTRIBUTION_BATCH_SIZE,
        max_feature_nodes=ATTRIBUTION_MAX_FEATURE_NODES,
        offload=ATTRIBUTION_OFFLOAD,
        verbose=True,
    )

    print(
        f"  GPU memory after attribution: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )
    return graph


def trace_generation_steps(
    model,
    prompt: str,
    *,
    prompt_id: str = "prompt_000",
    completion_id: str = "completion_000",
    max_new_tokens: int = MAX_TRACE_STEPS,
):
    print("\nTracing generation step by step...")
    tokenizer = model.tokenizer
    completion_dir = TRACE_DIR / prompt_id / completion_id
    completion_dir.mkdir(parents=True, exist_ok=True)

    input_ids = model.ensure_tokenized(prompt).unsqueeze(0)
    generated_token_ids: list[int] = []
    step_records: list[dict[str, Any]] = []
    # Gemma-3 uses <end_of_turn> (id 106) to signal generation end, not <eos>.
    # Collect all plausible stop tokens so we don't waste GPU steps on junk.
    _candidate_stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]
    _eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(_eot, int) and _eot != tokenizer.unk_token_id:
        _candidate_stop_ids.append(_eot)
    stop_token_ids = {tid for tid in _candidate_stop_ids if tid is not None}

    for step_idx in range(max_new_tokens):
        print(f"\nStep {step_idx:03d}")
        prefix_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        graph = extract_graph(model, input_ids[0])

        token_result = greedy_generate_next_token(model, input_ids)
        next_token_id = token_result["token_id"]
        next_token_text = token_result["token_text"]
        generated_token_ids.append(next_token_id)
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        step_record = {
            "step_index": step_idx,
            "prompt_id": prompt_id,
            "completion_id": completion_id,
            "prefix_token_count": int(input_ids.shape[1]),
            "prefix_text": prefix_text,
            "generated_token_ids": list(generated_token_ids),
            "generated_text": generated_text,
            "next_token_id": next_token_id,
            "next_token_text": next_token_text,
            "next_token_logprob": token_result["token_logprob"],
            "stop_reason": "eos" if next_token_id in stop_token_ids else None,
        }
        save_graph(
            graph, completion_dir / f"step_{step_idx:03d}", extra_meta=step_record
        )
        step_records.append(step_record)

        print(
            f"  Next token: id={next_token_id} text={next_token_text!r} "
            f"logprob={token_result['token_logprob']}"
        )

        input_ids = token_result["next_input_ids"]
        if next_token_id in stop_token_ids:
            print("  Encountered stop token, ending generation trace.")
            break

    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    run_manifest = {
        "prompt_id": prompt_id,
        "completion_id": completion_id,
        "prompt": prompt,
        "final_input_text": tokenizer.decode(input_ids[0], skip_special_tokens=False),
        "completion_text": completion_text,
        "n_steps_traced": len(step_records),
        "step_files": [
            f"step_{record['step_index']:03d}.pt" for record in step_records
        ],
        "steps": step_records,
    }
    manifest_path = completion_dir / "completion.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2))
    print(f"\nSaved completion manifest to {manifest_path}")
    print(f"Completion text: {completion_text[:500]}")
    return run_manifest


def inspect_graph(graph):
    print("\n" + "=" * 60)
    print("GRAPH INSPECTION")
    print("=" * 60)

    print(f"\nGraph type: {type(graph)}")
    print(f"Graph attributes: {[a for a in dir(graph) if not a.startswith('_')]}")

    if hasattr(graph, "adjacency_matrix"):
        adj = graph.adjacency_matrix
        print(f"\nAdjacency matrix shape: {adj.shape}")
        print(f"Adjacency matrix dtype: {adj.dtype}")
        print(f"Non-zero entries: {(adj != 0).sum().item()}")
        print(f"Value range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")

    if hasattr(graph, "active_features"):
        af = graph.active_features
        print(f"\nActive features shape: {af.shape}")
        print("  (rows = features, cols = [layer, position, feature_idx])")
        print(f"Number of active features: {af.shape[0]}")
        if af.shape[0] > 0:
            print(f"Layers represented: {sorted(af[:, 0].unique().tolist())}")
            print(f"Positions represented: {sorted(af[:, 1].unique().tolist())}")

    if hasattr(graph, "logit_targets"):
        print(f"\nLogit targets: {graph.logit_targets}")

    if hasattr(graph, "input_tokens"):
        print(f"\nInput tokens shape: {graph.input_tokens.shape}")
    if hasattr(graph, "input_string"):
        print(f"Input string (first 200 chars): {graph.input_string[:200]}")

    return graph


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    model = load_model()

    example = load_gsm8k_example(idx=0)
    prompt = format_prompt(model.tokenizer, example["question"])
    print(f"\nFormatted prompt:\n{prompt}")
    print()

    manifest = trace_generation_steps(
        model,
        prompt,
        prompt_id="prompt_000",
        completion_id="completion_000",
    )
    final_step_path = (
        TRACE_DIR
        / manifest["prompt_id"]
        / manifest["completion_id"]
        / f"step_{manifest['n_steps_traced'] - 1:03d}.pt"
    )

    print("\nPipeline exploration complete!")
    print("  Next steps:")
    print(f"  - Check {final_step_path.parent} for per-step graph artifacts")
    print("  - Use completion.json as the run manifest for downstream processing")
