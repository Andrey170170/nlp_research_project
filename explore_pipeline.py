"""
Exploration script for the circuit-tracer pipeline.
Designed for OSC (H100, 80GB VRAM).

Run: python explore_pipeline.py
"""

import json
import gc
from pathlib import Path
from types import MethodType

import torch
from circuit_tracer import ReplacementModel, attribute
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("experiments/explore")
ATTRIBUTION_BATCH_SIZE = 256
ATTRIBUTION_MAX_FEATURE_NODES = 8192
ATTRIBUTION_PRECOMPUTE_FEATURE_CAP = 8192
ATTRIBUTION_OFFLOAD = "cpu"

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


def install_memory_safe_attribution_patch(
    transcoder,
    *,
    max_active_features: int | None,
) -> None:
    original_compute = transcoder.compute_attribution_components

    def compute_attribution_components_pruned(self, inputs, zero_positions=slice(0, 1)):
        features, _ = self.encode_sparse(inputs, zero_positions=zero_positions)
        original_nnz = features._nnz()
        features = _prune_sparse_features(
            features, max_active_features=max_active_features
        )
        if features._nnz() != original_nnz:
            print(
                f"  Pruned active features for attribution: {original_nnz} -> {features._nnz()}"
            )

        encoder_vectors = _select_active_encoder_vectors(self, features)
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

    compute_attribution_components_pruned.__name__ = original_compute.__name__
    transcoder.compute_attribution_components = MethodType(
        compute_attribution_components_pruned, transcoder
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
    install_memory_safe_attribution_patch(
        model.transcoders,
        max_active_features=ATTRIBUTION_PRECOMPUTE_FEATURE_CAP,
    )

    print(f"  GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


def generate_completion(model, prompt: str, max_new_tokens: int = 200):
    print("\nGenerating completion...")
    tokenizer = model.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )

    completion = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"Completion: {completion[:500]}")
    return completion


def extract_graph(model, prompt: str):
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


def save_graph(graph, name: str = "test_graph"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = OUTPUT_DIR / f"{name}.pt"
    graph.to_pt(str(pt_path))
    print(f"\nSaved graph to {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

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
    meta_path = OUTPUT_DIR / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata to {meta_path}")


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

    generate_completion(model, prompt)
    graph = extract_graph(model, prompt)
    inspect_graph(graph)
    save_graph(graph)

    print("\nPipeline exploration complete!")
    print("  Next steps:")
    print("  - Check experiments/explore/ for saved outputs")
    print("  - Try tracing per-token during generation for temporal graphs")
