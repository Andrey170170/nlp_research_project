"""
Minimal exploration script for the circuit-tracer pipeline.
Goal: understand inputs, outputs, and API on 1 GSM8K example.

Run: uv run python explore_pipeline.py
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("experiments/explore")

# ── 1. Load one GSM8K example ──────────────────────────────────────────────

def load_gsm8k_example(idx: int = 0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    example = ds[idx]
    print(f"Question: {example['question'][:200]}...")
    print(f"Answer: {example['answer'][:200]}...")
    return example


# ── 2. Format prompt ──────────────────────────────────────────────────────

def format_prompt(tokenizer, question: str) -> str:
    """Format using the model's chat template (required by Gemma-3-IT transcoders)."""
    messages = [
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Please solve this step by step and end with 'Final answer: <number>'."
        )},
    ]
    # add_generation_prompt=True appends <start_of_turn>model\n
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ── 3. Load model + transcoders ───────────────────────────────────────────

HF_REPO = "google/gemma-scope-2-1b-it"
CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"
NATIVE_CLT_DIR = Path("data/clt_native")


def convert_google_to_native():
    """One-time conversion from Google's format to circuit-tracer native format.

    Processes one layer at a time to keep peak memory at ~1.3 GB.
    Native format supports lazy loading (decoders stay on disk until needed).
    """
    print(f"Converting {HF_REPO}/{CLT_SUBFOLDER} to native format...")
    print("  (This only needs to run once)")

    # Download Google format files
    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{CLT_SUBFOLDER}/params_layer_*.safetensors"],
    )
    google_dir = Path(local_dir) / CLT_SUBFOLDER
    layer_files = sorted(google_dir.glob("params_layer_*.safetensors"))
    n_layers = len(layer_files)
    print(f"  Found {n_layers} layer files")

    NATIVE_CLT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert one layer at a time to minimize memory
    for i, layer_file in enumerate(layer_files):
        print(f"  Converting layer {i}/{n_layers}...", end="\r")
        params = load_file(str(layer_file), device="cpu")

        # Google format: w_enc is [d_model, d_sae], circuit-tracer wants [d_sae, d_model]
        w_enc_i = params["w_enc"].T.contiguous().to(DTYPE)
        b_enc_i = params["b_enc"].to(DTYPE)
        b_dec_i = params["b_dec"].to(DTYPE)
        threshold_i = params["threshold"].to(DTYPE)

        # Save encoder file (W_enc, b_enc, b_dec, threshold)
        enc_dict = {
            f"W_enc_{i}": w_enc_i,
            f"b_enc_{i}": b_enc_i,
            f"b_dec_{i}": b_dec_i,
            f"threshold_{i}": threshold_i,
        }
        save_file(enc_dict, str(NATIVE_CLT_DIR / f"W_enc_{i}.safetensors"))

        # Google format: w_dec is [d_sae, n_layers, d_model]
        # For layer i, only outputs to layers i..n_layers-1
        w_dec_i = params["w_dec"][:, i:, :].contiguous().to(DTYPE)
        dec_dict = {f"W_dec_{i}": w_dec_i}
        save_file(dec_dict, str(NATIVE_CLT_DIR / f"W_dec_{i}.safetensors"))

        # Handle affine skip connection if present
        if "affine_skip_connection" in params:
            # Save per-layer skip weights alongside encoder for simplicity
            # We'll reconstruct the full W_skip after loading
            skip_dict = {f"W_skip_{i}": params["affine_skip_connection"].to(DTYPE)}
            save_file(skip_dict, str(NATIVE_CLT_DIR / f"W_skip_{i}.safetensors"))

        del params, w_enc_i, b_enc_i, b_dec_i, threshold_i, w_dec_i

    print(f"\n  Saved native format to {NATIVE_CLT_DIR}")


def load_transcoders():
    """Load CLT with lazy decoder loading (~1 MB GPU instead of ~12 GB)."""
    if not (NATIVE_CLT_DIR / "W_enc_0.safetensors").exists():
        convert_google_to_native()

    print("Loading transcoders (lazy decoder mode)...")
    transcoders = load_clt(
        str(NATIVE_CLT_DIR),
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        lazy_decoder=True,
        lazy_encoder=True,
        dtype=DTYPE,
        device=torch.device("cpu"),  # load params to CPU first
    )
    return transcoders


def load_model():
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  GPU memory before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    transcoders = load_transcoders()

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name="google/gemma-3-1b-it",
        transcoders=transcoders,
        dtype=DTYPE,
        backend="nnsight",
    )

    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model


# ── 4. Generate a completion (without tracing, just to see output) ────────

def generate_completion(model, prompt: str, max_new_tokens: int = 200):
    print("\nGenerating completion...")
    tokenizer = model.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    completion = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Completion: {completion[:500]}")
    return completion


# ── 5. Extract attribution graph ──────────────────────────────────────────

def extract_graph(model, prompt: str):
    print("\nExtracting attribution graph...")
    print(f"  GPU memory before attribution: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=5,         # attribute top-5 logits (reduce for speed)
        desired_logit_prob=0.9,  # mass coverage for logit selection
        batch_size=64,           # reduce for lower VRAM usage on consumer GPU
        max_feature_nodes=4096,  # cap features to save memory
        offload="cpu",           # offload model layers to CPU after forward pass
        verbose=True,
    )

    print(f"  GPU memory after attribution: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return graph


# ── 6. Inspect graph structure ────────────────────────────────────────────

def inspect_graph(graph):
    print("\n" + "=" * 60)
    print("GRAPH INSPECTION")
    print("=" * 60)

    print(f"\nGraph type: {type(graph)}")
    print(f"Graph attributes: {[a for a in dir(graph) if not a.startswith('_')]}")

    # Adjacency matrix
    if hasattr(graph, "adjacency_matrix"):
        adj = graph.adjacency_matrix
        print(f"\nAdjacency matrix shape: {adj.shape}")
        print(f"Adjacency matrix dtype: {adj.dtype}")
        print(f"Non-zero entries: {(adj != 0).sum().item()}")
        print(f"Value range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")

    # Active features
    if hasattr(graph, "active_features"):
        af = graph.active_features
        print(f"\nActive features shape: {af.shape}")
        print("  (rows = features, cols = [layer, position, feature_idx])")
        print(f"Number of active features: {af.shape[0]}")
        if af.shape[0] > 0:
            print(f"Layers represented: {sorted(af[:, 0].unique().tolist())}")
            print(f"Positions represented: {sorted(af[:, 1].unique().tolist())}")

    # Logit targets
    if hasattr(graph, "logit_targets"):
        print(f"\nLogit targets: {graph.logit_targets}")

    # Input tokens
    if hasattr(graph, "input_tokens"):
        print(f"\nInput tokens shape: {graph.input_tokens.shape}")
    if hasattr(graph, "input_string"):
        print(f"Input string (first 200 chars): {graph.input_string[:200]}")

    return graph


# ── 7. Save graph ─────────────────────────────────────────────────────────

def save_graph(graph, name: str = "test_graph"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as .pt (circuit-tracer native format)
    pt_path = OUTPUT_DIR / f"{name}.pt"
    graph.to_pt(str(pt_path))
    print(f"\nSaved graph to {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata as JSON for inspection
    meta = {
        "n_active_features": graph.active_features.shape[0] if hasattr(graph, "active_features") else None,
        "adjacency_shape": list(graph.adjacency_matrix.shape) if hasattr(graph, "adjacency_matrix") else None,
        "input_string": graph.input_string if hasattr(graph, "input_string") else None,
    }
    meta_path = OUTPUT_DIR / f"{name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata to {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print()

    # Step 1: Load model (need tokenizer for prompt formatting)
    model = load_model()

    # Step 2: Load and format a GSM8K example
    example = load_gsm8k_example(idx=0)
    prompt = format_prompt(model.tokenizer, example["question"])
    print(f"\nFormatted prompt:\n{prompt}")
    print()

    # Step 4: Quick generation test (no tracing)
    completion = generate_completion(model, prompt)

    # Step 5: Extract attribution graph
    # Use a minimal prompt first to verify the pipeline fits in VRAM.
    # Full GSM8K prompts (~50+ tokens) need >16 GB for 262k CLT attribution.
    tiny_prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "2+2="}],
        tokenize=False, add_generation_prompt=True,
    )
    n_tokens = len(model.tokenizer.encode(tiny_prompt))
    print(f"\nUsing tiny prompt for attribution test ({n_tokens} tokens): {tiny_prompt!r}")
    graph = extract_graph(model, tiny_prompt)

    # Step 6: Inspect what we got
    inspect_graph(graph)

    # Step 7: Save
    save_graph(graph)

    print("\nPipeline exploration complete!")
    print("  Next steps:")
    print("  - Check experiments/explore/ for saved outputs")
    print("  - Adjust batch_size / max_feature_nodes if OOM")
    print("  - Try tracing per-token during generation for temporal graphs")
