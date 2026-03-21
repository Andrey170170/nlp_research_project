"""
Exploration script for the circuit-tracer pipeline.
Designed for OSC (H100, 80GB VRAM).

Run: python explore_pipeline.py
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from circuit_tracer import ReplacementModel, attribute

DEVICE = "cuda"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("experiments/explore")

HF_REPO = "google/gemma-scope-2-1b-it"
CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"


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
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ── 3. Load model + transcoders ───────────────────────────────────────────

def load_model():
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  GPU memory before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Google's repo has config.json not config.yaml, so auto-loading via
    # transcoder_set string doesn't work. Load transcoders manually.
    from huggingface_hub import snapshot_download
    from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt

    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{CLT_SUBFOLDER}/params_layer_*.safetensors"],
    )
    clt_dir = Path(local_dir) / CLT_SUBFOLDER
    layer_files = sorted(clt_dir.glob("params_layer_*.safetensors"))
    paths = {i: str(f) for i, f in enumerate(layer_files)}
    print(f"  Found {len(paths)} transcoder layer files")

    transcoders = load_gemma_scope_2_clt(
        paths=paths,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        dtype=DTYPE,
    )

    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name="google/gemma-3-1b-it",
        transcoders=transcoders,
        dtype=DTYPE,
        backend="nnsight",
    )

    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model


# ── 4. Generate a completion ──────────────────────────────────────────────

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
        max_n_logits=5,
        desired_logit_prob=0.9,
        batch_size=256,
        max_feature_nodes=8192,
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


# ── 7. Save graph ─────────────────────────────────────────────────────────

def save_graph(graph, name: str = "test_graph"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = OUTPUT_DIR / f"{name}.pt"
    graph.to_pt(str(pt_path))
    print(f"\nSaved graph to {pt_path} ({pt_path.stat().st_size / 1e6:.1f} MB)")

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

    # Step 1: Load model
    model = load_model()

    # Step 2: Load and format a GSM8K example
    example = load_gsm8k_example(idx=0)
    prompt = format_prompt(model.tokenizer, example["question"])
    print(f"\nFormatted prompt:\n{prompt}")
    print()

    # Step 3: Quick generation test
    completion = generate_completion(model, prompt)

    # Step 4: Extract attribution graph for the prompt
    graph = extract_graph(model, prompt)

    # Step 5: Inspect what we got
    inspect_graph(graph)

    # Step 6: Save
    save_graph(graph)

    print("\nPipeline exploration complete!")
    print("  Next steps:")
    print("  - Check experiments/explore/ for saved outputs")
    print("  - Try tracing per-token during generation for temporal graphs")
