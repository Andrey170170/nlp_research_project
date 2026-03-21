"""
Minimal exploration script for the circuit-tracer pipeline.
Goal: understand inputs, outputs, and API on 1 GSM8K example.

Run: uv run python explore_pipeline.py
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from circuit_tracer import ReplacementModel, attribute

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

def format_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        f"Please solve this step by step and end with 'Final answer: <number>'.\n"
        f"Answer:"
    )


# ── 3. Load model + transcoders ───────────────────────────────────────────

def load_model():
    print("Loading Gemma-3-1B-IT with transcoders...")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  GPU memory before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Gemma-3 requires nnsight backend
    # Transcoder set: adjust if needed based on what's available
    # Check https://huggingface.co/google/gemma-scope-2-1b-it/tree/main/clt
    model = ReplacementModel.from_pretrained(
        model_name="google/gemma-3-1b-it",
        transcoder_set="gemma",  # May need HF repo path for Gemma-3 transcoders 
        dtype=DTYPE,
        backend="nnsight",
    )

    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model


# ── 4. Generate a completion (without tracing, just to see output) ────────

def generate_completion(model, prompt: str, max_new_tokens: int = 200):
    print("\nGenerating completion...")
    # Use the model's underlying HF model for generation
    # This is just to verify the model works before tracing
    tokenizer = model.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
        offload="cpu",           # offload transcoders to CPU RAM
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
        print(f"  (rows = features, cols = [layer, position, feature_idx])")
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
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print()

    # Step 1-2: Load and format a GSM8K example
    example = load_gsm8k_example(idx=0)
    prompt = format_prompt(example["question"])
    print(f"\nFormatted prompt:\n{prompt}")
    print()

    # Step 3: Load model
    model = load_model()

    # Step 4: Quick generation test (no tracing)
    completion = generate_completion(model, prompt)

    # Step 5: Extract attribution graph for the prompt
    # NOTE: This traces the PROMPT, not a full generation.
    # For per-token tracing during generation, you'd loop:
    #   for each new token, extend prompt and call attribute() again.
    graph = extract_graph(model, prompt)

    # Step 6: Inspect what we got
    inspect_graph(graph)

    # Step 7: Save
    save_graph(graph)

    print("\n✓ Pipeline exploration complete!")
    print("  Next steps:")
    print("  - Check experiments/explore/ for saved outputs")
    print("  - Adjust batch_size / max_feature_nodes if OOM")
    print("  - Try tracing per-token during generation for temporal graphs")
