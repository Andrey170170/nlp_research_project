#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch


HF_REPO = "google/gemma-scope-2-1b-it"
CLT_SUBFOLDER = "clt/width_262k_l0_medium_affine"
MODEL_NAME = "google/gemma-3-1b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _git_head(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_dirty(path: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _layer_file_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _install_feature_cap_patch(transcoders: Any, max_features: int) -> None:
    def patched(self: Any, inputs: torch.Tensor, zero_positions: slice = slice(0, 1)):
        features, encoder_vectors = self.encode_sparse(
            inputs,
            zero_positions=zero_positions,
        )
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

        pos_ids, layer_ids, _feat_ids, decoder_vectors, encoder_to_decoder_map = (
            self.select_decoder_vectors(features)
        )
        reconstruction = self.compute_reconstruction(
            pos_ids,
            layer_ids,
            decoder_vectors,
            inputs,
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
    print(f"  Installed clean-library feature cap patch (max_features={max_features})")


def _load_clean_model(*, max_feature_nodes: int):
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder.cross_layer_transcoder import load_gemma_scope_2_clt
    from huggingface_hub import snapshot_download

    print("Loading clean upstream Gemma-3-1B-IT with GemmaScope-2 CLTs...")
    print(f"  Device: {DEVICE}, dtype: {DTYPE}")
    local_dir = snapshot_download(
        HF_REPO,
        allow_patterns=[f"{CLT_SUBFOLDER}/params_layer_*.safetensors"],
    )
    clt_dir = Path(local_dir) / CLT_SUBFOLDER
    layer_files = sorted(
        clt_dir.glob("params_layer_*.safetensors"),
        key=_layer_file_index,
    )
    paths = {i: str(path) for i, path in enumerate(layer_files)}
    if layer_files:
        # Upstream's GemmaScope-2 loader iterates range(max(paths.keys())). Add a
        # sentinel key so the final real layer is included without changing the
        # clean library code under test.
        paths[len(layer_files)] = str(layer_files[-1])
    print(f"  Found {len(layer_files)} transcoder layer files")

    transcoders = load_gemma_scope_2_clt(
        paths=paths,
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
        device=torch.device(DEVICE),
        dtype=DTYPE,
        lazy_encoder=False,
        lazy_decoder=False,
    )
    _install_feature_cap_patch(transcoders, max_feature_nodes)
    return ReplacementModel.from_pretrained_and_transcoders(
        model_name=MODEL_NAME,
        transcoders=transcoders,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        backend="nnsight",
    )


def _generate_next_token(
    model: Any, input_ids: torch.Tensor, *, temperature: float
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
    token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
    logprob = None
    if outputs.scores:
        token_scores = outputs.scores[0][0].float()
        logprob = float(torch.log_softmax(token_scores, dim=-1)[next_token_id].item())
    return {
        "next_input_ids": outputs.sequences,
        "token_id": next_token_id,
        "token_text": token_text,
        "token_logprob": logprob,
    }


def _capture_resource_snapshot() -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
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


def run(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    clean_library_root = args.clean_library_root.resolve()
    sys.path.insert(0, str(clean_library_root))

    import circuit_tracer
    from circuit_tracer import attribute

    circuit_tracer_file = Path(circuit_tracer.__file__).resolve()
    if clean_library_root not in circuit_tracer_file.parents:
        raise RuntimeError(
            "Expected clean circuit_tracer import from "
            f"{clean_library_root}, got {circuit_tracer_file}"
        )

    prompt = args.prompt_file.read_text()
    prompt_meta = (
        json.loads(args.prompt_meta_file.read_text())
        if args.prompt_meta_file is not None
        else {}
    )
    output_dir = args.output_dir.resolve()
    completion_dir = output_dir / "prompt_000" / "completion_000"
    completion_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_kind": "clean_library_toy_parity",
                "prompt_file": str(args.prompt_file),
                "prompt_meta_file": None
                if args.prompt_meta_file is None
                else str(args.prompt_meta_file),
                "clean_library_root": str(clean_library_root),
                "circuit_tracer_file": str(circuit_tracer_file),
                "project_root": str(project_root),
                "project_git_head": _git_head(project_root),
                "project_dirty_files": _git_dirty(project_root),
                "clean_library_git_head": _git_head(clean_library_root),
                "clean_library_dirty_files": _git_dirty(clean_library_root),
                "max_feature_nodes": args.max_feature_nodes,
                "max_edges": args.max_edges,
                "attribution_batch_size": args.attribution_batch_size,
                "max_n_logits": args.max_n_logits,
                "desired_logit_prob": args.desired_logit_prob,
                "temperature": args.temperature,
                "device": DEVICE,
                "dtype": str(DTYPE),
                "compact_export": "Graph.to_compact_npz",
                "chunking": "none",
            },
            indent=2,
        )
    )

    model = _load_clean_model(max_feature_nodes=args.max_feature_nodes)
    input_ids = model.ensure_tokenized(prompt).unsqueeze(0)
    prompt_dir = output_dir / "prompt_000"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt_meta.json").write_text(
        json.dumps(
            {
                **prompt_meta,
                "prompt_text": prompt,
                "initial_input_token_count": int(input_ids.shape[1]),
                "prompt_token_count": int(input_ids.shape[1]),
            },
            indent=2,
        )
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    attribution_start = time.perf_counter()
    graph = attribute(
        input_ids[0],
        model,
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.attribution_batch_size,
        max_feature_nodes=args.max_feature_nodes,
        offload=None if args.no_offload else "cpu",
        verbose=args.verbose_attribution,
        update_interval=args.attribution_update_interval,
    )
    attribution_seconds = time.perf_counter() - attribution_start

    generation_start = time.perf_counter()
    token_result = _generate_next_token(model, input_ids, temperature=args.temperature)
    token_generation_seconds = time.perf_counter() - generation_start
    save_start = time.perf_counter()
    graph.to_compact_npz(
        completion_dir / "step_000.npz",
        step_idx=0,
        max_edges=args.max_edges,
        token_text=token_result["token_text"],
        logprob=token_result["token_logprob"],
    )
    artifact_save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - start

    step_record = {
        "step_index": 0,
        "prefix_token_count": int(input_ids.shape[1]),
        "generated_token_count": 1,
        "next_token_id": token_result["token_id"],
        "next_token_text": token_result["token_text"],
        "next_token_logprob": token_result["token_logprob"],
        "n_active_features": int(graph.active_features.shape[0]),
        "n_edges_retained": int(
            len(__import__("numpy").load(completion_dir / "step_000.npz")["weights"])
        ),
        "attribution_seconds": round(attribution_seconds, 6),
        "token_generation_seconds": round(token_generation_seconds, 6),
        "artifact_save_seconds": round(artifact_save_seconds, 6),
        "step_end_to_end_seconds": round(total_seconds, 6),
        "resource_snapshot": _capture_resource_snapshot(),
    }
    (completion_dir / "completion.json").write_text(
        json.dumps(
            {
                "prompt_id": "prompt_000",
                "completion_id": "completion_000",
                "prompt_source": prompt_meta.get("prompt_source", "toy_logic"),
                "fixture_name": prompt_meta.get("fixture_name", "toy_dax_wug"),
                "generated_token_count": 1,
                "n_steps_traced": 1,
                "steps": [step_record],
            },
            indent=2,
        )
    )
    print(json.dumps(step_record, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean-library toy parity trace")
    parser.add_argument("--clean-library-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-meta-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-feature-nodes", type=int, default=8192)
    parser.add_argument("--max-edges", type=int, default=20000)
    parser.add_argument("--attribution-batch-size", type=int, default=128)
    parser.add_argument("--max-n-logits", type=int, default=3)
    parser.add_argument("--desired-logit-prob", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attribution-update-interval", type=int, default=4)
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--verbose-attribution", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
