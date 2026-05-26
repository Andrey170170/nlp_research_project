"""Generate scenario configs for decoder-aware vs exact comparison.

Produces two scenarios for a single GSM8K prompt:

  1. exact_compact  — compact chunked trace, top-K edge selection (production baseline)
  2. exact_save_raw — same attribution but with --save-raw, giving full Graph objects.
                      graph_to_step_data() reads activation_values and uses decoder-aware
                      scoring (s_i = a_i^2 * ||D_i||_F^2) to select edges.
                      Also saves raw .pt files for post-hoc top-K reanalysis.

The analysis script (analyze_decoder_aware_comparison.py) then:
  - Loads .npz files from exact_compact  → top-K edge sets
  - Loads .npz files from exact_save_raw → decoder-aware edge sets
  - Loads .pt  files from exact_save_raw → re-applies top-K via step_from_pt(activation_values=None)
    as an independent check
  - Compares per-layer feature inclusion, edge mass, Jaccard, and layer-collapse metrics

Output: experiments/generated/decoder_aware_comparison_scenarios.json

Usage:
    uv run python experiments/build_decoder_aware_comparison_configs.py [--prompt-index N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).with_name("generated")


def _shared_defaults() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "completions": 1,
        "max_steps": 3,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "attribution_batch_size": 32,
        "decoder_chunk_size": 512,
        "verbose_attribution": True,
        "profile_attribution": True,
        "profile_log_interval": 1,
        "attribution_update_interval": 4,
        "save_raw": False,
        "no_offload": False,
        "no_lazy_encoder": False,
        "no_lazy_decoder": False,
    }


def build_configs(prompt_index: int) -> dict[str, Any]:
    p = f"p{prompt_index:04d}"

    scenarios: list[dict[str, Any]] = [
        {
            # Compact path: attribute_nnsight(compact_output=True) → compact_result_to_step_data()
            # Edge selection: global top-K by absolute weight.
            "name": f"exact_compact_{p}",
            "stage": "decoder_aware_comparison",
            "method": "exact",
            "gsm8k_indices": [prompt_index],
            "save_raw": False,
        },
        {
            # Full-graph path: attribute() → graph_to_step_data()
            # Edge selection: decoder-aware, s_i = a_i^2 * ||D_i||_F^2
            # Also saves .pt files for independent top-K re-analysis.
            "name": f"exact_save_raw_{p}",
            "stage": "decoder_aware_comparison",
            "method": "exact",
            "gsm8k_indices": [prompt_index],
            "save_raw": True,
        },
    ]

    return {
        "defaults": _shared_defaults(),
        "metadata": {
            "prompt_index": prompt_index,
            "stage": "decoder_aware_comparison",
            "description": (
                "Compact top-K path vs full-graph decoder-aware path "
                "(s_i = a_i^2 * ||D_i||_F^2). "
                "Both call the same underlying attribution algorithm; "
                "difference is purely in post-attribution edge selection. "
                "The save_raw scenario also persists .pt files for "
                "independent top-K re-analysis."
            ),
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decoder-aware vs compact top-K comparison scenario config"
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=94,
        help="GSM8K test index to use (default: 94, a medium-length calibration prompt)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_DIR / "decoder_aware_comparison_scenarios.json",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_configs(args.prompt_index)
    args.output_file.write_text(json.dumps(config, indent=2))
    print(f"Wrote {len(config['scenarios'])} scenarios to {args.output_file}")
    for s in config["scenarios"]:
        print(
            f"  {s['name']}  method={s['method']}  "
            f"save_raw={s.get('save_raw', False)}  gsm8k={s['gsm8k_indices']}"
        )


if __name__ == "__main__":
    main()
