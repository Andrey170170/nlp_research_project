from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SETS = Path(__file__).with_name("sparsification_prompt_sets.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated")
CALIBRATION_BUDGETS = [16_000, 32_000, 64_000, 128_000]


def _base_defaults() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_feature_nodes": 8192,
        "max_edges": 20000,
        "max_n_logits": 3,
        "desired_logit_prob": 0.8,
        "verbose_attribution": True,
        "profile_attribution": True,
        "profile_log_interval": 1,
        "attribution_update_interval": 4,
        "save_raw": False,
        "no_offload": False,
        "no_lazy_encoder": False,
        "no_lazy_decoder": False,
    }


def build_calibration_config(
    prompt_sets: dict[str, Any], *, per_layer_position_topk: int
) -> dict[str, Any]:
    defaults = _base_defaults() | {
        "completions": 1,
        "max_steps": 2,
        "attribution_batch_size": 32,
        "decoder_chunk_size": 512,
    }

    scenarios: list[dict[str, Any]] = []
    for prompt_idx in prompt_sets["calibration"]:
        scenarios.append(
            {
                "name": f"cal_exact_p{prompt_idx:04d}",
                "stage": "calibration",
                "method": "exact",
                "gsm8k_indices": [prompt_idx],
            }
        )
        scenarios.append(
            {
                "name": f"cal_old_cap_p{prompt_idx:04d}",
                "stage": "calibration",
                "method": "old_patch",
                "gsm8k_indices": [prompt_idx],
            }
        )
        for budget in CALIBRATION_BUDGETS:
            scenarios.append(
                {
                    "name": f"cal_sparse_{budget // 1000:03d}k_p{prompt_idx:04d}",
                    "stage": "calibration",
                    "method": "sparse",
                    "gsm8k_indices": [prompt_idx],
                    "sparsify_per_layer_position_topk": per_layer_position_topk,
                    "sparsify_global_cap": budget,
                }
            )

    return {
        "defaults": defaults,
        "metadata": {
            "prompt_sets_file": str(DEFAULT_PROMPT_SETS),
            "stage": "calibration",
            "per_layer_position_topk": per_layer_position_topk,
            "calibration_budgets": CALIBRATION_BUDGETS,
        },
        "scenarios": scenarios,
    }


def build_main_config(
    prompt_sets: dict[str, Any],
    *,
    per_layer_position_topk: int,
    global_cap: int,
    include_exact_subset: bool,
    include_robustness: bool,
) -> dict[str, Any]:
    defaults = _base_defaults() | {
        "completions": 1,
        "max_steps": 3,
        "attribution_batch_size": 32,
        "decoder_chunk_size": 512,
    }

    scenarios: list[dict[str, Any]] = []
    for prompt_idx in prompt_sets["main"]:
        scenarios.append(
            {
                "name": f"main_old_cap_p{prompt_idx:04d}",
                "stage": "main",
                "method": "old_patch",
                "gsm8k_indices": [prompt_idx],
            }
        )
        scenarios.append(
            {
                "name": f"main_sparse_{global_cap // 1000:03d}k_p{prompt_idx:04d}",
                "stage": "main",
                "method": "sparse",
                "gsm8k_indices": [prompt_idx],
                "sparsify_per_layer_position_topk": per_layer_position_topk,
                "sparsify_global_cap": global_cap,
            }
        )

    if include_exact_subset:
        for prompt_idx in prompt_sets["calibration"]:
            scenarios.append(
                {
                    "name": f"main_exact_ref_p{prompt_idx:04d}",
                    "stage": "main_reference",
                    "method": "exact",
                    "gsm8k_indices": [prompt_idx],
                    "max_steps": 2,
                }
            )

    if include_robustness:
        for prompt_idx in prompt_sets["robustness"]:
            for method in ["old_patch", "sparse"]:
                scenario = {
                    "name": f"robust_{method}_p{prompt_idx:04d}",
                    "stage": "robustness",
                    "method": method,
                    "gsm8k_indices": [prompt_idx],
                    "temperature": 0.7,
                    "completions": 2,
                    "max_steps": 3,
                }
                if method == "sparse":
                    scenario["sparsify_per_layer_position_topk"] = (
                        per_layer_position_topk
                    )
                    scenario["sparsify_global_cap"] = global_cap
                scenarios.append(scenario)

    return {
        "defaults": defaults,
        "metadata": {
            "prompt_sets_file": str(DEFAULT_PROMPT_SETS),
            "stage": "main",
            "selected_per_layer_position_topk": per_layer_position_topk,
            "selected_global_cap": global_cap,
            "include_exact_subset": include_exact_subset,
            "include_robustness": include_robustness,
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration/main scenario files for the sparsification experiment"
    )
    parser.add_argument(
        "--prompt-sets-file",
        type=Path,
        default=DEFAULT_PROMPT_SETS,
        help="JSON file containing calibration/main GSM8K prompt indices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated scenario files will be written",
    )
    parser.add_argument(
        "--per-layer-position-topk",
        type=int,
        default=16,
        help="Per-(layer, position) candidate cap used in sparse scenarios",
    )
    parser.add_argument(
        "--selected-global-cap",
        type=int,
        default=None,
        help="Chosen global candidate cap for the main experiment stage",
    )
    parser.add_argument(
        "--include-exact-subset",
        action="store_true",
        help="Include a tiny exact-reference subset in the main scenario file",
    )
    parser.add_argument(
        "--include-robustness",
        action="store_true",
        help="Include a small sampled-completion robustness subset in the main scenario file",
    )
    args = parser.parse_args()

    prompt_sets = json.loads(args.prompt_sets_file.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = build_calibration_config(
        prompt_sets,
        per_layer_position_topk=args.per_layer_position_topk,
    )
    calibration_path = args.output_dir / "sparsification_calibration_scenarios.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))
    print(f"Wrote calibration scenarios to {calibration_path}")

    if args.selected_global_cap is not None:
        main_config = build_main_config(
            prompt_sets,
            per_layer_position_topk=args.per_layer_position_topk,
            global_cap=args.selected_global_cap,
            include_exact_subset=args.include_exact_subset,
            include_robustness=args.include_robustness,
        )
        main_path = args.output_dir / "sparsification_main_scenarios.json"
        main_path.write_text(json.dumps(main_config, indent=2))
        print(f"Wrote main scenarios to {main_path}")


if __name__ == "__main__":
    main()
