from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated")
DEFAULT_FIXTURE_CATALOG = (
    DEFAULT_OUTPUT_DIR / "weekend_exact_chunked_fixtures" / "fixture_catalog.json"
)
DEFAULT_SELECTION_TEMPLATE = Path(__file__).with_name(
    "weekend_exact_chunked_followup_selection.template.json"
)

BASE_PROMPT_INDICES = [828, 94, 361]
LATE_FIXTURE_NAMES = ["828_late", "94_late", "361_late"]
STRESS_FIXTURE_NAMES = ["361_base", "361_late"]

CLUSTER_CONFIGS: dict[str, dict[str, Any]] = {
    "ascend": {
        "wave1": [
            (128, 2048),
            (192, 2048),
            (256, 2048),
            (128, 4096),
            (192, 4096),
            (256, 4096),
            (128, 8192),
            (192, 8192),
        ],
        "wave1_optional": {(192, 8192)},
        "cache_budgets_gib": [0, 8, 12, 16],
        "wave1_walltime": "02:30:00",
        "wave2_walltime": "01:30:00",
        "validation_walltime": "02:00:00",
    },
    "cardinal": {
        "wave1": [
            (128, 4096),
            (256, 4096),
            (384, 4096),
            (128, 8192),
            (256, 8192),
            (384, 8192),
            (256, 16384),
            (384, 16384),
        ],
        "wave1_optional": {(384, 16384)},
        "cache_budgets_gib": [0, 8, 16, 24, 32],
        "wave1_walltime": "02:00:00",
        "wave2_walltime": "01:00:00",
        "validation_walltime": "01:30:00",
    },
}


def base_defaults() -> dict[str, Any]:
    return {
        "completions": 1,
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
        "max_steps": 1,
        "method": "exact",
    }


def slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def load_fixture_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {fixture["fixture_name"]: fixture for fixture in data["fixtures"]}


def fixture_scenario_source(
    fixture_name: str,
    fixture_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixture = fixture_catalog[fixture_name]
    return {
        "fixture_name": fixture_name,
        "fixture_kind": fixture.get("fixture_kind"),
        "prepared_prompt_file": fixture["prepared_prompt_file"],
        "prepared_prompt_meta_file": fixture["prepared_prompt_meta_file"],
    }


def build_wave1_config(cluster: str) -> dict[str, Any]:
    cluster_config = CLUSTER_CONFIGS[cluster]
    scenarios: list[dict[str, Any]] = []
    for gsm8k_index in BASE_PROMPT_INDICES:
        for attribution_batch_size, decoder_chunk_size in cluster_config["wave1"]:
            optional = (attribution_batch_size, decoder_chunk_size) in cluster_config[
                "wave1_optional"
            ]
            scenario = {
                "name": (
                    f"{cluster}_wave1_p{gsm8k_index:04d}"
                    f"_b{attribution_batch_size:03d}"
                    f"_c{decoder_chunk_size}"
                    "_cache0"
                ),
                "stage": f"weekend_exact_chunked_wave1_{cluster}",
                "cluster": cluster,
                "fixture_name": f"{gsm8k_index}_base",
                "fixture_kind": "base",
                "gsm8k_indices": [gsm8k_index],
                "attribution_batch_size": attribution_batch_size,
                "feature_batch_size": attribution_batch_size,
                "logit_batch_size": attribution_batch_size,
                "decoder_chunk_size": decoder_chunk_size,
                "cross_batch_decoder_cache_bytes": 0,
                "optional_probe": optional,
            }
            scenarios.append(scenario)

    return {
        "defaults": base_defaults(),
        "metadata": {
            "cluster": cluster,
            "stage": f"weekend_exact_chunked_wave1_{cluster}",
            "recommended_walltime": cluster_config["wave1_walltime"],
            "base_prompt_indices": BASE_PROMPT_INDICES,
            "notes": [
                "Wave 1 no-cache screen on base prompts only.",
                "Primary factorial rule: feature_batch_size = logit_batch_size = attribution_batch_size.",
                "Each scenario traces exactly one prompt/completion for one step.",
            ],
        },
        "scenarios": scenarios,
    }


def build_wave2_config(
    cluster: str,
    *,
    selection: dict[str, Any],
    fixture_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cluster_config = CLUSTER_CONFIGS[cluster]
    shortlist = selection.get("wave2_shortlist", [])
    scenarios: list[dict[str, Any]] = []

    for shortlist_entry in shortlist:
        attribution_batch_size = int(shortlist_entry["attribution_batch_size"])
        decoder_chunk_size = int(shortlist_entry["decoder_chunk_size"])
        label = slugify(
            shortlist_entry.get(
                "name",
                f"b{attribution_batch_size}_c{decoder_chunk_size}",
            )
        )
        for fixture_name in STRESS_FIXTURE_NAMES:
            fixture_source = fixture_scenario_source(fixture_name, fixture_catalog)
            for cache_budget_gib in cluster_config["cache_budgets_gib"]:
                cache_budget_bytes = cache_budget_gib * (1024**3)
                scenarios.append(
                    {
                        "name": (
                            f"{cluster}_wave2_{label}_{fixture_name}"
                            f"_b{attribution_batch_size:03d}"
                            f"_c{decoder_chunk_size}"
                            f"_cache{cache_budget_gib}g"
                        ),
                        "stage": f"weekend_exact_chunked_wave2_{cluster}",
                        "cluster": cluster,
                        **fixture_source,
                        "shortlist_label": label,
                        "attribution_batch_size": attribution_batch_size,
                        "feature_batch_size": attribution_batch_size,
                        "logit_batch_size": attribution_batch_size,
                        "decoder_chunk_size": decoder_chunk_size,
                        "cross_batch_decoder_cache_bytes": cache_budget_bytes,
                    }
                )

    return {
        "defaults": base_defaults(),
        "metadata": {
            "cluster": cluster,
            "stage": f"weekend_exact_chunked_wave2_{cluster}",
            "recommended_walltime": cluster_config["wave2_walltime"],
            "stress_fixture_names": STRESS_FIXTURE_NAMES,
            "cache_budgets_gib": cluster_config["cache_budgets_gib"],
            "notes": [
                "Wave 2 cache sweep on 361_base and 361_late only.",
                "Populate wave2_shortlist in the follow-up selection file after reviewing wave 1 results.",
            ],
        },
        "scenarios": scenarios,
    }


def build_validation_config(
    cluster: str,
    *,
    selection: dict[str, Any],
    fixture_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cluster_config = CLUSTER_CONFIGS[cluster]
    scenarios: list[dict[str, Any]] = []

    for shortlist_entry in selection.get("validation_no_cache", []):
        attribution_batch_size = int(shortlist_entry["attribution_batch_size"])
        decoder_chunk_size = int(shortlist_entry["decoder_chunk_size"])
        label = slugify(
            shortlist_entry.get(
                "name",
                f"b{attribution_batch_size}_c{decoder_chunk_size}_nocache",
            )
        )
        for fixture_name in LATE_FIXTURE_NAMES:
            scenarios.append(
                {
                    "name": (
                        f"{cluster}_validation_{label}_{fixture_name}"
                        f"_b{attribution_batch_size:03d}"
                        f"_c{decoder_chunk_size}"
                        "_cache0"
                    ),
                    "stage": f"weekend_exact_chunked_validation_{cluster}",
                    "cluster": cluster,
                    **fixture_scenario_source(fixture_name, fixture_catalog),
                    "shortlist_label": label,
                    "validation_group": "no_cache",
                    "attribution_batch_size": attribution_batch_size,
                    "feature_batch_size": attribution_batch_size,
                    "logit_batch_size": attribution_batch_size,
                    "decoder_chunk_size": decoder_chunk_size,
                    "cross_batch_decoder_cache_bytes": 0,
                }
            )

    for shortlist_entry in selection.get("validation_cache", []):
        attribution_batch_size = int(shortlist_entry["attribution_batch_size"])
        decoder_chunk_size = int(shortlist_entry["decoder_chunk_size"])
        cache_budget_gib = int(shortlist_entry["cache_budget_gib"])
        label = slugify(
            shortlist_entry.get(
                "name",
                f"b{attribution_batch_size}_c{decoder_chunk_size}_cache{cache_budget_gib}g",
            )
        )
        fixture_name = "361_late"
        scenarios.append(
            {
                "name": (
                    f"{cluster}_validation_{label}_{fixture_name}"
                    f"_b{attribution_batch_size:03d}"
                    f"_c{decoder_chunk_size}"
                    f"_cache{cache_budget_gib}g"
                ),
                "stage": f"weekend_exact_chunked_validation_{cluster}",
                "cluster": cluster,
                **fixture_scenario_source(fixture_name, fixture_catalog),
                "shortlist_label": label,
                "validation_group": "cache",
                "attribution_batch_size": attribution_batch_size,
                "feature_batch_size": attribution_batch_size,
                "logit_batch_size": attribution_batch_size,
                "decoder_chunk_size": decoder_chunk_size,
                "cross_batch_decoder_cache_bytes": cache_budget_gib * (1024**3),
            }
        )

    return {
        "defaults": base_defaults(),
        "metadata": {
            "cluster": cluster,
            "stage": f"weekend_exact_chunked_validation_{cluster}",
            "recommended_walltime": cluster_config["validation_walltime"],
            "late_fixture_names": LATE_FIXTURE_NAMES,
            "notes": [
                "Broader late-prefix validation on shortlisted no-cache configs.",
                "Cache validation runs only 361_late so prompt-length effects stay separable from cache effects.",
            ],
        },
        "scenarios": scenarios,
    }


def write_config(output_dir: Path, file_name: str, payload: dict[str, Any]) -> None:
    path = output_dir / file_name
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scenario files for the weekend exact chunked benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated scenario files will be written",
    )
    parser.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_FIXTURE_CATALOG,
        help="Fixture catalog produced by prepare_weekend_prefix_fixtures.py",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional follow-up selection JSON for wave 2 and validation configs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cluster in CLUSTER_CONFIGS:
        write_config(
            args.output_dir,
            f"weekend_exact_chunked_wave1_{cluster}_scenarios.json",
            build_wave1_config(cluster),
        )

    if args.selection_file is None:
        print(
            "Skipping wave 2 / validation generation because no selection file was provided."
        )
        return

    selection_path = args.selection_file
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection file not found: {selection_path}. Start from {DEFAULT_SELECTION_TEMPLATE}."
        )

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            f"Fixture catalog not found: {args.fixture_catalog}. Run prepare_weekend_prefix_fixtures.py first."
        )

    selection = json.loads(selection_path.read_text())
    fixture_catalog = load_fixture_catalog(args.fixture_catalog)

    for cluster in CLUSTER_CONFIGS:
        cluster_selection = selection.get(cluster, {})

        wave2_shortlist = cluster_selection.get("wave2_shortlist", [])
        if wave2_shortlist:
            write_config(
                args.output_dir,
                f"weekend_exact_chunked_wave2_{cluster}_scenarios.json",
                build_wave2_config(
                    cluster,
                    selection=cluster_selection,
                    fixture_catalog=fixture_catalog,
                ),
            )

        validation_no_cache = cluster_selection.get("validation_no_cache", [])
        validation_cache = cluster_selection.get("validation_cache", [])
        if validation_no_cache or validation_cache:
            write_config(
                args.output_dir,
                f"weekend_exact_chunked_validation_{cluster}_scenarios.json",
                build_validation_config(
                    cluster,
                    selection=cluster_selection,
                    fixture_catalog=fixture_catalog,
                ),
            )


if __name__ == "__main__":
    main()
