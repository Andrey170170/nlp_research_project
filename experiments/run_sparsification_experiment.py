from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = (
    Path(__file__).with_name("generated") / "sparsification_calibration_scenarios.json"
)
DEFAULT_OUTPUT_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/sparsification_experiment")


def build_command(output_dir: Path, scenario: dict[str, Any]) -> list[str]:
    method = scenario["method"]
    script_name = (
        "trace_pipeline.py" if method == "old_patch" else "trace_pipeline_chunked.py"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / script_name),
        "--prompts",
        str(len(scenario["gsm8k_indices"])),
        "--gsm8k-indices",
        ",".join(str(i) for i in scenario["gsm8k_indices"]),
        "--completions",
        str(scenario["completions"]),
        "--temperature",
        str(scenario["temperature"]),
        "--output-dir",
        str(output_dir),
        "--max-feature-nodes",
        str(scenario["max_feature_nodes"]),
        "--max-edges",
        str(scenario["max_edges"]),
        "--max-steps",
        str(scenario["max_steps"]),
        "--attribution-batch-size",
        str(scenario["attribution_batch_size"]),
        "--max-n-logits",
        str(scenario["max_n_logits"]),
        "--desired-logit-prob",
        str(scenario["desired_logit_prob"]),
        "--attribution-update-interval",
        str(scenario["attribution_update_interval"]),
    ]

    if method != "old_patch":
        cmd.extend(["--decoder-chunk-size", str(scenario["decoder_chunk_size"])])
        if scenario.get("sparsify_per_layer_position_topk") is not None:
            cmd.extend(
                [
                    "--sparsify-per-layer-position-topk",
                    str(scenario["sparsify_per_layer_position_topk"]),
                ]
            )
        if scenario.get("sparsify_global_cap") is not None:
            cmd.extend(["--sparsify-global-cap", str(scenario["sparsify_global_cap"])])

    if scenario.get("verbose_attribution", False):
        cmd.append("--verbose-attribution")
    if scenario.get("profile_attribution", False):
        cmd.append("--profile-attribution")
    if "profile_log_interval" in scenario:
        cmd.extend(["--profile-log-interval", str(scenario["profile_log_interval"])])
    if scenario.get("diagnostic_feature_cap") is not None:
        cmd.extend(
            ["--diagnostic-feature-cap", str(scenario["diagnostic_feature_cap"])]
        )
    if scenario.get("save_raw", False):
        cmd.append("--save-raw")
    if scenario.get("no_offload", False):
        cmd.append("--no-offload")
    if scenario.get("no_lazy_encoder", False) and method != "old_patch":
        cmd.append("--no-lazy-encoder")
    if scenario.get("no_lazy_decoder", False) and method != "old_patch":
        cmd.append("--no-lazy-decoder")

    return cmd


def run_scenario(
    output_root: Path,
    scenario: dict[str, Any],
    *,
    env: dict[str, str],
) -> dict[str, Any]:
    scenario_name = scenario["name"]
    scenario_root = (
        output_root
        if output_root.name == scenario_name
        else output_root / scenario_name
    )
    run_output_dir = scenario_root / "artifacts"
    scenario_root.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (scenario_root / "scenario.json").write_text(json.dumps(scenario, indent=2))

    log_path = scenario_root / "run.log"
    cmd = build_command(run_output_dir, scenario)
    timeout_minutes = scenario.get("timeout_minutes")
    timeout_seconds = None if timeout_minutes is None else int(timeout_minutes * 60)

    result: dict[str, Any] = {
        "name": scenario_name,
        "stage": scenario.get("stage"),
        "method": scenario["method"],
        "gsm8k_indices": scenario["gsm8k_indices"],
        "command": cmd,
        "output_dir": str(run_output_dir),
        "status": "unknown",
    }
    if timeout_minutes is not None:
        result["timeout_minutes"] = timeout_minutes

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Scenario: {scenario_name}\n")
        log_file.write(f"Stage: {scenario.get('stage')}\n")
        log_file.write(f"Method: {scenario['method']}\n")
        log_file.write(f"Start: {time.ctime(start)}\n")
        log_file.write(f"Working directory: {REPO_ROOT}\n")
        log_file.write(f"Command: {shlex.join(cmd)}\n\n")
        log_file.flush()

        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["returncode"] = None
            log_file.write(f"\nTimed out after {timeout_minutes} minutes.\n")
        else:
            result["returncode"] = completed.returncode
            result["status"] = "ok" if completed.returncode == 0 else "failed"

    result["duration_seconds"] = round(time.time() - start, 2)
    result["log_path"] = str(log_path)
    (scenario_root / "result.json").write_text(json.dumps(result, indent=2))
    return result


def load_scenarios(scenarios_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = json.loads(scenarios_file.read_text())
    defaults = config.get("defaults", {})
    scenarios = [defaults | scenario for scenario in config["scenarios"]]
    return scenarios, config.get("metadata", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibration/main scenarios for the sparsification experiment"
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="JSON file describing experiment scenarios",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for scenario logs, artifacts, and summaries",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help="Run only the scenario at this 0-based index",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Run only the scenario with this name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print scenario indices and names without running anything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them",
    )
    args = parser.parse_args()

    scenarios, metadata = load_scenarios(args.scenarios_file)

    if args.list:
        for idx, scenario in enumerate(scenarios):
            print(
                f"[{idx:03d}] {scenario['name']} | stage={scenario.get('stage')} | method={scenario['method']}"
            )
        return

    if args.scenario_index is not None and args.scenario_name is not None:
        raise ValueError("Use only one of --scenario-index or --scenario-name")

    if args.scenario_index is not None:
        scenarios = [scenarios[args.scenario_index]]
    elif args.scenario_name is not None:
        scenarios = [
            scenario for scenario in scenarios if scenario["name"] == args.scenario_name
        ]
        if not scenarios:
            raise ValueError(f"Scenario not found: {args.scenario_name}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root / timestamp
        if len(scenarios) > 1
        else args.output_root / scenarios[0]["name"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"Writing experiment results to {output_root}")
    print(f"Loaded {len(scenarios)} scenario(s) from {args.scenarios_file}")
    if metadata:
        print(f"Scenario metadata: {json.dumps(metadata, indent=2)}")

    results = []
    for scenario in scenarios:
        name = scenario["name"]
        scenario_root = output_root if len(scenarios) == 1 else output_root / name
        if args.dry_run:
            cmd = build_command(scenario_root / "artifacts", scenario)
            print(f"DRY RUN {name}: {shlex.join(cmd)}")
            continue
        print(f"\n{'=' * 80}\nRunning scenario: {name}\n{'=' * 80}")
        results.append(run_scenario(output_root, scenario, env=env))
        print(
            f"Completed {name}: status={results[-1]['status']} duration={results[-1]['duration_seconds']:.2f}s"
        )

    if args.dry_run:
        return

    summary = {
        "timestamp": timestamp,
        "scenarios_file": str(args.scenarios_file),
        "metadata": metadata,
        "output_root": str(output_root),
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
