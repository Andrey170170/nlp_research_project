from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.run_sparsification_experiment import build_command as build_campaign_command


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = Path(__file__).with_name("chunked_trace_scenarios.json")
DEFAULT_OUTPUT_ROOT = Path("/fs/scratch/PAS2836/kopanev.1/trace_chunked_test_matrix")


def build_command(output_dir: Path, scenario: dict[str, Any]) -> list[str]:
    return build_campaign_command(output_dir, scenario)


def run_scenario(
    output_root: Path,
    scenario: dict[str, Any],
    *,
    env: dict[str, str],
) -> dict[str, Any]:
    scenario_name = scenario["name"]
    scenario_root = output_root / scenario_name
    run_output_dir = scenario_root / "artifacts"
    scenario_root.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    (scenario_root / "scenario.json").write_text(json.dumps(scenario, indent=2))

    log_path = scenario_root / "run.log"
    cmd = build_command(run_output_dir, scenario)
    timeout_seconds = int(scenario["timeout_minutes"] * 60)

    result: dict[str, Any] = {
        "name": scenario_name,
        "command": cmd,
        "output_dir": str(run_output_dir),
        "timeout_minutes": scenario["timeout_minutes"],
        "status": "unknown",
    }

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Scenario: {scenario_name}\n")
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
            log_file.write(
                f"\nTimed out after {scenario['timeout_minutes']} minutes.\n"
            )
        else:
            result["returncode"] = completed.returncode
            result["status"] = "ok" if completed.returncode == 0 else "failed"

    end = time.time()
    result["duration_seconds"] = round(end - start, 2)
    result["log_path"] = str(log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Ascend-friendly matrix of chunked tracing smoke tests"
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="JSON file describing the scenario matrix",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for logs, artifacts, and summary output",
    )
    args = parser.parse_args()

    config = json.loads(args.scenarios_file.read_text())
    defaults = config.get("defaults", {})
    scenarios = [defaults | scenario for scenario in config["scenarios"]]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"Writing matrix results to {output_root}")
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios_file}")

    results = []
    for scenario in scenarios:
        name = scenario["name"]
        print(f"\n{'=' * 80}")
        print(f"Running scenario: {name}")
        print(f"{'=' * 80}")
        results.append(run_scenario(output_root, scenario, env=env))

        status = results[-1]["status"]
        duration = results[-1]["duration_seconds"]
        print(f"Completed {name}: status={status} duration={duration:.2f}s")

    summary = {
        "timestamp": timestamp,
        "scenarios_file": str(args.scenarios_file),
        "output_root": str(output_root),
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nSummary written to {summary_path}")
    for result in results:
        print(
            f"- {result['name']}: {result['status']} "
            f"({result['duration_seconds']:.2f}s) log={result['log_path']}"
        )


if __name__ == "__main__":
    main()
