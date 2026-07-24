from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import DEFAULT_SCRATCH_ROOT, REPO_ROOT
from .io_utils import read_json, write_json
from .scenarios.chpc_baseline import build_chpc_baseline_config


DEFAULT_OUTPUT_ROOT = (
    DEFAULT_SCRATCH_ROOT / "granite" / "sweep" / "performance_optimization"
)
DEFAULT_BASELINE_REGISTRY = (
    REPO_ROOT
    / "experiments"
    / "baselines"
    / "exact_trace_performance_granite_20260709.json"
)
RUNNER = REPO_ROOT / "experiments" / "run_sparsification_experiment.py"
SIBLING_ROOT = REPO_ROOT.parent / "circuit-tracer_chunked"

METRIC_KEYS = {
    "feature_jaccard": "worst_step_feature_jaccard",
    "edge_jaccard": "worst_step_all_edge_jaccard",
    "top256_edge_jaccard": "worst_step_all_edge_top256_jaccard",
    "weighted_edge_jaccard": "worst_step_all_edge_weighted_jaccard",
    "target_token_match": "worst_step_target_token_match",
    "edge_magnitude_l1_deviation": (
        "worst_step_all_edge_normalized_l1_deviation"
    ),
}
FIDELITY_THRESHOLDS = {
    "bounded": {
        "worst_step_feature_jaccard_min": 0.98,
        "worst_step_all_edge_jaccard_min": 0.98,
        "worst_step_all_edge_top256_jaccard_min": 0.98,
        "worst_step_all_edge_weighted_jaccard_min": 0.98,
        "worst_step_target_token_match_min": 1.0,
        "worst_step_all_edge_normalized_l1_deviation_max": 0.02,
    },
    "exact": {
        "worst_step_feature_jaccard_min": 1.0,
        "worst_step_all_edge_jaccard_min": 1.0,
        "worst_step_all_edge_top256_jaccard_min": 1.0,
        "worst_step_all_edge_weighted_jaccard_min": 0.999999,
        "worst_step_target_token_match_min": 1.0,
        "worst_step_all_edge_normalized_l1_deviation_max": 0.000001,
    },
}


@dataclass(frozen=True)
class Case:
    variant: str
    fixture: str

    @property
    def key(self) -> str:
        return f"performance/{self.variant}/{self.fixture}"


SUITES: dict[str, tuple[Case, ...]] = {
    "clt-smoke": (Case("gemma3_1b_clt", "361_base"),),
    "clt-pair": (
        Case("gemma3_1b_clt", "828_base"),
        Case("gemma3_1b_clt", "361_base"),
    ),
    "plt-hard": (Case("gemma3_1b_plt", "361_base"),),
    "all": (
        Case("gemma3_1b_clt", "828_base"),
        Case("gemma3_1b_clt", "361_base"),
        Case("gemma3_1b_plt", "361_base"),
    ),
}


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def workspace_state(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "dirty_files": _git_output(root, "status", "--short").splitlines(),
        "source_content_sha256": source_content_sha256(root),
    }


def source_content_sha256(root: Path) -> str:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = sorted(
        Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in completed.stdout.split(b"\0")
        if raw
    )
    digest = hashlib.sha256()
    for relative_path in paths:
        digest.update(os.fsencode(str(relative_path)))
        digest.update(b"\0")
        path = root / relative_path
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing-or-nonfile\0")
        digest.update(b"\0")
    return digest.hexdigest()


def capture_source_state() -> dict[str, Any]:
    return {
        "project": workspace_state(REPO_ROOT),
        "sibling": workspace_state(SIBLING_ROOT),
    }


def gpu_provenance(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_rows = [
        {
            "name": fields[0],
            "uuid": fields[1],
            "driver_version": fields[2],
            "memory_total_mib": int(fields[3]),
        }
        for line in completed.stdout.splitlines()
        if line.strip()
        for fields in ([field.strip() for field in line.split(",", maxsplit=3)],)
        if len(fields) == 4
    ]
    return {
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "slurm_job_name": env.get("SLURM_JOB_NAME"),
        "slurm_cluster_name": env.get("SLURM_CLUSTER_NAME"),
        "slurm_job_partition": env.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_returncode": completed.returncode,
        "gpus": gpu_rows,
    }


def assert_h200_allocation(
    provenance: dict[str, Any],
    *,
    allow_test_environment: bool,
) -> None:
    if allow_test_environment:
        return
    if not provenance.get("slurm_job_id"):
        raise RuntimeError(
            "exact-trace-perf run requires an active SLURM allocation; "
            "use --dry-run outside one"
        )
    gpus = provenance.get("gpus") or []
    if len(gpus) != 1:
        raise RuntimeError(
            f"exact-trace-perf requires exactly one visible GPU, found {len(gpus)}"
        )
    if "H200" not in str(gpus[0].get("name", "")).upper():
        names = ", ".join(str(gpu.get("name")) for gpu in gpus)
        raise RuntimeError(f"exact-trace-perf requires H200 GPU(s), found: {names}")
    if int(gpus[0].get("memory_total_mib") or 0) < 130_000:
        raise RuntimeError("exact-trace-perf requires one full, non-MIG H200")
    if provenance.get("slurm_cluster_name") != "granite":
        raise RuntimeError("exact-trace-perf timing requires SLURM cluster granite")
    if provenance.get("slurm_job_partition") != "rai-gpu-grn":
        raise RuntimeError(
            "exact-trace-perf timing requires SLURM partition rai-gpu-grn"
        )


def execution_evidence(
    provenance: dict[str, Any],
    *,
    allow_test_environment: bool,
) -> dict[str, Any]:
    return {
        "test_environment_override_used": allow_test_environment,
        "timing_evidence_eligible": not allow_test_environment,
        "timing_evidence_status": (
            "test_override_non_evidence"
            if allow_test_environment
            else "h200_allocation"
        ),
        "observed_gpu_names": [
            str(gpu.get("name")) for gpu in provenance.get("gpus") or []
        ],
    }


def _case_scenario(case: Case, fidelity: str) -> dict[str, Any]:
    payload = build_chpc_baseline_config(variant=case.variant, cluster="granite")
    scenario = next(
        row for row in payload["scenarios"] if row["fixture_name"] == case.fixture
    )
    scenario = dict(scenario)
    scenario["name"] = f"perf_{case.variant}_{case.fixture}"
    scenario["stage"] = "exact_trace_performance_optimization"
    scenario["tier"] = "sweep"
    scenario["resource_profile"] = "performance_optimization_h200"
    scenario["baseline_check"] = {
        "enabled": True,
        "mode": "gate",
        "registry_key": case.key,
        "baseline_required": True,
        "thresholds": FIDELITY_THRESHOLDS[fidelity],
    }
    return {"defaults": payload["defaults"], "scenarios": [scenario]}


def _runner_command(
    *,
    scenario_file: Path,
    output_root: Path,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--scenarios-file",
        str(scenario_file),
        "--output-root",
        str(output_root),
        "--baseline-registry",
        str(DEFAULT_BASELINE_REGISTRY),
        "--fail-on-baseline-missing",
        "--fail-on-validation-fail",
        "--run-id",
        run_id,
        "--run-name",
        "Exact-trace performance optimization",
        "--run-goal",
        "Improve exact-trace runtime without exceeding the selected parity budget.",
    ]


def _stream_runner(command: Sequence[str], *, output_root: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)
    offsets: dict[Path, int] = {}
    while process.poll() is None:
        _tail_logs(output_root, offsets)
        time.sleep(1)
    _tail_logs(output_root, offsets)
    return int(process.returncode or 0)


def _tail_logs(output_root: Path, offsets: dict[Path, int]) -> None:
    for path in sorted(output_root.glob("*/run.log")):
        offset = offsets.get(path, 0)
        with path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                print(f"[{path.parent.name}] {line}", end="", flush=True)
            offsets[path] = handle.tell()


def _baseline_entries() -> dict[str, dict[str, Any]]:
    registry = read_json(DEFAULT_BASELINE_REGISTRY)
    entries = registry.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid performance baseline registry: {DEFAULT_BASELINE_REGISTRY}")
    return entries


def _result_report(
    case: Case,
    *,
    candidate_root: Path,
    baseline_entry: dict[str, Any],
    runner_returncode: int = 0,
) -> dict[str, Any]:
    scenario_root = candidate_root / f"perf_{case.variant}_{case.fixture}"
    result = read_json(scenario_root / "result.json")
    comparison_path = scenario_root / "baseline_compare.json"
    comparison = read_json(comparison_path) if comparison_path.is_file() else {}
    candidate_duration_raw = result.get("duration_seconds")
    candidate_duration = (
        float(candidate_duration_raw)
        if isinstance(candidate_duration_raw, (int, float))
        else None
    )
    baseline_duration = float(baseline_entry["duration_seconds"])
    metrics = {
        label: comparison.get(metric_key)
        for label, metric_key in METRIC_KEYS.items()
    }
    return {
        "case": case.key,
        "status": result.get("status"),
        "runner_returncode": runner_returncode,
        "candidate_duration_seconds": candidate_duration,
        "baseline_duration_seconds": baseline_duration,
        "speedup": (
            baseline_duration / candidate_duration
            if candidate_duration is not None and candidate_duration > 0
            else None
        ),
        **metrics,
        "worst_step_evidence": comparison.get("worst_step_evidence", {}),
        "passed": result.get("baseline_check", {}).get("passed") is True,
        "failure_reasons": result.get("baseline_check", {}).get(
            "failure_reasons", []
        ),
        "scenario_root": str(scenario_root),
    }


def _print_report(report: dict[str, Any]) -> None:
    def render(value: Any, digits: int) -> str:
        return f"{float(value):.{digits}f}" if value is not None else "n/a"

    print(
        f"{report['case']}: "
        f"candidate={render(report.get('candidate_duration_seconds'), 2)}s "
        f"baseline={render(report.get('baseline_duration_seconds'), 2)}s "
        f"speedup={render(report.get('speedup'), 3)}x "
        f"feature={render(report.get('feature_jaccard'), 6)} "
        f"all_edge={render(report.get('edge_jaccard'), 6)} "
        f"all_top256={render(report.get('top256_edge_jaccard'), 6)} "
        f"all_weighted={render(report.get('weighted_edge_jaccard'), 6)} "
        f"edge_magnitude_l1={render(report.get('edge_magnitude_l1_deviation'), 6)} "
        f"token={render(report.get('target_token_match'), 0)} "
        f"gate={'PASS' if report['passed'] else 'FAIL'}"
    )


def _run(args: argparse.Namespace) -> int:
    cases = SUITES[args.suite]
    run_id = args.run_id or time.strftime("perf-%Y%m%d-%H%M%S")
    run_root = args.output_root / run_id
    if args.dry_run:
        for case in cases:
            scenario_file = run_root / "configs" / f"{case.variant}_{case.fixture}.json"
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            print(f"{case.key}: {shlex.join(_runner_command(scenario_file=scenario_file, output_root=candidate_root, run_id=run_id))}")
        return 0

    provenance = gpu_provenance()
    assert_h200_allocation(
        provenance,
        allow_test_environment=args.allow_non_h200_test_only,
    )
    source_state = capture_source_state()
    evidence = execution_evidence(
        provenance,
        allow_test_environment=args.allow_non_h200_test_only,
    )
    baseline_entries = _baseline_entries()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "configs").mkdir()
    (run_root / "candidates").mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "suite": args.suite,
        "fidelity": args.fidelity,
        "thresholds": FIDELITY_THRESHOLDS[args.fidelity],
        "baseline_registry": str(DEFAULT_BASELINE_REGISTRY),
        "baseline_registry_id": read_json(DEFAULT_BASELINE_REGISTRY).get(
            "registry_id"
        ),
        "baseline_references": {
            case.key: baseline_entries[case.key] for case in cases
        },
        "workspace_mode": "live",
        "live_workspace_rationale": (
            "The isolated optimization worktree is the candidate under test; "
            "the CLI freezes and verifies both repository states for each case."
        ),
        "no_edits_during_run_enforced": True,
        "source_state_before": source_state,
        "execution_provenance": provenance,
        "execution_evidence": evidence,
        "governor_state_policy": "read_only_no_promotion_or_default_mutation",
        "cases": [case.key for case in cases],
    }
    write_json(run_root / "run_manifest.json", manifest)

    reports: list[dict[str, Any]] = []
    write_json(
        run_root / "performance_report.json",
        {
            **manifest,
            "status": "running",
            "passed": False,
            "reports": reports,
        },
    )
    current_case: str | None = None
    try:
        for case in cases:
            current_case = case.key
            before_case = capture_source_state()
            if before_case != source_state:
                raise RuntimeError(
                    f"Source state changed before {case.key}; refusing mixed-source run"
                )
            scenario_file = run_root / "configs" / f"{case.variant}_{case.fixture}.json"
            write_json(scenario_file, _case_scenario(case, args.fidelity))
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            command = _runner_command(
                scenario_file=scenario_file,
                output_root=candidate_root,
                run_id=run_id,
            )
            print(f"Running {case.key}: {shlex.join(command)}", flush=True)
            returncode = _stream_runner(command, output_root=candidate_root)
            after_case = capture_source_state()
            if after_case != source_state:
                raise RuntimeError(
                    f"Source state changed during {case.key}; result is not comparable"
                )
            scenario_root = candidate_root / f"perf_{case.variant}_{case.fixture}"
            has_result = (scenario_root / "result.json").is_file()
            if returncode != 0 and not has_result:
                raise RuntimeError(
                    f"Existing sparsification runner failed for {case.key} "
                    f"with exit code {returncode}"
                )
            report = _result_report(
                case,
                candidate_root=candidate_root,
                baseline_entry=baseline_entries[case.key],
                runner_returncode=returncode,
            )
            reports.append(report)
            _print_report(report)
            write_json(
                run_root / "performance_report.json",
                {
                    **manifest,
                    "status": "running",
                    "passed": False,
                    "reports": reports,
                },
            )
            if returncode != 0 and report["passed"]:
                raise RuntimeError(
                    f"Existing sparsification runner exited {returncode} after a "
                    f"passing comparison for {case.key}"
                )
    except Exception as error:
        write_json(
            run_root / "performance_report.json",
            {
                **manifest,
                "status": "failed",
                "passed": False,
                "current_case": current_case,
                "error": f"{type(error).__name__}: {error}",
                "source_state_after": capture_source_state(),
                "reports": reports,
            },
        )
        raise

    summary = {
        **manifest,
        "status": "completed",
        "source_state_after": capture_source_state(),
        "passed": all(
            report["passed"] and report["runner_returncode"] == 0
            for report in reports
        ),
        "reports": reports,
    }
    write_json(run_root / "performance_report.json", summary)
    print(f"Report: {run_root / 'performance_report.json'}")
    return 0 if summary["passed"] else 1


def _list_suites() -> int:
    for name, cases in SUITES.items():
        rendered = ", ".join(case.key for case in cases)
        print(f"{name}: {rendered}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="H200 exact-trace performance and graph-parity loop"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List fixed performance suites")
    run = subparsers.add_parser("run", help="Run one suite and enforce parity")
    run.add_argument("suite", choices=tuple(SUITES))
    run.add_argument(
        "--fidelity",
        choices=tuple(FIDELITY_THRESHOLDS),
        default="bounded",
    )
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--allow-non-h200-test-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _list_suites()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
