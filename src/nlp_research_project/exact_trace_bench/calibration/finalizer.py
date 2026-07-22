"""Scheduler-aware calibration finalization independent of task-runner survival."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..calibration_observations import (
    build_calibration_observation,
    write_calibration_observation,
)
from ..io_utils import read_json, write_json

SACCT_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "MaxRSS",
    "MaxVMSize",
    "AllocTRES",
    "ReqMem",
    "Start",
    "End",
    "NodeList",
    "Reason",
)
TERMINAL_SUCCESS = frozenset({"COMPLETED"})
TERMINAL_CENSORED = frozenset({"OUT_OF_MEMORY", "TIMEOUT"})
TERMINAL_FAILURE = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "PREEMPTED",
        "REVOKED",
    }
)


@dataclass(frozen=True)
class SchedulerAccounting:
    job_id: str
    state: str
    exit_code: str | None = None
    elapsed_seconds: int | None = None
    max_rss: str | None = None
    max_vm_size: str | None = None
    allocated_tres: str | None = None
    requested_memory: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    node_list: str | None = None
    reason: str | None = None

    @property
    def classification(self) -> str:
        state = self.state.split(None, 1)[0].split("+", 1)[0]
        if state in TERMINAL_SUCCESS:
            return "success"
        if state in TERMINAL_CENSORED:
            return "censored"
        if state in TERMINAL_FAILURE:
            return "failure"
        return "nonterminal_or_unknown"

    @property
    def terminal(self) -> bool:
        return self.classification != "nonterminal_or_unknown"

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "classification": self.classification,
            "terminal": self.terminal,
        }


SacctRunner = Callable[[str], str]


def run_sacct(job_id: str) -> str:
    completed = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "-j",
            job_id,
            "--format",
            ",".join(SACCT_FIELDS),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def parse_sacct(text: str) -> tuple[SchedulerAccounting, ...]:
    records: list[SchedulerAccounting] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) < len(SACCT_FIELDS):
            raise ValueError(
                f"sacct row has {len(values)} fields; expected {len(SACCT_FIELDS)}"
            )
        row = dict(zip(SACCT_FIELDS, values, strict=False))
        elapsed = row["ElapsedRaw"]
        records.append(
            SchedulerAccounting(
                job_id=row["JobIDRaw"],
                state=row["State"],
                exit_code=row["ExitCode"] or None,
                elapsed_seconds=int(elapsed) if elapsed.isdigit() else None,
                max_rss=row["MaxRSS"] or None,
                max_vm_size=row["MaxVMSize"] or None,
                allocated_tres=row["AllocTRES"] or None,
                requested_memory=row["ReqMem"] or None,
                started_at=row["Start"] or None,
                ended_at=row["End"] or None,
                node_list=row["NodeList"] or None,
                reason=row["Reason"] or None,
            )
        )
    return tuple(records)


def _synthetic_result(root: Path, accounting: SchedulerAccounting) -> dict[str, Any]:
    status = {
        "success": "success",
        "censored": "timeout" if accounting.state.startswith("TIMEOUT") else "oom",
        "failure": "failed",
    }.get(accounting.classification, "unknown")
    return {
        "status": status,
        "returncode": None,
        "duration_seconds": accounting.elapsed_seconds,
        "output_dir": str(root / "artifacts"),
        "result_origin": "scheduler_finalizer",
    }


def _resource_sidecars(root: Path, job_id: str) -> tuple[Path, ...]:
    monitor_root = root.parent / "_slurm_gpu_monitor" / job_id
    if not monitor_root.is_dir():
        return ()
    return tuple(sorted(monitor_root.glob("*.gpulog")))


def finalize_observations(
    roots: Iterable[Path],
    *,
    job_id: str,
    sacct_runner: SacctRunner = run_sacct,
    sacct_parser: Callable[[str], tuple[SchedulerAccounting, ...]] = parse_sacct,
    baseline_entries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    accounting_rows = sacct_parser(sacct_runner(job_id))
    by_id = {row.job_id: row for row in accounting_rows}
    fallback = by_id.get(job_id) or (
        accounting_rows[0] if len(accounting_rows) == 1 else None
    )
    if fallback is None:
        raise ValueError(f"no unambiguous sacct allocation record for {job_id}")
    if not fallback.terminal:
        raise ValueError(f"job {fallback.job_id} is not terminal: {fallback.state}")
    step_rows = tuple(
        row for row in accounting_rows if row.job_id.startswith(f"{fallback.job_id}.")
    )
    scheduler_accounting = {
        **fallback.to_json(),
        "steps": [row.to_json() for row in step_rows],
    }

    finalized: list[str] = []
    unsupported: list[dict[str, str]] = []
    for root in roots:
        scenario_path = root / "scenario.json"
        if not scenario_path.is_file():
            unsupported.append({"root": str(root), "reason": "missing_scenario_json"})
            continue
        scenario = read_json(scenario_path)
        campaign = scenario.get("calibration_campaign")
        reference = campaign.get("reference") if isinstance(campaign, Mapping) else None
        reference_key = (
            reference.get("registry_key") if isinstance(reference, Mapping) else None
        )
        existing_path = root / "calibration_observation.json"
        existing = read_json(existing_path) if existing_path.is_file() else {}
        existing_provenance = existing.get("provenance")
        existing_reference = (
            existing_provenance.get("reference")
            if isinstance(existing_provenance, Mapping)
            else None
        )
        baseline_entry = (baseline_entries or {}).get(str(reference_key))
        if baseline_entry is None and isinstance(existing_reference, Mapping):
            baseline_entry = existing_reference
        result_path = root / "result.json"
        result = (
            read_json(result_path)
            if result_path.is_file()
            else _synthetic_result(root, fallback)
        )
        if not result_path.is_file():
            write_json(result_path, result)
        finalization = {
            "kind": "scheduler_afterany",
            "source_job_id": fallback.job_id,
            "source_state": fallback.state,
            "regenerated_after_runner_loss": result.get("result_origin")
            == "scheduler_finalizer",
            "finalized_at": fallback.ended_at,
        }
        payload = build_calibration_observation(
            scenario_root=root,
            scenario=scenario,
            result=result,
            baseline_entry=baseline_entry,
            resource_sidecar_paths=_resource_sidecars(root, fallback.job_id),
            scheduler_accounting=scheduler_accounting,
            finalization=finalization,
        )
        if payload is None:
            unsupported.append(
                {"root": str(root), "reason": "not_calibration_campaign"}
            )
            continue
        path = write_calibration_observation(root, payload)
        finalized.append(str(path))
    return {
        "job_id": job_id,
        "accounting": [row.to_json() for row in accounting_rows],
        "finalized": finalized,
        "unsupported": unsupported,
    }
