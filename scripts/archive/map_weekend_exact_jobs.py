from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"

LOG_NAME_RE = re.compile(
    r"slurm-(?P<job_name>.+)-(?P<array_id>\d+)_(?P<array_task>\d+)\.out$"
)


@dataclass
class JobMapping:
    parent_array_id: int
    child_job_id: int | None
    array_task: int
    job_name: str
    cluster: str | None
    scenario_file: str | None
    scenario_name: str | None
    output_root: str | None
    output_dir: str | None
    status: str | None
    duration_seconds: float | None
    out_log: str
    err_log: str | None


def parse_log_name(path: Path) -> tuple[str, int, int] | None:
    match = LOG_NAME_RE.match(path.name)
    if match is None:
        return None
    return (
        match.group("job_name"),
        int(match.group("array_id")),
        int(match.group("array_task")),
    )


def extract_value(lines: list[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def load_scenario_name(
    scenario_file_value: str | None,
    *,
    array_task: int,
) -> str | None:
    if scenario_file_value is None:
        return None
    scenario_path = Path(scenario_file_value)
    if not scenario_path.is_absolute():
        scenario_path = REPO_ROOT / scenario_path
    if not scenario_path.exists():
        return None
    payload = json.loads(scenario_path.read_text())
    scenarios = payload.get("scenarios", [])
    if not (0 <= array_task < len(scenarios)):
        return None
    scenario = scenarios[array_task]
    if not isinstance(scenario, dict):
        return None
    name = scenario.get("name")
    return str(name) if name is not None else None


def infer_status(lines: list[str]) -> tuple[str | None, float | None]:
    for line in reversed(lines):
        if line.startswith("Completed "):
            match = re.search(r"status=([A-Za-z_]+) duration=([0-9.]+)s", line)
            if match is not None:
                return match.group(1), float(match.group(2))
            return "completed", None
    return None, None


def build_mapping(out_log: Path) -> JobMapping:
    parsed_name = parse_log_name(out_log)
    if parsed_name is None:
        raise ValueError(f"Unrecognized log filename format: {out_log.name}")
    job_name, parent_array_id, array_task = parsed_name

    lines = out_log.read_text(errors="replace").splitlines()
    child_job_id_raw = extract_value(lines, "Job ID: ")
    child_job_id = int(child_job_id_raw) if child_job_id_raw is not None else None
    cluster = extract_value(lines, "Cluster: ")
    scenario_file = extract_value(lines, "Scenarios file: ")
    output_root = extract_value(lines, "Output root: ")
    scenario_name = extract_value(lines, "Running scenario: ")
    if scenario_name is None:
        scenario_name = load_scenario_name(scenario_file, array_task=array_task)
    status, duration_seconds = infer_status(lines)

    output_dir = None
    if output_root is not None and scenario_name is not None:
        output_dir = str(Path(output_root) / scenario_name)

    err_log = out_log.with_suffix(".err")
    return JobMapping(
        parent_array_id=parent_array_id,
        child_job_id=child_job_id,
        array_task=array_task,
        job_name=job_name,
        cluster=cluster,
        scenario_file=scenario_file,
        scenario_name=scenario_name,
        output_root=output_root,
        output_dir=output_dir,
        status=status,
        duration_seconds=duration_seconds,
        out_log=str(out_log),
        err_log=str(err_log) if err_log.exists() else None,
    )


def find_logs_for_array(array_id: int) -> list[Path]:
    matches = []
    for path in LOG_DIR.glob("*.out"):
        parsed_name = parse_log_name(path)
        if parsed_name is None:
            continue
        _, parent_array_id, _ = parsed_name
        if parent_array_id == array_id:
            matches.append(path)

    def array_task_key(path: Path) -> int:
        parsed_name = parse_log_name(path)
        return parsed_name[2] if parsed_name is not None else -1

    return sorted(matches, key=array_task_key)


def find_logs_for_child_job(job_id: int) -> list[Path]:
    matches = []
    needle = f"Job ID: {job_id}"
    for path in LOG_DIR.glob("*.out"):
        text = path.read_text(errors="replace")
        if needle in text:
            matches.append(path)
    return sorted(matches)


def format_table(mappings: list[JobMapping]) -> str:
    headers = [
        "array_id",
        "child_id",
        "task",
        "cluster",
        "job_name",
        "status",
        "scenario",
    ]
    rows = [
        [
            str(mapping.parent_array_id),
            "?" if mapping.child_job_id is None else str(mapping.child_job_id),
            str(mapping.array_task),
            mapping.cluster or "?",
            mapping.job_name,
            mapping.status or "?",
            mapping.scenario_name or "?",
        ]
        for mapping in mappings
    ]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    output_lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        output_lines.append(
            "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        )
    return "\n".join(output_lines)


def print_details(mappings: list[JobMapping]) -> None:
    print(format_table(mappings))
    print()
    for mapping in mappings:
        print(f"[{mapping.array_task}] {mapping.scenario_name or '?'}")
        print(f"  parent array id: {mapping.parent_array_id}")
        print(f"  child job id: {mapping.child_job_id}")
        print(f"  cluster: {mapping.cluster}")
        print(f"  scenario file: {mapping.scenario_file}")
        print(f"  output root: {mapping.output_root}")
        print(f"  output dir: {mapping.output_dir}")
        print(f"  status: {mapping.status}")
        print(f"  duration_seconds: {mapping.duration_seconds}")
        print(f"  out log: {mapping.out_log}")
        print(f"  err log: {mapping.err_log}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map weekend exact benchmark array/job IDs to scenarios and log files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--array-id",
        type=int,
        help="Parent SLURM array id from submission time (maps the whole array)",
    )
    group.add_argument(
        "--job-id",
        type=int,
        help="Child SLURM job id shown in OSC Active Jobs / Grafana",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table",
    )
    args = parser.parse_args()

    if args.array_id is not None:
        logs = find_logs_for_array(args.array_id)
        if not logs:
            raise SystemExit(f"No .out logs found for array id {args.array_id}")
    else:
        logs = find_logs_for_child_job(args.job_id)
        if not logs:
            raise SystemExit(
                f"No .out logs found containing child job id {args.job_id}"
            )

    mappings = [build_mapping(path) for path in logs]
    if args.json:
        print(json.dumps([asdict(mapping) for mapping in mappings], indent=2))
        return

    print_details(mappings)


if __name__ == "__main__":
    main()
