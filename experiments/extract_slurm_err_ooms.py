from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from extract_utils import ensure_dir, read_json, safe_stem, write_csv


DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_BENCHMARK_ROOT = Path("/fs/scratch/PAS3272/kopanev.1/weekend_exact_chunked")
DEFAULT_OUTPUT_DIR = Path("experiments/extracted/weekend_exact_chunked")

JOB_ID_RE = re.compile(r"Job ID: (?P<job_id>\d+)")
ARRAY_TASK_RE = re.compile(r"Array task: (?P<array_task>\S+)")
NODE_RE = re.compile(r"Node: (?P<node>.+)")
CLUSTER_RE = re.compile(r"Cluster: (?P<cluster>.+)")
SCENARIOS_FILE_RE = re.compile(r"Scenarios file: (?P<scenarios_file>.+)")
OUTPUT_ROOT_RE = re.compile(r"Output root: (?P<output_root>.+)")
WRITING_DIR_RE = re.compile(r"Writing experiment results to (?P<scenario_root>.+)")
RUNNING_SCENARIO_RE = re.compile(r"Running scenario: (?P<scenario_name>.+)")
OOM_KILL_RE = re.compile(r"Detected (?P<count>\d+) oom_kill events")
CUDA_OOM_RE = re.compile(r"torch\.OutOfMemoryError: CUDA out of memory")
TIMEOUT_RE = re.compile(r"time limit|timed out", re.IGNORECASE)


def _classify_err_text(text: str) -> tuple[str | None, int | None]:
    oom_kill_match = OOM_KILL_RE.search(text)
    if oom_kill_match:
        return "ram_oom", int(oom_kill_match.group("count"))
    if "Exceeded step memory limit" in text:
        return "ram_oom", None
    if "OOM Killed" in text or "oom_kill" in text:
        return "ram_oom", None
    if CUDA_OOM_RE.search(text):
        return "cuda_oom", None
    if TIMEOUT_RE.search(text):
        return "timeout", None
    if text.strip():
        return "other_error", None
    return None, None


def _parse_out_metadata(out_path: Path) -> dict[str, Any]:
    if not out_path.exists():
        return {}

    metadata: dict[str, Any] = {"out_file": str(out_path)}
    for raw_line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for regex, key in [
            (JOB_ID_RE, "job_id"),
            (ARRAY_TASK_RE, "array_task"),
            (NODE_RE, "node"),
            (CLUSTER_RE, "cluster"),
            (SCENARIOS_FILE_RE, "scenarios_file"),
            (OUTPUT_ROOT_RE, "output_root"),
            (WRITING_DIR_RE, "scenario_root"),
            (RUNNING_SCENARIO_RE, "scenario_name"),
        ]:
            match = regex.search(line)
            if match:
                metadata[key] = match.group(key)
    return metadata


def build_row(err_path: Path, benchmark_root: Path) -> dict[str, Any]:
    err_text = err_path.read_text(encoding="utf-8", errors="replace")
    failure_family, oom_kill_count = _classify_err_text(err_text)
    out_path = err_path.with_suffix(".out")
    out_metadata = _parse_out_metadata(out_path)
    scenario_root = out_metadata.get("scenario_root")
    result_path = Path(scenario_root) / "result.json" if scenario_root else None
    result_status = None
    if result_path is not None and result_path.exists():
        result_status = read_json(result_path).get("status")

    return {
        "err_file": str(err_path),
        "out_file": out_metadata.get("out_file"),
        "slurm_stem": safe_stem(err_path),
        "job_id": out_metadata.get("job_id"),
        "array_task": out_metadata.get("array_task"),
        "node": out_metadata.get("node"),
        "cluster": out_metadata.get("cluster"),
        "scenarios_file": out_metadata.get("scenarios_file"),
        "output_root": out_metadata.get("output_root"),
        "scenario_root": scenario_root,
        "scenario_name": out_metadata.get("scenario_name"),
        "failure_family": failure_family,
        "oom_kill_event_count": oom_kill_count,
        "err_line_count": len(err_text.splitlines()),
        "err_nonempty": bool(err_text.strip()),
        "matches_benchmark_root": bool(
            scenario_root and str(scenario_root).startswith(str(benchmark_root))
        ),
        "result_json_exists": result_path.exists()
        if result_path is not None
        else False,
        "result_status": result_status,
        "err_excerpt": err_text.strip()[:500] if err_text.strip() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SLURM .err/.out files for RAM OOM and other job-level failures"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory containing SLURM .err/.out files",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Benchmark root used to tag relevant weekend-exact rows",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted CSV files will be written",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    rows = [
        build_row(path, args.benchmark_root)
        for path in sorted(args.logs_dir.glob("*.err"))
    ]
    write_csv(
        args.output_dir / "slurm_err_summary.csv",
        rows,
        preferred_headers=[
            "err_file",
            "out_file",
            "job_id",
            "array_task",
            "node",
            "cluster",
            "scenarios_file",
            "output_root",
            "scenario_root",
            "scenario_name",
            "failure_family",
            "oom_kill_event_count",
            "matches_benchmark_root",
            "result_json_exists",
            "result_status",
            "err_excerpt",
        ],
    )
    print(
        f"Wrote {len(rows)} SLURM err rows to {args.output_dir / 'slurm_err_summary.csv'}"
    )


if __name__ == "__main__":
    main()
