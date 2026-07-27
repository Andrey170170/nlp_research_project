from __future__ import annotations

import argparse
import hashlib
import os
import resource
import shlex
import signal
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
    "edge_magnitude_l1_deviation": ("worst_step_all_edge_normalized_l1_deviation"),
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
DEFAULT_RUN_GOAL = (
    "Improve exact-trace runtime without exceeding the selected parity budget."
)
PERFORMANCE_TARGET_SECONDS = {
    "performance/gemma3_1b_plt/361_base": 600.0,
}
PERFORMANCE_STRETCH_TARGET_SECONDS = {
    "performance/gemma3_1b_plt/361_base": 300.0,
}
ACTIVE_ROW_INITIAL_PREDICTED_DURATION_SECONDS = (65.0, 90.0)
ACTIVE_ROW_FUSED_TARGET_SECONDS = 245.0
ACTIVE_ROW_EXPECTED_BYTES = 66_158 * 1152 * 2
ACTIVE_ROW_FRAMEBUFFER_REFERENCE = {
    "run_id": "perf-plt-streaming-baseline-c65536-20260724-03",
    "resource_summary_path": (
        "/scratch/general/vast/u1653998/nlp_research_project/exact_trace_bench/"
        "granite/sweep/performance_optimization/"
        "perf-plt-streaming-baseline-c65536-20260724-03/candidates/"
        "gemma3_1b_plt_361_base/resource_summary.json"
    ),
    "gpu_framebuffer_peak_mib": 26113.0,
}
_PLT_BOUNDED_TAPE_V1 = {
    "decoder_chunk_size": 65536,
    "nnsight_session_capacity": 256,
    "phase1_trace_batch_policy": "cap_effective_batches",
    "phase1_trace_batch_size_max": 128,
    "phase3_compute_microbatch_max_rows": 128,
    "phase4_execution_batch_max_rows": 256,
    "feature_vjp_tape_batch_window": 2,
    "feature_vjp_tape_max_bytes": 12 * 1024**3,
}
CANDIDATE_PROFILES: dict[str, dict[str, Any]] = {
    "canonical": {},
    "clt-phase1-cap128-v1": {
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "nnsight_session_capacity": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 128,
    },
    "plt-bounded-fast-v1": {
        "decoder_chunk_size": 32768,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-fast-v2": {
        "decoder_chunk_size": 65536,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-fast-v3": {
        "decoder_chunk_size": 65536,
        "cross_batch_decoder_cache_bytes": 16 * 1024**3,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
    },
    "plt-bounded-tape-v1": {
        **_PLT_BOUNDED_TAPE_V1,
    },
    "plt-bounded-tape-prefetch-v1": {
        **_PLT_BOUNDED_TAPE_V1,
        "decoder_page_prefetch_depth": 1,
    },
    "plt-bounded-frontier-v1": {
        "decoder_chunk_size": 65536,
        "nnsight_session_capacity": 512,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 512,
    },
    "plt-active-rows-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 1,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    },
    "plt-active-rows-tape-v1": {
        "decoder_chunk_size": 4096,
        "cross_batch_decoder_cache_bytes": 0,
        "nnsight_session_capacity": 256,
        "phase1_trace_batch_policy": "cap_effective_batches",
        "phase1_trace_batch_size_max": 128,
        "phase3_compute_microbatch_max_rows": 128,
        "phase4_execution_batch_max_rows": 256,
        "feature_vjp_tape_batch_window": 2,
        "feature_vjp_tape_max_bytes": 12 * 1024**3,
        "decoder_page_prefetch_depth": 0,
        "decoder_active_row_residency": True,
        "decoder_active_row_max_bytes": 1024**3,
    },
}
BOUNDED_ONLY_CANDIDATE_PROFILES = frozenset(
    {
        "plt-bounded-fast-v1",
        "plt-bounded-fast-v2",
        "plt-bounded-fast-v3",
        "plt-bounded-tape-v1",
        "plt-bounded-tape-prefetch-v1",
        "plt-bounded-frontier-v1",
    }
)
GPU_SAMPLE_COMMAND = (
    "nvidia-smi",
    (
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,"
        "memory.used,memory.total"
    ),
    "--format=csv,noheader,nounits",
    "--loop=1",
)


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


def _candidate_overrides(case: Case, candidate_profile: str) -> dict[str, Any]:
    if candidate_profile == "clt-phase1-cap128-v1":
        return (
            dict(CANDIDATE_PROFILES[candidate_profile])
            if case.variant == "gemma3_1b_clt"
            else {}
        )
    if case.variant != "gemma3_1b_plt":
        return {}
    return dict(CANDIDATE_PROFILES[candidate_profile])


def _case_scenario(
    case: Case,
    fidelity: str,
    candidate_profile: str = "canonical",
) -> dict[str, Any]:
    payload = build_chpc_baseline_config(variant=case.variant, cluster="granite")
    scenario = next(
        row for row in payload["scenarios"] if row["fixture_name"] == case.fixture
    )
    scenario = dict(scenario)
    scenario["name"] = f"perf_{case.variant}_{case.fixture}"
    scenario["stage"] = "exact_trace_performance_optimization"
    scenario["tier"] = "sweep"
    scenario["resource_profile"] = "performance_optimization_h200"
    scenario.update(_candidate_overrides(case, candidate_profile))
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
    run_goal: str = DEFAULT_RUN_GOAL,
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
        run_goal,
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _gpu_resource_summary(samples_path: Path) -> dict[str, Any]:
    if not samples_path.is_file():
        return {"gpu_sample_count": 0}
    rows: list[tuple[float, float, float, float, float]] = []
    for line in samples_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        rows.append((values[0], values[1], values[2], values[3], values[4]))
    if not rows:
        return {"gpu_sample_count": 0}
    sm, memory, power, framebuffer, total = (list(column) for column in zip(*rows))
    return {
        "gpu_sample_count": len(rows),
        "gpu_sm_utilization_mean_percent": sum(sm) / len(sm),
        "gpu_sm_utilization_p95_percent": _percentile(sm, 0.95),
        "gpu_sm_utilization_max_percent": max(sm),
        "gpu_memory_utilization_mean_percent": sum(memory) / len(memory),
        "gpu_memory_utilization_p95_percent": _percentile(memory, 0.95),
        "gpu_memory_utilization_max_percent": max(memory),
        "gpu_power_mean_watts": sum(power) / len(power),
        "gpu_power_max_watts": max(power),
        "gpu_framebuffer_peak_mib": max(framebuffer),
        "gpu_framebuffer_total_mib": max(total),
        "gpu_framebuffer_peak_fraction": max(framebuffer) / max(total),
    }


def _stop_process(
    process: subprocess.Popen[Any] | None,
    *,
    process_group: bool = False,
) -> None:
    if process is None:
        return
    if process_group:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    else:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process_group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
    else:
        if process_group:
            # The wrapper can exit before its descendants. Ensure nothing remains
            # in the isolated runner group after the graceful group signal.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _stream_runner(command: Sequence[str], *, output_root: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    output_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / "gpu_samples.csv"
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    samples_handle = samples_path.open("w", encoding="utf-8")
    sampler: subprocess.Popen[Any] | None = None
    process: subprocess.Popen[Any] | None = None
    runner_completed = False
    sampler_was_running = False
    offsets: dict[Path, int] = {}
    try:
        sampler = subprocess.Popen(
            GPU_SAMPLE_COMMAND,
            cwd=REPO_ROOT,
            env=env,
            stdout=samples_handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
        while process.poll() is None:
            _tail_logs(output_root, offsets)
            time.sleep(1)
        _tail_logs(output_root, offsets)
        runner_completed = True
    finally:
        if not runner_completed:
            _stop_process(process, process_group=True)
        sampler_was_running = sampler is not None and sampler.poll() is None
        _stop_process(sampler)
        samples_handle.close()
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall_seconds = time.perf_counter() - start
        cpu_seconds = (
            usage_after.ru_utime
            + usage_after.ru_stime
            - usage_before.ru_utime
            - usage_before.ru_stime
        )
        allocated_cpus = int(
            env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE") or 1
        )
        gpu_summary = _gpu_resource_summary(samples_path)
        gpu_sample_count = int(gpu_summary["gpu_sample_count"])
        if sampler is None:
            sampling_status = "failed_to_start"
            sampling_failure_reason = "GPU sampler did not start"
        elif not sampler_was_running:
            sampling_status = "exited_early"
            sampling_failure_reason = (
                "GPU sampler exited before harness cleanup; "
                f"returncode={sampler.returncode}"
            )
        elif gpu_sample_count == 0:
            sampling_status = "no_samples"
            sampling_failure_reason = "GPU sampler produced no usable samples"
        else:
            sampling_status = "ok"
            sampling_failure_reason = None
        resource_summary = {
            **gpu_summary,
            "sample_file": str(samples_path),
            "gpu_sampling_status": sampling_status,
            "gpu_sampling_failure_reason": sampling_failure_reason,
            "gpu_sampler_returncode": (
                sampler.returncode if sampler is not None else None
            ),
            "resource_validation_passed": sampling_status == "ok",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "allocated_cpus": allocated_cpus,
            "cpu_one_core_utilization_percent": (
                100.0 * cpu_seconds / wall_seconds if wall_seconds > 0 else None
            ),
            "cpu_allocated_utilization_percent": (
                100.0 * cpu_seconds / wall_seconds / allocated_cpus
                if wall_seconds > 0 and allocated_cpus > 0
                else None
            ),
        }
        write_json(output_root / "resource_summary.json", resource_summary)
    if process is None:
        raise RuntimeError("Trace runner did not start")
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
        raise ValueError(
            f"Invalid performance baseline registry: {DEFAULT_BASELINE_REGISTRY}"
        )
    return entries


def _active_row_mechanism_gate(
    requested: bool,
    diagnostics: Any,
    max_bytes: int,
) -> tuple[bool | None, list[str]]:
    if not requested:
        return None, []
    if not isinstance(diagnostics, dict):
        return False, ["active-row diagnostics missing from result artifact summary"]

    reasons: list[str] = []
    resident = diagnostics.get("resident")
    build = diagnostics.get("build")
    phase4 = diagnostics.get("phase4")
    if diagnostics.get("requested") is not True:
        reasons.append("active-row runtime did not record requested=true")
    if diagnostics.get("effective") is not True:
        reasons.append("active-row residency was not effective")
    fallback = diagnostics.get("fallback_reason")
    if fallback is not None:
        reasons.append(f"active-row fallback was recorded: {fallback}")
    if not all(isinstance(value, dict) for value in (resident, build, phase4)):
        reasons.append("active-row resident/build/phase4 diagnostics are incomplete")
        return False, reasons

    assert isinstance(resident, dict)
    assert isinstance(build, dict)
    assert isinstance(phase4, dict)
    seed = diagnostics.get("seed")
    if not isinstance(seed, dict):
        reasons.append("active-row fused-seed diagnostics are incomplete")
        return False, reasons
    resident_bytes = resident.get("bytes")
    estimated_bytes = resident.get("estimated_bytes")
    if not isinstance(resident_bytes, int) or resident_bytes <= 0:
        reasons.append("active-row resident bytes must be positive")
    if not isinstance(estimated_bytes, int) or estimated_bytes <= 0:
        reasons.append("active-row estimated bytes must be positive")
    if (
        isinstance(resident_bytes, int)
        and isinstance(estimated_bytes, int)
        and resident_bytes != estimated_bytes
    ):
        reasons.append("active-row resident bytes differ from admitted estimate")
    if (
        not isinstance(max_bytes, int)
        or max_bytes <= 0
        or not isinstance(resident_bytes, int)
        or resident_bytes > max_bytes
    ):
        reasons.append(
            "active-row resident bytes exceed or lack the configured byte cap"
        )
    if isinstance(resident_bytes, int) and not (
        ACTIVE_ROW_EXPECTED_BYTES / 10
        <= resident_bytes
        <= ACTIVE_ROW_EXPECTED_BYTES * 10
    ):
        reasons.append(
            "active-row resident bytes are outside the predicted order of magnitude"
        )
    if not isinstance(resident.get("row_count"), int) or resident["row_count"] <= 0:
        reasons.append("active-row resident row count must be positive")
    if resident.get("owner_count") != 1:
        reasons.append("active-row owner count must equal 1")
    if build.get("count") != 1:
        reasons.append("active-row build count must equal 1")
    if build.get("source") != "phase0_fused_seed":
        reasons.append("active-row build source must equal phase0_fused_seed")
    traversal = build.get("traversal_bytes")
    loaded = build.get("decoder_load_bytes")
    if (
        not isinstance(traversal, int)
        or traversal != 0
        or not isinstance(loaded, int)
        or loaded != 0
    ):
        reasons.append(
            "fused active-row materialization must add zero traversal/load bytes"
        )
    if (
        not isinstance(build.get("decoder_page_load_count"), int)
        or build["decoder_page_load_count"] != 0
    ):
        reasons.append("fused active-row materialization must add zero decoder page loads")
    shared_traversal = seed.get("shared_traversal_bytes")
    shared_loaded = seed.get("shared_decoder_load_bytes")
    if (
        not isinstance(shared_traversal, int)
        or shared_traversal <= 0
        or not isinstance(shared_loaded, int)
        or shared_loaded <= 0
        or shared_traversal != shared_loaded
    ):
        reasons.append(
            "fused active-row seed shared traversal/load bytes are missing or inconsistent"
        )
    if (
        not isinstance(seed.get("shared_decoder_page_load_count"), int)
        or seed["shared_decoder_page_load_count"] <= 0
    ):
        reasons.append("fused active-row seed shared decoder page-load count must be positive")
    seed_bytes = seed.get("bytes")
    if (
        not isinstance(seed_bytes, int)
        or seed_bytes <= 0
        or not isinstance(resident_bytes, int)
        or seed_bytes > resident_bytes
    ):
        reasons.append("active-row seed bytes must be positive and no larger than resident bytes")
    unique_row_count = seed.get("unique_row_count")
    resident_row_count = resident.get("row_count")
    if (
        not isinstance(unique_row_count, int)
        or unique_row_count <= 0
        or not isinstance(resident_row_count, int)
        or unique_row_count > resident_row_count
    ):
        reasons.append(
            "active-row seed unique-row count must be positive and no larger than resident rows"
        )
    if seed.get("materialization_h2d_bytes") != resident_bytes:
        reasons.append("active-row seed H2D bytes must equal resident bytes")
    if seed.get("missing_keys") != 0:
        reasons.append("active-row seed must cover every final decoder-row key")
    if not isinstance(resident.get("device"), str) or not resident["device"].startswith(
        "cuda"
    ):
        reasons.append("active-row residency device must be CUDA")
    if (
        phase4.get("decoder_page_load_count_delta") != 0
        or phase4.get("decoder_load_bytes_delta") != 0
    ):
        reasons.append(
            "Phase4 decoder page-load and load-byte deltas must both equal zero"
        )
    return not reasons, reasons


def _active_row_framebuffer_gate(
    requested: bool,
    resource_summary: dict[str, Any],
) -> tuple[bool | None, dict[str, Any], list[str]]:
    reference = dict(ACTIVE_ROW_FRAMEBUFFER_REFERENCE)
    if not requested:
        return None, {"status": "not_applicable", "reference": reference}, []
    candidate = resource_summary.get("gpu_framebuffer_peak_mib")
    limit = reference["gpu_framebuffer_peak_mib"]
    comparison = {
        "status": "unavailable",
        "candidate_peak_mib": candidate,
        "reference_peak_mib": limit,
        "reference": reference,
    }
    if not isinstance(candidate, (int, float)):
        return (
            False,
            comparison,
            ["active-row framebuffer comparison unavailable: candidate peak missing"],
        )
    passed = float(candidate) <= float(limit)
    comparison["status"] = "passed" if passed else "exceeded_reference"
    reasons = (
        []
        if passed
        else [
            f"candidate peak framebuffer {float(candidate):.1f} MiB exceeds "
            f"audited reference {float(limit):.1f} MiB"
        ]
    )
    return passed, comparison, reasons


def _result_report(
    case: Case,
    *,
    candidate_root: Path,
    baseline_entry: dict[str, Any],
    runner_returncode: int = 0,
) -> dict[str, Any]:
    scenario_root = candidate_root / f"perf_{case.variant}_{case.fixture}"
    result = read_json(scenario_root / "result.json")
    scenario_path = scenario_root / "scenario.json"
    scenario = read_json(scenario_path) if scenario_path.is_file() else {}
    active_rows_requested = scenario.get("decoder_active_row_residency") is True
    active_rows_max_bytes_raw = scenario.get("decoder_active_row_max_bytes", 0)
    active_rows_max_bytes = (
        int(active_rows_max_bytes_raw)
        if isinstance(active_rows_max_bytes_raw, int)
        else 0
    )
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
        label: comparison.get(metric_key) for label, metric_key in METRIC_KEYS.items()
    }
    parity_passed = result.get("baseline_check", {}).get("passed") is True
    parity_failure_reasons = result.get("baseline_check", {}).get("failure_reasons", [])
    performance_failure_reasons: list[str] = []
    reconciliation_required = False
    if active_rows_requested:
        performance_target = ACTIVE_ROW_FUSED_TARGET_SECONDS
        stretch_target = None
        performance_passed = bool(
            candidate_duration is not None
            and candidate_duration <= performance_target
        )
        stretch_passed = None
        reconciliation_required = not performance_passed
        if candidate_duration is None:
            performance_failure_reasons.append(
                "active-row duration is missing; the fused engineering target cannot be evaluated"
            )
        elif not performance_passed:
            performance_failure_reasons.append(
                f"active-row duration {candidate_duration:.2f}s exceeds the reconciled "
                f"fused engineering target {performance_target:.2f}s; "
                "reconciliation is required"
            )
    else:
        performance_target = PERFORMANCE_TARGET_SECONDS.get(case.key)
        stretch_target = PERFORMANCE_STRETCH_TARGET_SECONDS.get(case.key)
        performance_passed = (
            candidate_duration <= performance_target
            if performance_target is not None and candidate_duration is not None
            else None
        )
        if performance_target is not None and candidate_duration is None:
            performance_passed = False
        stretch_passed = (
            candidate_duration <= stretch_target
            if stretch_target is not None and candidate_duration is not None
            else None
        )
        if stretch_target is not None and candidate_duration is None:
            stretch_passed = False
        if performance_passed is False:
            if candidate_duration is None:
                performance_failure_reasons.append(
                    f"performance target {performance_target:.2f}s could not be "
                    "evaluated because candidate duration is missing"
                )
            else:
                performance_failure_reasons.append(
                    f"candidate duration {candidate_duration:.2f}s exceeds "
                    f"performance target {performance_target:.2f}s"
                )
    resource_summary = (
        read_json(candidate_root / "resource_summary.json")
        if (candidate_root / "resource_summary.json").is_file()
        else {}
    )
    resource_validation_passed = (
        resource_summary.get("resource_validation_passed") is True
    )
    resource_failure_reasons = (
        []
        if resource_validation_passed
        else [
            str(
                resource_summary.get("gpu_sampling_failure_reason")
                or "GPU resource sampling was not validated"
            )
        ]
    )
    diagnostics = (result.get("artifact_summary") or {}).get(
        "decoder_active_row_residency"
    )
    mechanism_passed, mechanism_failure_reasons = _active_row_mechanism_gate(
        active_rows_requested,
        diagnostics,
        active_rows_max_bytes,
    )
    framebuffer_passed, framebuffer_comparison, framebuffer_failure_reasons = (
        _active_row_framebuffer_gate(active_rows_requested, resource_summary)
    )
    resource_failure_reasons.extend(framebuffer_failure_reasons)
    resource_gate_passed = resource_validation_passed and (
        framebuffer_passed is not False
    )
    passed = (
        parity_passed
        and performance_passed is not False
        and resource_gate_passed
        and mechanism_passed is not False
        and not reconciliation_required
        and runner_returncode == 0
    )
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
        "profiling_summary": result.get("profiling_summary") or {},
        "resource_summary": resource_summary,
        "performance_target_seconds": performance_target,
        "performance_stretch_target_seconds": stretch_target,
        "active_row_initial_predicted_duration_seconds": (
            list(ACTIVE_ROW_INITIAL_PREDICTED_DURATION_SECONDS)
            if active_rows_requested
            else None
        ),
        "active_row_fused_target_seconds": (
            ACTIVE_ROW_FUSED_TARGET_SECONDS if active_rows_requested else None
        ),
        "reconciliation_required": reconciliation_required,
        "decoder_active_row_residency": diagnostics,
        "mechanism_validation_passed": mechanism_passed,
        "mechanism_validation_status": "not_applicable"
        if mechanism_passed is None
        else ("passed" if mechanism_passed else "failed"),
        "framebuffer_comparison": framebuffer_comparison,
        "framebuffer_passed": framebuffer_passed,
        "parity_passed": parity_passed,
        "performance_passed": performance_passed,
        "performance_stretch_passed": stretch_passed,
        "resource_validation_passed": resource_validation_passed,
        "resource_gate_passed": resource_gate_passed,
        "passed": passed,
        "parity_failure_reasons": parity_failure_reasons,
        "performance_failure_reasons": performance_failure_reasons,
        "mechanism_failure_reasons": mechanism_failure_reasons,
        "resource_failure_reasons": resource_failure_reasons,
        "failure_reasons": [
            *parity_failure_reasons,
            *performance_failure_reasons,
            *mechanism_failure_reasons,
            *resource_failure_reasons,
        ],
        "scenario_root": str(scenario_root),
    }


def _print_report(report: dict[str, Any]) -> None:
    def render(value: Any, digits: int) -> str:
        return f"{float(value):.{digits}f}" if value is not None else "n/a"

    profiling = report.get("profiling_summary") or {}
    phase_timings = (
        f"phase3={render(profiling.get('phase3_duration_seconds'), 2)}s "
        f"phase4={render(profiling.get('phase4_duration_seconds'), 2)}s "
        f"phase4_batch={render(profiling.get('phase4_avg_batch_seconds'), 2)}s "
        f"batches={profiling.get('phase4_batches_observed', 'n/a')}/"
        f"{profiling.get('phase4_total_batches', 'n/a')}"
    )
    target = report.get("performance_target_seconds")
    stretch_target = report.get("performance_stretch_target_seconds")
    performance_gate = report.get("performance_passed")
    performance_status = (
        "n/a" if performance_gate is None else ("PASS" if performance_gate else "FAIL")
    )
    resource_status = "PASS" if report.get("resource_gate_passed") else "FAIL"
    mechanism_gate = report.get("mechanism_validation_passed")
    mechanism_status = (
        "n/a" if mechanism_gate is None else ("PASS" if mechanism_gate else "FAIL")
    )
    reconciliation_status = (
        "REQUIRED" if report.get("reconciliation_required") else "OK"
    )
    print(
        f"{report['case']}: "
        f"candidate={render(report.get('candidate_duration_seconds'), 2)}s "
        f"baseline={render(report.get('baseline_duration_seconds'), 2)}s "
        f"speedup={render(report.get('speedup'), 3)}x "
        f"target={render(target, 2)}s "
        f"stretch={render(stretch_target, 2)}s "
        f"{phase_timings} "
        f"feature={render(report.get('feature_jaccard'), 6)} "
        f"all_edge={render(report.get('edge_jaccard'), 6)} "
        f"all_top256={render(report.get('top256_edge_jaccard'), 6)} "
        f"all_weighted={render(report.get('weighted_edge_jaccard'), 6)} "
        f"edge_magnitude_l1={render(report.get('edge_magnitude_l1_deviation'), 6)} "
        f"token={render(report.get('target_token_match'), 0)} "
        f"parity={'PASS' if report.get('parity_passed') else 'FAIL'} "
        f"performance={performance_status} "
        f"stretch={'PASS' if report.get('performance_stretch_passed') else 'MISS'} "
        f"resource={resource_status} "
        f"mechanism={mechanism_status} "
        f"reconciliation={reconciliation_status} "
        f"gate={'PASS' if report['passed'] else 'FAIL'}"
    )


def _gate_summary(reports: Sequence[dict[str, Any]]) -> dict[str, bool | None]:
    parity_passed = all(report["parity_passed"] for report in reports)
    performance_results = [
        report["performance_passed"]
        for report in reports
        if report["performance_passed"] is not None
    ]
    performance_passed = all(performance_results) if performance_results else None
    resource_gate_passed = bool(reports) and all(
        report.get("resource_gate_passed") is True for report in reports
    )
    mechanism_results = [
        report.get("mechanism_validation_passed")
        for report in reports
        if report.get("mechanism_validation_passed") is not None
    ]
    mechanism_validation_passed = all(mechanism_results) if mechanism_results else None
    reconciliation_required = any(
        report.get("reconciliation_required") is True for report in reports
    )
    passed = (
        bool(reports)
        and parity_passed
        and performance_passed is not False
        and resource_gate_passed
        and mechanism_validation_passed is not False
        and not reconciliation_required
        and all(report["runner_returncode"] == 0 for report in reports)
    )
    return {
        "parity_passed": parity_passed,
        "performance_passed": performance_passed,
        "resource_gate_passed": resource_gate_passed,
        "mechanism_validation_passed": mechanism_validation_passed,
        "reconciliation_required": reconciliation_required,
        "passed": passed,
    }


def _run(args: argparse.Namespace) -> int:
    cases = SUITES[args.suite]
    if (
        args.fidelity == "exact"
        and args.candidate_profile in BOUNDED_ONLY_CANDIDATE_PROFILES
    ):
        raise ValueError(
            f"candidate profile {args.candidate_profile!r} is bounded-only; "
            "use --fidelity bounded"
        )
    run_id = args.run_id or time.strftime("perf-%Y%m%d-%H%M%S")
    run_root = args.output_root / run_id
    if args.dry_run:
        for case in cases:
            scenario_file = run_root / "configs" / f"{case.variant}_{case.fixture}.json"
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            print(
                f"{case.key}: "
                f"{shlex.join(_runner_command(scenario_file=scenario_file, output_root=candidate_root, run_id=run_id, run_goal=args.run_goal))}"
            )
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
        "run_goal": args.run_goal,
        "candidate_profile": args.candidate_profile,
        "candidate_overrides_by_case": {
            case.key: _candidate_overrides(case, args.candidate_profile)
            for case in cases
        },
        "suite": args.suite,
        "fidelity": args.fidelity,
        "thresholds": FIDELITY_THRESHOLDS[args.fidelity],
        "baseline_registry": str(DEFAULT_BASELINE_REGISTRY),
        "baseline_registry_id": read_json(DEFAULT_BASELINE_REGISTRY).get("registry_id"),
        "baseline_references": {case.key: baseline_entries[case.key] for case in cases},
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
            write_json(
                scenario_file,
                _case_scenario(case, args.fidelity, args.candidate_profile),
            )
            candidate_root = run_root / "candidates" / f"{case.variant}_{case.fixture}"
            command = _runner_command(
                scenario_file=scenario_file,
                output_root=candidate_root,
                run_id=run_id,
                run_goal=args.run_goal,
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
            if returncode != 0 and report["parity_passed"]:
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
        **_gate_summary(reports),
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
    run.add_argument("--run-goal", default=DEFAULT_RUN_GOAL)
    run.add_argument(
        "--candidate-profile",
        choices=tuple(CANDIDATE_PROFILES),
        default="canonical",
    )
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
