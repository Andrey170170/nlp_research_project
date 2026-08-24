from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from nlp_research_project.exact_trace_bench.config import REPO_ROOT
from nlp_research_project.exact_trace_bench.jobs import (
    SBATCH_FIXTURE_PREP_SCRIPTS,
    SBATCH_SCRIPTS,
)
from nlp_research_project.exact_trace_bench.scenarios import (
    CHPC_BASELINE_RESOURCE_PROFILE,
)
from nlp_research_project.exact_trace_bench.workspace import (
    create_workspace_snapshot,
    load_snapshot_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SCRIPTS_ARCHIVE_ROOT = SCRIPTS_ROOT / "archive"


def test_canonical_sbatch_templates_live_under_slurm() -> None:
    paths = [*SBATCH_SCRIPTS.values(), *SBATCH_FIXTURE_PREP_SCRIPTS.values()]
    assert paths
    for path in paths:
        assert path.exists()
        assert path.relative_to(REPO_ROOT).parts[:2] == ("slurm", "exact_trace_bench")


def test_granite_h200_baseline_profile_is_launchable() -> None:
    script = SBATCH_SCRIPTS[("granite", CHPC_BASELINE_RESOURCE_PROFILE)]

    assert script.name == "trace_baseline_h200.granite.sbatch"


def test_canonical_sbatch_templates_use_snapshot_import_paths() -> None:
    paths = [*SBATCH_SCRIPTS.values(), *SBATCH_FIXTURE_PREP_SCRIPTS.values()]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert (
            'export PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT:$LIB_WORKSPACE_ROOT'
            in text
        )
        if path.name.endswith(".granite.sbatch"):
            assert '"$UV_PROJECT_ENVIRONMENT/bin/python"' in text
        else:
            assert "uv run --no-sync" in text


def test_granite_templates_source_snapshot_guard_without_submit_dir_fallback() -> None:
    paths = sorted(
        (PROJECT_ROOT / "slurm" / "exact_trace_bench").glob("*.granite.sbatch")
    )
    assert len(paths) == 12
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert (
            'source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"'
            in text
        )
        assert "SLURM_SUBMIT_DIR" not in text


def test_preheated_prepared_wrapper_resolves_and_preflights_bundle() -> None:
    text = (
        PROJECT_ROOT
        / "slurm"
        / "exact_trace_bench"
        / "full_answer_preheated_prepared.granite.sbatch"
    ).read_text(encoding="utf-8")
    assert 'if [[ -d "$PREPARED_WORKLOAD_PATH" ]]; then' in text
    assert (
        'PREPARED_WORKLOAD_PATH="${PREPARED_WORKLOAD_PATH%/}/prepared_workload.json"'
        in text
    )
    assert (
        '[[ ! -f "$PREPARED_WORKLOAD_PATH" || ! -r "$PREPARED_WORKLOAD_PATH" ]]' in text
    )
    snapshot_imports = text.index('export PYTHONPATH="$WORKSPACE_ROOT/src:')
    expectation_resolution = text.index("prepared_expectations")
    preflight = text.index("--validate-only")
    cache_preflight = text.index("validate_hf_cache.py")
    runtime_environment = text.index("runtime_environment.json")
    preheat = text.index('echo "Preheat start:')
    launch = text.rindex('run-prepared-campaign-workload "$PREPARED_WORKLOAD_PATH"')
    assert snapshot_imports < expectation_resolution < preflight
    assert preflight < cache_preflight < runtime_environment < preheat < launch
    assert "reconcile_expectation EXPECTED_BACKWARD_ENGINE_MODE" in text
    assert "reconcile_expectation EXPECTED_FORWARD_GRAPH_MODE" in text
    assert "reconcile_expectation EXPECTED_VJP_KERNEL_MODE" in text
    assert "reconcile_expectation EXPECTED_FORWARD_LANE_COUNT" in text
    assert "reconcile_expectation EXPECTED_SESSION_CAPACITY" in text
    assert "reconcile_expectation EXPECTED_BACKWARD_BATCH_CAPACITY" in text
    assert "reconcile_expectation EXPECTED_PHASE1_TRACE_BATCH_SIZE_MAX" in text
    assert "reconcile_expectation EXPECTED_PHASE1_EFFECTIVE_TRACE_BATCH_SIZE" in text
    assert "reconcile_expectation EXPECTED_PHASE3_BATCH_SIZE" in text
    assert "reconcile_expectation EXPECTED_PHASE4_BATCH_SIZE" in text
    assert "reconcile_expectation REQUIRE_FULL_COMPLETION" in text
    assert "reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY" in text
    assert (
        "reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_RESIDENCY_REQUIREMENT"
        in text
    )
    assert "reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_MAX_BYTES" in text
    assert (
        "reconcile_expectation EXPECTED_DECODER_ACTIVE_ROW_SAFETY_MARGIN_BYTES" in text
    )
    assert (
        "EXPECTED_PROVIDER_FILE_COUNT:?EXPECTED_PROVIDER_FILE_COUNT is required" in text
    )
    assert "--expected-backward-engine-mode" in text
    assert "--expected-forward-lane-count" in text
    assert "--expected-phase1-trace-batch-size-max" in text
    assert "--expected-phase1-effective-trace-batch-size" in text
    assert "validation_args+=(--require-full-completion)" in text
    assert "validation_args+=(--require-structured-vjp-evidence)" in text
    assert '[[ "$EXPECTED_VJP_KERNEL_MODE" == "autograd_batched" ]]' in text
    assert "validation_args+=(--require-phase4-device-timing)" in text
    assert 'REQUIRE_PHASE4_DEVICE_TIMING="${REQUIRE_PHASE4_DEVICE_TIMING:-0}"' in text
    assert '"$EXPECTED_PHASE4_DEVICE_TIMING_BACKEND"' in text
    timing_opt_in = text.index('if [[ "$REQUIRE_PHASE4_DEVICE_TIMING" == "1" ]]')
    timing_backend = text.index(
        'if [[ -n "${EXPECTED_PHASE4_DEVICE_TIMING_BACKEND:-}" ]]', timing_opt_in
    )
    timing_opt_out = text.index(
        'elif [[ "$REQUIRE_PHASE4_DEVICE_TIMING" != "0" ]]', timing_backend
    )
    assert timing_opt_in < timing_backend < timing_opt_out
    assert "validation_args+=(--require-unambiguous-per-device-gpu-evidence)" in text
    assert (
        'REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE="${REQUIRE_UNAMBIGUOUS_PER_DEVICE_GPU_EVIDENCE:-0}"'
        in text
    )
    assert '"${EXPECTED_GRAPH_FEATURE_COUNT:-}"' in text
    assert '--expected-graph-feature-count "$EXPECTED_GRAPH_FEATURE_COUNT"' in text
    assert "EXPECTED_GRAPH_EDGE_COUNT" not in text
    assert "validation_args+=(--require-typed-only-graph)" in text
    assert 'REQUIRE_TYPED_ONLY_GRAPH="${REQUIRE_TYPED_ONLY_GRAPH:-0}"' in text
    for variable, flag in (
        ("EXPECTED_TYPED_SCHEMA_VERSION", "--expected-graph-schema-version"),
        ("EXPECTED_TYPED_SAVE_FORMAT", "--expected-graph-save-format"),
        (
            "EXPECTED_TYPED_RETENTION_POLICY_ID",
            "--expected-graph-retention-policy-id",
        ),
        (
            "EXPECTED_TYPED_RETENTION_POLICY_FINGERPRINT",
            "--expected-graph-retention-policy-fingerprint",
        ),
    ):
        assert variable in text
        assert flag in text
    assert "--expected-graph-feature-count 8192" not in text
    assert "--expected-graph-edge-count 20000" not in text
    assert "--expected-decoder-active-row-residency" in text
    assert "--expected-decoder-active-row-safety-margin-bytes" in text
    assert (
        "nlp_research_project.exact_trace_bench.full_answer.runtime_environment" in text
    )
    assert 'preheat_args+=(--workers "$PREHEAT_WORKERS")' in text


def test_preheated_sequence_preflights_all_entries_before_one_preheat() -> None:
    text = (
        PROJECT_ROOT
        / "slurm"
        / "exact_trace_bench"
        / "full_answer_preheated_sequence.granite.sbatch"
    ).read_text(encoding="utf-8")
    snapshot_imports = text.index('export PYTHONPATH="$WORKSPACE_ROOT/src:')
    inspect = text.index("inspect-prepared-campaign-sequence")
    validate_only = text.index('--state-root "$SEQUENCE_STATE_ROOT" --validate-only')
    cache_preflight = text.index("validate_hf_cache.py")
    preheat = text.index('echo "Preheat start:')
    launch = text.rindex("run-prepared-campaign-sequence")
    assert (
        snapshot_imports < inspect < validate_only < cache_preflight < preheat < launch
    )
    assert text.count("preheat_file_cache.py") == 1
    assert 'preheat_args+=(--workers "$PREHEAT_WORKERS")' in text
    assert "--format output-paths" in text
    assert "root collides with" in text


def test_preheated_pair_checks_offline_cache_before_creating_output() -> None:
    text = (
        PROJECT_ROOT
        / "slurm"
        / "exact_trace_bench"
        / "full_answer_pair_preheated.granite.sbatch"
    ).read_text(encoding="utf-8")
    cache_preflight = text.index("validate_hf_cache.py")
    output_creation = text.index('mkdir -p "$PAIR_OUTPUT_ROOT"')
    preheat = text.index('echo "Preheat start:')
    assert cache_preflight < output_creation < preheat
    assert 'preheat_args+=(--workers "$PREHEAT_WORKERS")' in text


def test_chpc_env_uses_one_authoritative_offline_hf_cache(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "slurm" / "exact_trace_bench" / "chpc_env.sh"
    cache_root = tmp_path / "shared-hf"
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_CACHE_ROOT": str(tmp_path / "project-cache"),
            "EXACT_TRACE_HF_CACHE_ROOT": str(cache_root),
            "HF_HOME": str(tmp_path / "stale-home"),
            "HF_HUB_CACHE": str(tmp_path / "stale-hub"),
            "TRANSFORMERS_CACHE": str(tmp_path / "stale-transformers"),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        }
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{script}" && printf "%s\\n" "$HF_HOME" "$HF_HUB_CACHE" '
            '"$TRANSFORMERS_CACHE" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE"',
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        str(cache_root),
        str(cache_root),
        str(cache_root),
        "1",
        "1",
    ]


def test_prefetch_templates_explicitly_enable_online_hf_access() -> None:
    for name in (
        "download_model_and_transcoders.granite.sbatch",
        "download_transcoders.granite.sbatch",
    ):
        text = (PROJECT_ROOT / "slurm" / "exact_trace_bench" / name).read_text(
            encoding="utf-8"
        )
        online = text.index("export EXACT_TRACE_HF_OFFLINE=0")
        shared_env = text.index(
            'source "$WORKSPACE_ROOT/slurm/exact_trace_bench/chpc_env.sh"'
        )
        assert online < shared_env


def test_wrapper_scripts_call_console_entrypoint() -> None:
    wrapper_expectations = {
        "exact_trace_bench_fast_ascend.sh": "fast-ascend",
        "exact_trace_bench_fast_cardinal.sh": "fast-cardinal",
        "exact_trace_bench_full_ascend.sh": "full-ascend",
        "exact_trace_bench_full_cardinal.sh": "full-cardinal",
        "exact_trace_bench_fast_all.sh": "fast-all",
        "exact_trace_bench_full_all.sh": "full-all",
    }
    for script_name, preset in wrapper_expectations.items():
        text = (SCRIPTS_ARCHIVE_ROOT / script_name).read_text(encoding="utf-8")
        assert f"uv run exact-trace-bench submit-preset --preset {preset}" in text


def test_root_scripts_directory_only_contains_archive_and_readme() -> None:
    entries = {
        path.name for path in SCRIPTS_ROOT.iterdir() if path.name != "__pycache__"
    }
    assert entries == {"README.md", "archive", "transfer_scratch_pas2836.sh"}


def test_module_entrypoint_help_is_login_safe() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-m", "nlp_research_project.exact_trace_bench", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert proc.returncode == 0
    assert "snapshot-workspace" in proc.stdout


def test_workspace_snapshot_manifest_records_repo_state(tmp_path: Path) -> None:
    source_root = tmp_path / "project"
    sibling = tmp_path / "circuit-tracer_chunked"
    source_root.mkdir()
    sibling.mkdir()
    (source_root / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        'circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }\n',
        encoding="utf-8",
    )
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (sibling / "lib.py").write_text("VALUE = 2\n", encoding="utf-8")

    snapshot = create_workspace_snapshot(
        snapshot_root=tmp_path / "snapshots",
        source_root=source_root,
        read_only=False,
    )
    manifest = load_snapshot_manifest(snapshot)

    assert manifest["snapshot_root"] == str(snapshot)
    assert manifest["read_only"] is False
    assert set(manifest["repo_state"]) == {"branch", "commit", "dirty_files"}
    assert manifest["uv_source_snapshots"][0]["repo_state"]["dirty_files"] == []
    assert (snapshot / "module.py").exists()


def test_workspace_snapshot_normalizes_relative_source_root(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "project"
    sibling = tmp_path / "circuit-tracer_chunked"
    source_root.mkdir()
    sibling.mkdir()
    (source_root / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        'circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }\n',
        encoding="utf-8",
    )
    (sibling / "lib.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    snapshot = create_workspace_snapshot(
        snapshot_root=Path("snapshots"),
        source_root=Path("project"),
        read_only=False,
    )
    manifest = load_snapshot_manifest(snapshot)

    assert snapshot == (tmp_path / "snapshots" / snapshot.parent.name / "project")
    assert manifest["source_root"] == str(source_root)
    assert Path(manifest["uv_source_snapshots"][0]["snapshot_path"]) == (
        snapshot.parent / "circuit-tracer_chunked"
    )


def test_workspace_snapshot_excludes_nested_uv_cache(tmp_path: Path) -> None:
    source_root = tmp_path / "project"
    uv_cache = source_root / "nested" / ".uv-cache"
    uv_cache.mkdir(parents=True)
    (uv_cache / "sentinel").write_text("must not be copied\n", encoding="utf-8")

    snapshot = create_workspace_snapshot(
        snapshot_root=tmp_path / "snapshots",
        source_root=source_root,
        read_only=False,
    )
    manifest = load_snapshot_manifest(snapshot)

    assert not (snapshot / "nested" / ".uv-cache").exists()
    assert ".uv-cache" in manifest["ignored_names"]
