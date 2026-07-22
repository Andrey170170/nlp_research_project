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
    assert len(paths) == 9
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert 'source "$WORKSPACE_ROOT/slurm/exact_trace_bench/require_snapshot_workspace.sh"' in text
        assert "SLURM_SUBMIT_DIR" not in text


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
