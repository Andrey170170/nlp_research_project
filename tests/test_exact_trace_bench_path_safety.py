from __future__ import annotations

import importlib
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

from nlp_research_project.exact_trace_bench import cli, presets, workspace


def _fake_module(file_path: Path) -> SimpleNamespace:
    return SimpleNamespace(__file__=str(file_path))


def test_verify_import_paths_prefers_workspace_src_and_reports_files(
    tmp_path: Path, monkeypatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_src = workspace_root / "src"
    library_root = tmp_path / "circuit-tracer_chunked"
    (workspace_src / "nlp_research_project" / "exact_trace_bench").mkdir(parents=True)
    (
        workspace_src / "nlp_research_project" / "exact_trace_bench" / "__init__.py"
    ).write_text(
        "__all__ = []\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "nlp_research_project.exact_trace_bench":
            return _fake_module(
                workspace_src
                / "nlp_research_project"
                / "exact_trace_bench"
                / "__init__.py"
            )
        if name == "circuit_tracer":
            return _fake_module(library_root / "circuit_tracer" / "__init__.py")
        if name == "nlp_research_project.exact_trace_bench.trace_runtime":
            return _fake_module(
                workspace_src
                / "nlp_research_project"
                / "exact_trace_bench"
                / "trace_runtime"
                / "__init__.py"
            )
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    result = workspace.verify_import_paths(
        workspace_root=workspace_root,
        library_root=library_root,
    )

    assert calls == [
        "nlp_research_project.exact_trace_bench",
        "circuit_tracer",
        "nlp_research_project.exact_trace_bench.trace_runtime",
    ]
    assert sys.path[:3] == [
        str(workspace_src),
        str(workspace_root),
        str(library_root),
    ]
    assert result["workspace_src"] == str(workspace_src.resolve())
    assert result["exact_trace_bench_file"] == str(
        (
            workspace_src / "nlp_research_project" / "exact_trace_bench" / "__init__.py"
        ).resolve()
    )
    assert result["circuit_tracer_file"] == str(
        (library_root / "circuit_tracer" / "__init__.py").resolve()
    )
    assert result["trace_runtime_file"] == str(
        (
            workspace_src
            / "nlp_research_project"
            / "exact_trace_bench"
            / "trace_runtime"
            / "__init__.py"
        ).resolve()
    )


def test_verify_import_paths_rejects_exact_trace_bench_outside_workspace_src(
    tmp_path: Path, monkeypatch
) -> None:
    workspace_root = tmp_path / "workspace"
    library_root = tmp_path / "circuit-tracer_chunked"
    outside_file = tmp_path / "elsewhere" / "__init__.py"

    def fake_import_module(name: str):
        if name == "nlp_research_project.exact_trace_bench":
            return _fake_module(outside_file)
        return _fake_module(library_root / f"{name}" / "__init__.py")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    try:
        workspace.verify_import_paths(
            workspace_root=workspace_root,
            library_root=library_root,
        )
    except ImportError as exc:
        assert "workspace_root/src" in str(exc)
    else:
        raise AssertionError("Expected ImportError for out-of-tree exact_trace_bench")


def _make_mutable_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "project"
    sibling_root = tmp_path / "circuit-tracer_chunked"
    source_root.mkdir()
    sibling_root.mkdir()
    (source_root / "src").mkdir()
    (source_root / "pyproject.toml").write_text(
        "[tool.uv.sources]\n"
        'circuit-tracer = { path = "../circuit-tracer_chunked", editable = true }\n',
        encoding="utf-8",
    )
    (sibling_root / "library.py").write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = workspace.create_workspace_snapshot(
        snapshot_root=tmp_path / "snapshots",
        source_root=source_root,
        read_only=False,
    )
    library_snapshot = workspace.sibling_library_root(snapshot)
    assert library_snapshot is not None
    return snapshot, library_snapshot


def test_make_snapshot_read_only_updates_manifest_before_freezing(tmp_path: Path) -> None:
    snapshot, _ = _make_mutable_snapshot(tmp_path)

    workspace.make_snapshot_read_only(snapshot)

    manifest = workspace.load_snapshot_manifest(snapshot)
    assert manifest["read_only"] is True
    assert snapshot.stat().st_mode & 0o222 == 0
    assert workspace._manifest_path_for_workspace(snapshot).stat().st_mode & 0o222 == 0


def test_validate_launch_snapshot_returns_provenance(tmp_path: Path) -> None:
    snapshot, library_snapshot = _make_mutable_snapshot(tmp_path)
    workspace.make_snapshot_read_only(snapshot)

    provenance = workspace.validate_launch_snapshot(
        workspace_root=snapshot,
        library_root=library_snapshot,
        import_roots=(snapshot / "src", snapshot, library_snapshot),
    )

    assert provenance["workspace_mode"] == "immutable"
    assert provenance["workspace_root"] == str(snapshot.resolve())
    assert provenance["library_workspace_root"] == str(library_snapshot.resolve())
    assert provenance["read_only"] is True


def test_validate_launch_snapshot_rejects_invalid_roots_and_manifest(
    tmp_path: Path,
) -> None:
    snapshot, library_snapshot = _make_mutable_snapshot(tmp_path)
    try:
        workspace.validate_launch_snapshot(
            workspace_root=snapshot,
            library_root=library_snapshot,
        )
    except ValueError as exc:
        assert "writable" in str(exc) or "read_only" in str(exc)
    else:
        raise AssertionError("Expected mutable snapshot validation to fail")

    workspace.make_snapshot_read_only(snapshot)
    try:
        workspace.validate_launch_snapshot(
            workspace_root=snapshot,
            library_root=library_snapshot,
            import_roots=(tmp_path / "outside",),
        )
    except ValueError as exc:
        assert "outside supplied snapshot roots" in str(exc)
    else:
        raise AssertionError("Expected out-of-tree import root validation to fail")


def test_launch_plan_cli_defaults_immutable_and_supports_live_alias(tmp_path: Path) -> None:
    parser = cli.build_parser()
    immutable = parser.parse_args(
        [
            "launch-plan",
            "--cluster",
            "granite",
            "--scenarios-file",
            str(tmp_path / "scenarios.json"),
        ]
    )
    assert immutable.immutable_workspace is True
    assert immutable.mem is None

    memory_override = parser.parse_args(
        [
            "launch-plan",
            "--cluster",
            "granite",
            "--scenarios-file",
            str(tmp_path / "scenarios.json"),
            "--mem",
            "600G",
        ]
    )
    assert memory_override.mem == "600G"

    live = parser.parse_args(
        [
            "launch-plan",
            "--cluster",
            "granite",
            "--scenarios-file",
            str(tmp_path / "scenarios.json"),
            "--no-immutable-workspace",
            "--live-workspace-rationale",
            "interactive diagnosis",
        ]
    )
    assert live.immutable_workspace is False
    assert live.live_workspace_rationale == "interactive diagnosis"


def test_run_preset_freezes_snapshot_before_submitting_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "repo"
    generated_dir = source_root / "experiments" / "generated" / "exact_trace_bench"
    snapshot_root = tmp_path / "snapshots"
    snapshot_path = snapshot_root / "workspace_test"
    events: list[str] = []

    monkeypatch.setattr(
        presets,
        "load_fixture_catalog",
        lambda path: {},
    )
    monkeypatch.setattr(
        presets,
        "create_workspace_snapshot",
        lambda **kwargs: snapshot_path,
    )
    monkeypatch.setattr(
        presets,
        "make_snapshot_read_only",
        lambda path: events.append(f"freeze:{path}"),
    )
    monkeypatch.setattr(
        presets,
        "build_tier_config",
        lambda **kwargs: {"cluster": kwargs["cluster"], "tier": kwargs["tier"]},
    )

    def fake_write_tier_config(payload, *, output_dir, tier, cluster):
        events.append(f"write:{cluster}/{tier}")
        output_path = output_dir / f"{cluster}_{tier}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return output_path

    def fake_render_launch_plan(**kwargs):
        events.append(f"plan:{kwargs['cluster']}/{kwargs['scenarios_file'].stem}")
        return {
            "sbatch_script": str(
                source_root / "slurm" / "exact_trace_bench" / "fake.sbatch"
            ),
            "sbatch_argv": [
                "sbatch",
                str(source_root / "slurm" / "exact_trace_bench" / "fake.sbatch"),
            ],
            "sbatch_command": "sbatch fake.sbatch",
            "immutable_workspace": kwargs["immutable_workspace"],
        }

    def fake_subprocess_run(argv, check):
        events.append(f"submit:{Path(argv[1]).name}")
        raise subprocess.CalledProcessError(returncode=1, cmd=argv)

    monkeypatch.setattr(presets, "write_tier_config", fake_write_tier_config)
    monkeypatch.setattr(presets, "render_launch_plan", fake_render_launch_plan)
    monkeypatch.setattr(presets.subprocess, "run", fake_subprocess_run)

    try:
        presets.run_preset(
            preset="fast-ascend",
            generated_dir=generated_dir,
            scratch_root=tmp_path / "scratch",
            snapshot_root=snapshot_root,
            source_root=source_root,
            print_only=False,
        )
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("Expected submitted preset to fail in the test harness")

    assert events.index("freeze:%s" % snapshot_path) < events.index(
        "submit:fake.sbatch"
    )
    assert any(event.startswith("freeze:") for event in events)


def test_run_preset_does_not_freeze_source_root_when_snapshot_creation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "repo"
    source_root.mkdir()
    events: list[str] = []

    monkeypatch.setattr(presets, "load_fixture_catalog", lambda path: {})

    def fail_create_workspace_snapshot(**kwargs):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        presets,
        "create_workspace_snapshot",
        fail_create_workspace_snapshot,
    )
    monkeypatch.setattr(
        presets,
        "make_snapshot_read_only",
        lambda path: events.append(f"freeze:{path}"),
    )

    try:
        presets.run_preset(
            preset="fast-ascend",
            generated_dir=source_root / "experiments" / "generated",
            fixture_catalog=source_root / "missing_catalog.json",
            snapshot_root=tmp_path / "snapshots",
            source_root=source_root,
            print_only=True,
        )
    except RuntimeError as exc:
        assert str(exc) == "snapshot failed"
    else:
        raise AssertionError("Expected snapshot creation failure")

    assert events == []
