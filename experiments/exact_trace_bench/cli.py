from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    REPO_ROOT,
    DEFAULT_SCRATCH_ROOT,
)
from .extract import run_full_extraction
from .fixtures import describe_fixture_tiers
from .graph_compare import compare_artifact_dirs
from .io_utils import ensure_dir
from .jobs import render_launch_plan
from .scenarios import SCENARIO_TIERS, build_tier_config, write_tier_config
from .workspace import (
    DEFAULT_SNAPSHOT_ROOT,
    create_workspace_snapshot,
    sibling_library_root,
    verify_import_paths,
)


def _cmd_build_scenarios(args: argparse.Namespace) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = [args.tier] if not args.all_tiers else list(SCENARIO_TIERS)

    fixture_catalog = args.fixture_catalog
    catalog_by_name = {}
    if fixture_catalog.exists():
        from .fixtures import load_fixture_catalog

        catalog_by_name = load_fixture_catalog(fixture_catalog)

    ensure_dir(args.output_dir)
    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_tier_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
            )
            output_path = write_tier_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_extract(args: argparse.Namespace) -> None:
    counts = run_full_extraction(
        input_root=args.input_root,
        output_dir=args.output_dir,
        logs_dir=None if args.skip_slurm or args.logs_dir is None else args.logs_dir,
    )
    print(json.dumps(counts, indent=2))
    print(f"Wrote extraction outputs to {args.output_dir}")


def _cmd_compare_compact(args: argparse.Namespace) -> None:
    summary = compare_artifact_dirs(args.left_artifacts, args.right_artifacts)
    if args.output_json is not None:
        ensure_dir(args.output_json.parent)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote comparison summary to {args.output_json}")
    print(json.dumps(summary, indent=2))


def _cmd_snapshot_workspace(args: argparse.Namespace) -> None:
    snapshot = create_workspace_snapshot(
        snapshot_root=args.snapshot_root,
        source_root=args.source_root,
        label=args.label,
        read_only=not args.mutable_permissions,
    )
    if args.print_path_only:
        print(snapshot)
    else:
        print(
            json.dumps(
                {
                    "snapshot_path": str(snapshot),
                    "library_snapshot_path": (
                        None
                        if sibling_library_root(snapshot) is None
                        else str(sibling_library_root(snapshot))
                    ),
                },
                indent=2,
            )
        )


def _cmd_launch_plan(args: argparse.Namespace) -> None:
    plan = render_launch_plan(
        cluster=args.cluster,
        scenarios_file=args.scenarios_file,
        output_root=args.output_root,
        immutable_workspace=args.immutable_workspace,
        snapshot_root=args.snapshot_root,
        source_root=args.source_root,
        workspace_label=args.workspace_label,
        walltime=args.walltime,
    )
    print(json.dumps(plan, indent=2))


def _cmd_verify_imports(args: argparse.Namespace) -> None:
    result = verify_import_paths(
        workspace_root=args.workspace_root,
        library_root=args.library_root,
    )
    print(json.dumps(result, indent=2))


def _cmd_describe_fixtures(_: argparse.Namespace) -> None:
    print(json.dumps(describe_fixture_tiers(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact trace benchmark harness helpers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_scenarios = subparsers.add_parser(
        "build-scenarios",
        help="Generate benchmark scenario JSON files",
    )
    build_scenarios.add_argument(
        "--tier",
        choices=SCENARIO_TIERS,
        default="fast",
        help="Scenario tier to generate",
    )
    build_scenarios.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all tiers (fast/anomaly/long_eval)",
    )
    build_scenarios.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_scenarios.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_scenarios.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_FIXTURE_CATALOG,
        help="Fixture catalog JSON (fallback paths are used if missing)",
    )
    build_scenarios.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for scenario configs",
    )
    build_scenarios.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_scenarios.set_defaults(func=_cmd_build_scenarios)

    extract = subparsers.add_parser(
        "extract",
        help="Extract and aggregate benchmark rows from results/logs",
    )
    extract.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Benchmark scratch root containing scenario result directories",
    )
    extract.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXTRACTED_DIR,
        help="Directory where extracted tables are written",
    )
    extract.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory with SLURM .err/.out files; omitted by default to avoid mixing unrelated historical logs",
    )
    extract.add_argument(
        "--skip-slurm",
        action="store_true",
        help="Skip parsing SLURM .err/.out files",
    )
    extract.set_defaults(func=_cmd_extract)

    compare_compact = subparsers.add_parser(
        "compare-compact",
        help="Compare two artifact trees with saved compact step_*.npz outputs",
    )
    compare_compact.add_argument("left_artifacts", type=Path)
    compare_compact.add_argument("right_artifacts", type=Path)
    compare_compact.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for comparison JSON",
    )
    compare_compact.set_defaults(func=_cmd_compare_compact)

    snapshot = subparsers.add_parser(
        "snapshot-workspace",
        help="Create immutable workspace snapshot for launch",
    )
    snapshot.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Root directory where snapshots are created",
    )
    snapshot.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace source root to snapshot",
    )
    snapshot.add_argument("--label", default=None)
    snapshot.add_argument(
        "--mutable-permissions",
        action="store_true",
        help="Keep copied files writable (default is read-only)",
    )
    snapshot.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print only snapshot path (useful for shell scripts)",
    )
    snapshot.set_defaults(func=_cmd_snapshot_workspace)

    describe = subparsers.add_parser(
        "describe-fixtures",
        help="Print canonical prompt tier fixture mapping",
    )
    describe.set_defaults(func=_cmd_describe_fixtures)

    launch_plan = subparsers.add_parser(
        "launch-plan",
        help="Render a scratch-rooted sbatch launch plan for a scenarios file",
    )
    launch_plan.add_argument("--cluster", choices=["ascend", "cardinal"], required=True)
    launch_plan.add_argument("--scenarios-file", type=Path, required=True)
    launch_plan.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override scratch output root; defaults to the scenario metadata recommendation",
    )
    launch_plan.add_argument(
        "--immutable-workspace",
        action="store_true",
        help="Launch from a read-only workspace snapshot",
    )
    launch_plan.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Where immutable workspace snapshots are created",
    )
    launch_plan.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace source root for immutable launch snapshots",
    )
    launch_plan.add_argument("--workspace-label", default=None)
    launch_plan.add_argument("--walltime", default=None)
    launch_plan.set_defaults(func=_cmd_launch_plan)

    verify_imports = subparsers.add_parser(
        "verify-imports",
        help="Print resolved import paths for the project and circuit_tracer",
    )
    verify_imports.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Project workspace root to inspect",
    )
    verify_imports.add_argument(
        "--library-root",
        type=Path,
        default=None,
        help="Optional explicit sibling library root to prepend",
    )
    verify_imports.set_defaults(func=_cmd_verify_imports)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
