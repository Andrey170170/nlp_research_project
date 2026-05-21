from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .config import (
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_FIXTURE_CATALOG,
    DEFAULT_GENERATED_DIR,
    REPO_ROOT,
    DEFAULT_SCRATCH_ROOT,
    DEFAULT_WAVE0_BASELINE_REGISTRY,
    DEFAULT_WAVE0_FIXTURE_CATALOG,
    DEFAULT_WAVE0_FIXTURE_OUTPUT_DIR,
    DEFAULT_WAVE0_FIXTURE_TARGET_SPEC,
)
from .extract import run_full_extraction
from .fixtures import describe_fixture_tiers
from .graph_compare import compare_artifact_dirs
from .io_utils import ensure_dir
from .jobs import render_fixture_prep_plan, render_launch_plan
from .phase0_replay_matrix_compare import compare_phase0_replay_matrix_to_json
from .phase3_seed_bundle_compare import compare_phase3_seed_bundles_to_json
from .presets import preset_names, run_preset
from .scenarios import (
    SCENARIO_TIERS,
    WAVE2A_PHASE1_TIERS,
    WAVE2B_PHASE4_TIERS,
    WAVE2C_ROW_ENCODER_TIERS,
    WAVE3_INTERACTION_CONFIRMATION_TIERS,
    build_tier_config,
    build_wave2a_phase1_config,
    build_wave2b_phase4_config,
    build_wave2c_row_encoder_config,
    build_wave3_interaction_confirmation_config,
    build_wave0_baseline_config,
    write_tier_config,
    write_wave2a_phase1_config,
    write_wave2b_phase4_config,
    write_wave2c_row_encoder_config,
    write_wave3_interaction_confirmation_config,
    write_wave0_baseline_config,
)
from .semantic_feature_compare import compare_semantic_feature_descriptors_to_json
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


def _cmd_build_wave0_scenarios(args: argparse.Namespace) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = [args.tier] if not args.all_tiers else list(SCENARIO_TIERS)

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            "Wave 0 scenario generation requires a prepared fixture catalog. "
            f"Missing: {args.fixture_catalog}"
        )

    from .fixtures import load_fixture_catalog

    catalog_by_name = load_fixture_catalog(args.fixture_catalog)
    ensure_dir(args.output_dir)

    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_wave0_baseline_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
            )
            output_path = write_wave0_baseline_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote Wave 0 scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_build_wave2a_phase1_scenarios(args: argparse.Namespace) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = [args.tier] if not args.all_tiers else list(WAVE2A_PHASE1_TIERS)

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            "Wave 2A Phase-1 scenario generation requires a prepared fixture catalog. "
            f"Missing: {args.fixture_catalog}"
        )

    from .fixtures import load_fixture_catalog

    catalog_by_name = load_fixture_catalog(args.fixture_catalog)
    ensure_dir(args.output_dir)

    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_wave2a_phase1_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
                baseline_registry=args.baseline_registry,
            )
            output_path = write_wave2a_phase1_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote Wave 2A Phase-1 scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_build_wave2b_phase4_scenarios(args: argparse.Namespace) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = [args.tier] if not args.all_tiers else list(WAVE2B_PHASE4_TIERS)

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            "Wave 2B Phase-4 scenario generation requires a prepared fixture catalog. "
            f"Missing: {args.fixture_catalog}"
        )

    from .fixtures import load_fixture_catalog

    catalog_by_name = load_fixture_catalog(args.fixture_catalog)
    ensure_dir(args.output_dir)

    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_wave2b_phase4_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
                baseline_registry=args.baseline_registry,
            )
            output_path = write_wave2b_phase4_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote Wave 2B Phase-4 scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_build_wave2c_row_encoder_scenarios(args: argparse.Namespace) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = [args.tier] if not args.all_tiers else list(WAVE2C_ROW_ENCODER_TIERS)

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            "Wave 2C row/encoder scenario generation requires a prepared fixture catalog. "
            f"Missing: {args.fixture_catalog}"
        )

    from .fixtures import load_fixture_catalog

    catalog_by_name = load_fixture_catalog(args.fixture_catalog)
    ensure_dir(args.output_dir)

    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_wave2c_row_encoder_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
                baseline_registry=args.baseline_registry,
            )
            output_path = write_wave2c_row_encoder_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote Wave 2C row/encoder/staging/planner scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_build_wave3_interaction_confirmation_scenarios(
    args: argparse.Namespace,
) -> None:
    clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
    tiers = (
        [args.tier]
        if not args.all_tiers
        else list(WAVE3_INTERACTION_CONFIRMATION_TIERS)
    )

    if not args.fixture_catalog.exists():
        raise FileNotFoundError(
            "Wave 3 interaction-confirmation scenario generation requires a prepared fixture catalog. "
            f"Missing: {args.fixture_catalog}"
        )

    from .fixtures import load_fixture_catalog

    catalog_by_name = load_fixture_catalog(args.fixture_catalog)
    ensure_dir(args.output_dir)

    written_paths: list[Path] = []
    for cluster in clusters:
        for tier in tiers:
            payload = build_wave3_interaction_confirmation_config(
                tier=tier,
                cluster=cluster,
                catalog_by_name=catalog_by_name,
                scratch_root=args.scratch_root,
                baseline_registry=args.baseline_registry,
                include_optional_speed_interaction=args.include_optional_speed_interaction,
            )
            output_path = write_wave3_interaction_confirmation_config(
                payload,
                output_dir=args.output_dir,
                tier=tier,
                cluster=cluster,
            )
            written_paths.append(output_path)

    print("Wrote Wave 3 interaction-confirmation scenario configs:")
    for path in written_paths:
        print(f"  - {path}")


def _cmd_build_baseline_registry(args: argparse.Namespace) -> None:
    from .baselines import build_baseline_registry_from_run_roots, wave0_run_roots

    if args.run_root:
        run_roots = [Path(path) for path in args.run_root]
    else:
        clusters = [args.cluster] if not args.all_clusters else ["ascend", "cardinal"]
        tiers = [args.tier] if not args.all_tiers else list(SCENARIO_TIERS)
        run_roots = wave0_run_roots(
            run_id=args.run_id,
            scratch_root=args.scratch_root,
            clusters=tuple(clusters),
            tiers=tuple(tiers),
        )
    registry = build_baseline_registry_from_run_roots(
        run_roots,
        registry_id=args.registry_id or args.run_id,
        project_root=args.project_root,
        library_root=args.library_root,
    )
    write_path = args.output
    ensure_dir(write_path.parent)
    write_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(write_path), "entries": len(registry["entries"])}, indent=2
        )
    )


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


def _cmd_compare_phase3_seed_bundles(args: argparse.Namespace) -> None:
    summary = compare_phase3_seed_bundles_to_json(
        args.left_bundle,
        args.right_bundle,
        output_json=args.output_json,
    )
    if args.output_json is not None:
        print(f"Wrote Phase-3 seed bundle comparison to {args.output_json}")
    print(json.dumps(summary, indent=2))


def _cmd_compare_semantic_features(args: argparse.Namespace) -> None:
    summary = compare_semantic_feature_descriptors_to_json(
        args.left_descriptor,
        args.right_descriptor,
        output_json=args.output_json,
        position_window=args.position_window,
        similarity_threshold=args.similarity_threshold,
    )
    if args.output_json is not None:
        print(f"Wrote semantic feature comparison to {args.output_json}")
    print(json.dumps(summary, indent=2))


def _cmd_compare_phase0_replay_matrix(args: argparse.Namespace) -> None:
    summary = compare_phase0_replay_matrix_to_json(
        ascend_baseline=args.ascend_baseline,
        cardinal_baseline=args.cardinal_baseline,
        ascend_self_replay=args.ascend_self_replay,
        cardinal_self_replay=args.cardinal_self_replay,
        ascend_with_cardinal=args.ascend_with_cardinal,
        cardinal_with_ascend=args.cardinal_with_ascend,
        output_json=args.output_json,
    )
    if args.output_json is not None:
        print(f"Wrote phase-0 replay matrix comparison to {args.output_json}")
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
        run_id=args.run_id,
        run_name=args.run_name,
        run_description=args.run_description,
        run_goal=args.run_goal,
        immutable_workspace=args.immutable_workspace,
        snapshot_root=args.snapshot_root,
        source_root=args.source_root,
        workspace_label=args.workspace_label,
        walltime=args.walltime,
        baseline_registry=args.baseline_registry,
        fail_on_baseline_missing=args.fail_on_baseline_missing,
        fail_on_validation_fail=args.fail_on_validation_fail,
    )
    print(json.dumps(plan, indent=2))


def _cmd_submit_fixture_prep(args: argparse.Namespace) -> None:
    plan = render_fixture_prep_plan(
        cluster=args.cluster,
        target_spec_file=args.target_spec_file,
        output_dir=args.output_dir,
        decoder_chunk_size=args.decoder_chunk_size,
        cross_batch_decoder_cache_bytes=args.cross_batch_decoder_cache_bytes,
        immutable_workspace=not args.no_immutable_workspace,
        snapshot_root=args.snapshot_root,
        source_root=args.source_root,
        workspace_label=args.workspace_label,
        walltime=args.walltime,
        run_name=args.run_name,
    )
    if not args.print_only:
        subprocess.run(plan["sbatch_argv"], check=True)
    print(json.dumps(plan, indent=2))


def _cmd_verify_imports(args: argparse.Namespace) -> None:
    result = verify_import_paths(
        workspace_root=args.workspace_root,
        library_root=args.library_root,
    )
    print(json.dumps(result, indent=2))


def _cmd_describe_fixtures(_: argparse.Namespace) -> None:
    print(json.dumps(describe_fixture_tiers(), indent=2))


def _cmd_submit_preset(args: argparse.Namespace) -> None:
    plans = run_preset(
        preset=args.preset,
        generated_dir=args.generated_dir,
        fixture_catalog=args.fixture_catalog,
        scratch_root=args.scratch_root,
        immutable_workspace=not args.no_immutable_workspace,
        snapshot_root=args.snapshot_root,
        source_root=args.source_root,
        workspace_label_prefix=args.workspace_label_prefix,
        walltime=args.walltime,
        run_id=args.run_id,
        run_name=args.run_name,
        run_description=args.run_description,
        run_goal=args.run_goal,
        print_only=args.print_only,
    )
    print(json.dumps({"preset": args.preset, "plans": plans}, indent=2))


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

    build_wave0_scenarios = subparsers.add_parser(
        "build-wave0-scenarios",
        help="Generate expanded Wave 0 baseline scenario JSON files",
    )
    build_wave0_scenarios.add_argument(
        "--tier",
        choices=SCENARIO_TIERS,
        default="fast",
        help="Wave 0 scenario tier to generate",
    )
    build_wave0_scenarios.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all Wave 0 tiers (fast/anomaly/long_eval)",
    )
    build_wave0_scenarios.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_wave0_scenarios.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_wave0_scenarios.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_CATALOG,
        help="Prepared Wave 0 fixture catalog JSON; must already exist",
    )
    build_wave0_scenarios.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for Wave 0 scenario configs",
    )
    build_wave0_scenarios.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_wave0_scenarios.set_defaults(func=_cmd_build_wave0_scenarios)

    build_wave2a_phase1 = subparsers.add_parser(
        "build-wave2a-phase1-scenarios",
        help="Generate Wave 2A Phase-1 trace-batch policy scenario JSON files",
    )
    build_wave2a_phase1.add_argument(
        "--tier",
        choices=WAVE2A_PHASE1_TIERS,
        default="fast",
        help="Wave 2A Phase-1 tier to generate",
    )
    build_wave2a_phase1.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all Wave 2A Phase-1 tiers (fast/anomaly)",
    )
    build_wave2a_phase1.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_wave2a_phase1.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_wave2a_phase1.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_CATALOG,
        help="Prepared Wave 0 fixture catalog JSON; must already exist",
    )
    build_wave2a_phase1.add_argument(
        "--baseline-registry",
        type=Path,
        default=DEFAULT_WAVE0_BASELINE_REGISTRY,
        help="Pinned Wave 0 baseline registry recorded in generated metadata",
    )
    build_wave2a_phase1.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for Wave 2A Phase-1 scenario configs",
    )
    build_wave2a_phase1.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_wave2a_phase1.set_defaults(func=_cmd_build_wave2a_phase1_scenarios)

    build_wave2b_phase4 = subparsers.add_parser(
        "build-wave2b-phase4-scenarios",
        help="Generate Wave 2B Phase-4 scheduler/refresh/ranker/executor scenario JSON files",
    )
    build_wave2b_phase4.add_argument(
        "--tier",
        choices=WAVE2B_PHASE4_TIERS,
        default="fast",
        help="Wave 2B Phase-4 tier to generate",
    )
    build_wave2b_phase4.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all Wave 2B Phase-4 tiers (fast/anomaly)",
    )
    build_wave2b_phase4.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_wave2b_phase4.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_wave2b_phase4.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_CATALOG,
        help="Prepared Wave 0 fixture catalog JSON; must already exist",
    )
    build_wave2b_phase4.add_argument(
        "--baseline-registry",
        type=Path,
        default=DEFAULT_WAVE0_BASELINE_REGISTRY,
        help="Pinned Wave 0 baseline registry recorded in generated metadata",
    )
    build_wave2b_phase4.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for Wave 2B Phase-4 scenario configs",
    )
    build_wave2b_phase4.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_wave2b_phase4.set_defaults(func=_cmd_build_wave2b_phase4_scenarios)

    build_wave2c_row_encoder = subparsers.add_parser(
        "build-wave2c-row-encoder-scenarios",
        help="Generate Wave 2C row/encoder/staging/planner scenario JSON files",
    )
    build_wave2c_row_encoder.add_argument(
        "--tier",
        choices=WAVE2C_ROW_ENCODER_TIERS,
        default="fast",
        help="Wave 2C row/encoder tier to generate",
    )
    build_wave2c_row_encoder.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all Wave 2C row/encoder tiers (fast/anomaly)",
    )
    build_wave2c_row_encoder.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_wave2c_row_encoder.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_wave2c_row_encoder.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_CATALOG,
        help="Prepared Wave 0 fixture catalog JSON; must already exist",
    )
    build_wave2c_row_encoder.add_argument(
        "--baseline-registry",
        type=Path,
        default=DEFAULT_WAVE0_BASELINE_REGISTRY,
        help="Pinned Wave 0 baseline registry recorded in generated metadata",
    )
    build_wave2c_row_encoder.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for Wave 2C row/encoder scenario configs",
    )
    build_wave2c_row_encoder.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_wave2c_row_encoder.set_defaults(func=_cmd_build_wave2c_row_encoder_scenarios)

    build_wave3_interaction = subparsers.add_parser(
        "build-wave3-interaction-confirmation-scenarios",
        help="Generate Wave 3 interaction-confirmation scenario JSON files",
    )
    build_wave3_interaction.add_argument(
        "--tier",
        choices=WAVE3_INTERACTION_CONFIRMATION_TIERS,
        default="fast",
        help="Wave 3 interaction-confirmation tier to generate",
    )
    build_wave3_interaction.add_argument(
        "--all-tiers",
        action="store_true",
        help="Generate all Wave 3 interaction-confirmation tiers (fast/anomaly)",
    )
    build_wave3_interaction.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to target",
    )
    build_wave3_interaction.add_argument(
        "--all-clusters",
        action="store_true",
        help="Generate configs for both clusters",
    )
    build_wave3_interaction.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_CATALOG,
        help="Prepared Wave 0 fixture catalog JSON; must already exist",
    )
    build_wave3_interaction.add_argument(
        "--baseline-registry",
        type=Path,
        default=DEFAULT_WAVE0_BASELINE_REGISTRY,
        help="Pinned Wave 0 baseline registry recorded in generated metadata",
    )
    build_wave3_interaction.add_argument(
        "--include-optional-speed-interaction",
        action="store_true",
        help="Include optional deferred_v1 + streaming_v1 + row_subchunk_512 speed interaction",
    )
    build_wave3_interaction.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Output directory for Wave 3 interaction-confirmation scenario configs",
    )
    build_wave3_interaction.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    build_wave3_interaction.set_defaults(
        func=_cmd_build_wave3_interaction_confirmation_scenarios,
    )

    build_baseline_registry = subparsers.add_parser(
        "build-baseline-registry",
        help="Build a pinned baseline registry from completed scenario roots",
    )
    build_baseline_registry.add_argument(
        "--run-id",
        default="wave0-baseline-20260520-01",
        help="Run id under scratch cluster/tier roots",
    )
    build_baseline_registry.add_argument(
        "--registry-id",
        default=None,
        help="Optional registry id stored in the output JSON",
    )
    build_baseline_registry.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster to include when --all-clusters is not set",
    )
    build_baseline_registry.add_argument(
        "--all-clusters",
        action="store_true",
        help="Include both Ascend and Cardinal roots",
    )
    build_baseline_registry.add_argument(
        "--tier",
        choices=SCENARIO_TIERS,
        default="fast",
        help="Tier to include when --all-tiers is not set",
    )
    build_baseline_registry.add_argument(
        "--all-tiers",
        action="store_true",
        help="Include fast, anomaly, and long_eval roots",
    )
    build_baseline_registry.add_argument(
        "--run-root",
        action="append",
        default=None,
        help="Explicit completed run root; may be repeated. Overrides cluster/tier/run-id expansion.",
    )
    build_baseline_registry.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root containing cluster/tier/run-id outputs",
    )
    build_baseline_registry.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output registry JSON path",
    )
    build_baseline_registry.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project repo path recorded in registry provenance",
    )
    build_baseline_registry.add_argument(
        "--library-root",
        type=Path,
        default=REPO_ROOT.parent / "circuit-tracer_chunked",
        help="Sibling library repo path recorded in registry provenance",
    )
    build_baseline_registry.set_defaults(func=_cmd_build_baseline_registry)

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

    compare_phase3 = subparsers.add_parser(
        "compare-phase3-seed-bundles",
        help="Compare two saved Phase-3 seed bundle artifacts",
    )
    compare_phase3.add_argument("left_bundle", type=Path)
    compare_phase3.add_argument("right_bundle", type=Path)
    compare_phase3.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for comparison JSON",
    )
    compare_phase3.set_defaults(func=_cmd_compare_phase3_seed_bundles)

    compare_semantic = subparsers.add_parser(
        "compare-semantic-features",
        help="Compare two saved feature semantic descriptor artifacts",
    )
    compare_semantic.add_argument("left_descriptor", type=Path)
    compare_semantic.add_argument("right_descriptor", type=Path)
    compare_semantic.add_argument(
        "--position-window",
        type=int,
        default=0,
        help="Maximum position distance allowed for semantic substitute matching",
    )
    compare_semantic.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine threshold for high-confidence semantic substitute matches",
    )
    compare_semantic.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for comparison JSON",
    )
    compare_semantic.set_defaults(func=_cmd_compare_semantic_features)

    compare_phase0_replay = subparsers.add_parser(
        "compare-phase0-replay-matrix",
        help=(
            "Compare baseline/self-replay/cross-swap artifact roots and report "
            "self-replay gates + donor movement scores"
        ),
    )
    compare_phase0_replay.add_argument("ascend_baseline", type=Path)
    compare_phase0_replay.add_argument("cardinal_baseline", type=Path)
    compare_phase0_replay.add_argument("ascend_self_replay", type=Path)
    compare_phase0_replay.add_argument("cardinal_self_replay", type=Path)
    compare_phase0_replay.add_argument("ascend_with_cardinal", type=Path)
    compare_phase0_replay.add_argument("cardinal_with_ascend", type=Path)
    compare_phase0_replay.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for matrix comparison JSON",
    )
    compare_phase0_replay.set_defaults(func=_cmd_compare_phase0_replay_matrix)

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
        help="Override base scratch output root; launch-plan still appends a unique run folder",
    )
    launch_plan.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier used as the run folder name under --output-root",
    )
    launch_plan.add_argument(
        "--run-name",
        default=None,
        help="Human-readable run name used in metadata and output root slug defaults",
    )
    launch_plan.add_argument(
        "--run-description",
        default=None,
        help="Short free-text run description stored with the run artifacts",
    )
    launch_plan.add_argument(
        "--run-goal",
        default=None,
        help="Free-text run goal stored with the run artifacts",
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
    launch_plan.add_argument(
        "--baseline-registry",
        type=Path,
        default=None,
        help="Optional pinned baseline registry forwarded to runner jobs",
    )
    launch_plan.add_argument(
        "--fail-on-baseline-missing",
        action="store_true",
        help="Forward fail-on-missing-baseline behavior to runner jobs",
    )
    launch_plan.add_argument(
        "--fail-on-validation-fail",
        action="store_true",
        help="Forward fail-on-validation-fail behavior to runner jobs",
    )
    launch_plan.set_defaults(func=_cmd_launch_plan)

    fixture_prep = subparsers.add_parser(
        "submit-fixture-prep",
        help="Render or submit a SLURM job to prepare Wave 0 prompt fixtures",
    )
    fixture_prep.add_argument(
        "--cluster",
        choices=["ascend", "cardinal"],
        default="ascend",
        help="Cluster profile to use for fixture preparation",
    )
    fixture_prep.add_argument(
        "--target-spec-file",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_TARGET_SPEC,
        help="Fixture target spec JSON",
    )
    fixture_prep.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WAVE0_FIXTURE_OUTPUT_DIR,
        help="Directory where prepared fixtures and catalog will be written",
    )
    fixture_prep.add_argument(
        "--decoder-chunk-size",
        type=int,
        default=256,
        help="Decoder chunk size used when loading the tracing model",
    )
    fixture_prep.add_argument(
        "--cross-batch-decoder-cache-bytes",
        type=int,
        default=None,
        help="Optional decoder cache budget used for fixture prep model loading",
    )
    fixture_prep.add_argument(
        "--no-immutable-workspace",
        action="store_true",
        help="Run against the live workspace instead of a workspace snapshot",
    )
    fixture_prep.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Where immutable workspace snapshots are created",
    )
    fixture_prep.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace source root for immutable launch snapshots",
    )
    fixture_prep.add_argument("--workspace-label", default=None)
    fixture_prep.add_argument("--walltime", default=None)
    fixture_prep.add_argument(
        "--run-name",
        default=None,
        help="Human-readable label used for the SLURM job name",
    )
    fixture_prep.add_argument(
        "--print-only",
        action="store_true",
        help="Print the sbatch plan without submitting it",
    )
    fixture_prep.set_defaults(func=_cmd_submit_fixture_prep)

    submit_preset = subparsers.add_parser(
        "submit-preset",
        help="Generate scenarios and submit common benchmark presets",
    )
    submit_preset.add_argument("--preset", choices=preset_names(), required=True)
    submit_preset.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Directory where generated scenario JSON files are written",
    )
    submit_preset.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_FIXTURE_CATALOG,
        help="Fixture catalog JSON (fallback paths are used if missing)",
    )
    submit_preset.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
        help="Scratch root used for recommended output_root metadata",
    )
    submit_preset.add_argument(
        "--no-immutable-workspace",
        action="store_true",
        help="Submit against the live workspace instead of snapshotting by default",
    )
    submit_preset.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Where immutable workspace snapshots are created",
    )
    submit_preset.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace source root for immutable launch snapshots",
    )
    submit_preset.add_argument("--workspace-label-prefix", default="preset")
    submit_preset.add_argument("--walltime", default=None)
    submit_preset.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier to reuse across preset launch plans",
    )
    submit_preset.add_argument(
        "--run-name",
        default=None,
        help="Optional run name prefix for preset launch metadata",
    )
    submit_preset.add_argument(
        "--run-description",
        default=None,
        help="Optional run description for preset launch metadata",
    )
    submit_preset.add_argument(
        "--run-goal",
        default=None,
        help="Optional run goal for preset launch metadata",
    )
    submit_preset.add_argument(
        "--print-only",
        action="store_true",
        help="Print the plans without calling sbatch",
    )
    submit_preset.set_defaults(func=_cmd_submit_preset)

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
