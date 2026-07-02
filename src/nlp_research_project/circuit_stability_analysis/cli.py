from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import cluster_features
from .metrics import run_metrics
from .pairs import build_pair_manifest
from .profiles import build_feature_profiles
from .roles import build_roles, freeze_roles
from .summaries import summarize_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="circuit-stability-analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-roles")
    gen.add_argument("--tokens", type=Path, required=True)
    gen.add_argument("--trajectory-texts", type=Path)
    gen.add_argument("--output-dir", type=Path, required=True)
    freeze = sub.add_parser("freeze-roles-v1")
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--reviewed", type=Path)
    freeze.add_argument("--make-review-iter01", action="store_true")
    prof = sub.add_parser("extract-profiles")
    prof.add_argument("--tokens", type=Path, required=True)
    prof.add_argument("--roles", type=Path, required=True)
    prof.add_argument("--output-dir", type=Path, required=True)
    prof.add_argument("--max-graphs", type=int)
    prof.add_argument("--max-edges-per-bucket", type=int)
    prof.add_argument("--observation-top-k-per-graph", type=int, default=0)
    clus = sub.add_parser("cluster-features")
    clus.add_argument("--profiles-dir", type=Path)
    clus.add_argument("--output-dir", type=Path, required=True)
    clus.add_argument("--min-observation-count", type=int, default=1)
    clus.add_argument("--min-total-abs-mass", type=float, default=0.0)
    clus.add_argument("--top-n-per-layer", type=int, default=512)
    clus.add_argument("--neighbors", type=int, default=10)
    clus.add_argument("--clusters-per-layer", type=int, default=2)
    clus.add_argument("--min-affinity", type=float, default=0.0)
    clus.add_argument("--decoder-cache-dir", type=Path)
    clus.add_argument("--decoder-prior-weight", type=float, default=0.0)
    clus.add_argument("--seed", type=int, default=0)
    pairs = sub.add_parser("build-pair-manifest")
    pairs.add_argument("--tokens", type=Path, required=True)
    pairs.add_argument("--roles", type=Path)
    pairs.add_argument("--output", type=Path, required=True)
    pairs.add_argument("--role-version", default="unknown")
    pairs.add_argument("--cluster-version", default="unknown")
    pairs.add_argument("--lags", default="1,2,5,10,20")
    pairs.add_argument("--final-window", type=int, default=10)
    pairs.add_argument("--null-cap-per-trajectory", type=int, default=20)
    metrics = sub.add_parser("run-metrics")
    metrics.add_argument("--pair-manifest", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--cluster-manifest", type=Path)
    metrics.add_argument("--decoder-cache-dir", type=Path)
    metrics.add_argument(
        "--decoder-soft-thresholds",
        default="0.70,0.80,0.90",
    )
    metrics.add_argument("--max-pairs", type=int)
    summ = sub.add_parser("summarize-metrics")
    summ.add_argument("--metric-rows", type=Path, required=True)
    summ.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate-roles":
        build_roles(args.tokens, args.trajectory_texts, args.output_dir)
    elif args.command == "freeze-roles-v1":
        freeze_roles(args.output_dir, args.reviewed, args.make_review_iter01)
    elif args.command == "extract-profiles":
        build_feature_profiles(
            args.tokens,
            args.roles,
            args.output_dir,
            max_graphs=args.max_graphs,
            observation_top_k_per_graph=args.observation_top_k_per_graph,
            max_edges_per_bucket=args.max_edges_per_bucket,
        )
    elif args.command == "cluster-features":
        cluster_features(
            args.profiles_dir or args.output_dir,
            args.output_dir,
            min_observation_count=args.min_observation_count,
            min_total_abs_mass=args.min_total_abs_mass,
            top_n_per_layer=args.top_n_per_layer,
            neighbors=args.neighbors,
            clusters_per_layer=args.clusters_per_layer,
            min_affinity=args.min_affinity,
            decoder_cache_dir=args.decoder_cache_dir,
            decoder_prior_weight=args.decoder_prior_weight,
            seed=args.seed,
        )
    elif args.command == "build-pair-manifest":
        build_pair_manifest(
            args.tokens,
            args.output,
            roles=args.roles,
            role_version=args.role_version,
            cluster_version=args.cluster_version,
            lags=[int(x) for x in args.lags.split(",") if x],
            final_window=args.final_window,
            null_cap_per_trajectory=args.null_cap_per_trajectory,
        )
    elif args.command == "run-metrics":
        run_metrics(
            args.pair_manifest,
            args.output,
            cluster_manifest=args.cluster_manifest,
            decoder_cache_dir=args.decoder_cache_dir,
            thresholds=[float(x) for x in args.decoder_soft_thresholds.split(",") if x],
            max_pairs=args.max_pairs,
        )
    elif args.command == "summarize-metrics":
        summarize_metrics(args.metric_rows, args.output_dir)


if __name__ == "__main__":
    main()
