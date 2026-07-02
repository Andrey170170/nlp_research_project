from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import cluster_features
from .profiles import build_feature_profiles
from .roles import build_roles, freeze_roles


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


if __name__ == "__main__":
    main()
