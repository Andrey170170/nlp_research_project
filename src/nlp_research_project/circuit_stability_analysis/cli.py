from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args(argv)
    if args.command == "generate-roles":
        build_roles(args.tokens, args.trajectory_texts, args.output_dir)
    elif args.command == "freeze-roles-v1":
        freeze_roles(args.output_dir, args.reviewed, args.make_review_iter01)


if __name__ == "__main__":
    main()
