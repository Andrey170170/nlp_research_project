from __future__ import annotations

import argparse
from pathlib import Path

from .campaign import run_scenario_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one persisted exact-trace scenario")
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_scenario_file(args.scenario_file, args.output_dir)


if __name__ == "__main__":
    main()
