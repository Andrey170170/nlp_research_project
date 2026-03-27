from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the number of scenarios in a JSON file"
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        required=True,
        help="Scenario JSON file with a top-level scenarios list",
    )
    args = parser.parse_args()

    payload = json.loads(args.scenarios_file.read_text())
    print(len(payload.get("scenarios", [])))


if __name__ == "__main__":
    main()
