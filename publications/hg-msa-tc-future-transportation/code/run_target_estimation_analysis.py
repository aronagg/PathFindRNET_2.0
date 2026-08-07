"""Run Task 07 frozen HG target reproduction and diagnostic failure analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from target_estimation.analysis import default_paths, run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("analyze",),
        help="Reproduce frozen targets, then run explicitly post-freeze diagnostics.",
    )
    arguments = parser.parse_args()
    if arguments.command == "analyze":
        print(json.dumps(run_all(default_paths()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
