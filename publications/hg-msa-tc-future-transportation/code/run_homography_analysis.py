"""Run Task 08 frozen homography quality and perturbation analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from homography.analysis import (  # noqa: E402
    DEFAULT_PERTURBATION_REPLICATES,
    default_paths,
    run_all,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze",))
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_PERTURBATION_REPLICATES,
        help="Deterministic perturbation replicates per scene and noise scale.",
    )
    arguments = parser.parse_args()
    print(json.dumps(run_all(default_paths(), arguments.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
