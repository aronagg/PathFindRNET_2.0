"""Generate Task 07 target-estimation figures and scientific documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from target_estimation.analysis import default_paths  # noqa: E402
from target_estimation.reporting import run_documents, run_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("figures", "documents", "all"))
    arguments = parser.parse_args()
    paths = default_paths()
    result = {}
    if arguments.command in {"figures", "all"}:
        result.update(run_figures(paths))
    if arguments.command in {"documents", "all"}:
        result.update(run_documents(paths))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
