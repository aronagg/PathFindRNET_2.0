"""Generate Task 08 homography figures and scientific documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from homography.analysis import default_paths  # noqa: E402
from homography.reporting import (  # noqa: E402
    run_all_reporting,
    run_documents,
    run_figures,
    update_publication_docs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("figures", "documents", "all"))
    arguments = parser.parse_args()
    paths = default_paths()
    if arguments.command == "figures":
        result = run_figures(paths)
    elif arguments.command == "documents":
        result = {**run_documents(paths), **update_publication_docs(paths)}
    else:
        result = run_all_reporting(paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
