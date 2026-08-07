"""Generate EMAS_HG-v1 sensitivity figures and scientific documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from emas_sensitivity.reporting import run_reporting  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("figures", "documents"))
    arguments = parser.parse_args()
    run_reporting(arguments.command)


if __name__ == "__main__":
    main()
