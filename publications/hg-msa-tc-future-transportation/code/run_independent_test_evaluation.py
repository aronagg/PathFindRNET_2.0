"""Repository-root wrapper for the locked independent-test CLI."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from independent_test.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
