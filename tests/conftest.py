import sys
from pathlib import Path

# Ensure the package under ./src is importable when running tests locally
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
