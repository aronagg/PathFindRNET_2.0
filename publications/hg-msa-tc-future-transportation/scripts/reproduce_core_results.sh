#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="publications/hg-msa-tc-future-transportation/code"

echo "Regenerating final synthesis tables and figures from persisted outputs only..."
python publications/hg-msa-tc-future-transportation/code/final_synthesis.py

if [[ "${1:-}" == "--run-tests" ]]; then
  python -m pytest publications/hg-msa-tc-future-transportation/tests -q
fi

echo "Done. No clustering, target estimation, homography calibration, baseline generation, or reference-label generation was run."
