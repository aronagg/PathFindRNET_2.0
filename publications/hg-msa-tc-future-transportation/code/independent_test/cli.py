"""CLI for locked independent-test execution and separate evaluation."""

from __future__ import annotations

import argparse

from .protocol import create_unlock, default_paths, run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "unlock", "cluster", "evaluate", "sensitivity", "figures", "reports"),
    )
    parser.add_argument("--confirm-independent-test-evaluation", action="store_true")
    args = parser.parse_args()
    paths = default_paths()

    if args.command == "preflight":
        run_preflight(paths, write_report=True)
    elif args.command == "unlock":
        create_unlock(paths, run_preflight(paths, write_report=False))
    elif args.command == "cluster":
        from .runner import run_frozen_clustering

        run_frozen_clustering(paths, args.confirm_independent_test_evaluation)
    elif args.command == "evaluate":
        from .evaluator import run_primary_evaluation

        run_primary_evaluation(paths)
    elif args.command == "sensitivity":
        from .evaluator import run_reference_sensitivity

        run_reference_sensitivity(paths)
    elif args.command == "figures":
        from .evaluator import create_figures

        create_figures(paths)
    elif args.command == "reports":
        from .reporting import write_all_reports

        write_all_reports(paths)


if __name__ == "__main__":
    main()
