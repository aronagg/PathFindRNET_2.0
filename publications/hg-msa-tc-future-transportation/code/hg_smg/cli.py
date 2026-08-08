"""Command-line interface for development-only HG-SMG-TC execution."""

from __future__ import annotations

import argparse

from . import runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    full = subparsers.add_parser("full-split")
    full.add_argument("--jobs", type=int, default=1)
    uatp = subparsers.add_parser("uatp")
    uatp.add_argument("--replicates", type=int, default=500)
    uatp.add_argument("--sac-replicates", type=int, default=500)
    uatp.add_argument("--jobs", type=int, default=1)
    uatp.add_argument("--replicate-jobs", type=int, default=1)
    uatp.add_argument(
        "--variants",
        default="A5",
        help="Comma-separated executable variant IDs.",
    )
    subparsers.add_parser("pcms")
    determinism = subparsers.add_parser("determinism")
    determinism.add_argument("--replicates", type=int, default=500)
    determinism.add_argument("--sac-replicates", type=int, default=500)
    determinism.add_argument("--replicate-jobs", type=int, default=1)
    subparsers.add_parser("report")
    future = subparsers.add_parser("future-test")
    future.add_argument("--confirmation")
    locked = subparsers.add_parser("locked-test")
    locked_sub = locked.add_subparsers(dest="locked_command", required=True)
    locked_sub.add_parser("preflight")
    locked_sub.add_parser("authorize")
    assign = locked_sub.add_parser("assign")
    assign.add_argument("--confirmation")
    locked_sub.add_parser("evaluate")
    locked_sub.add_parser("figures")
    locked_sub.add_parser("reports")
    package = locked_sub.add_parser("package")
    package.add_argument("--test-result", default="not_run")
    all_cmd = locked_sub.add_parser("all")
    all_cmd.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        payload = runner.preflight()
        print(payload["actual_protocol_sha256"])
    elif args.command == "full-split":
        runner.run_full_split(args.jobs)
        print(runner.RESULTS_ROOT)
    elif args.command == "uatp":
        variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
        runner.run_uatp(
            args.replicates,
            args.sac_replicates,
            args.jobs,
            variants,
            args.replicate_jobs,
        )
        print(runner.RESULTS_ROOT / "uatp_summary.csv")
    elif args.command == "pcms":
        runner.run_pcms()
        print(runner.RESULTS_ROOT / "pcms_selected_configurations.csv")
    elif args.command == "determinism":
        print(
            runner.run_determinism_check(
                args.replicates, args.sac_replicates, args.replicate_jobs
            )
        )
    elif args.command == "report":
        import json

        from . import reporting

        summary_path = runner.RESULTS_ROOT / "determinism_summary.json"
        determinism = json.loads(summary_path.read_text(encoding="utf-8"))
        reporting.generate_figures()
        reporting.write_reports(determinism)
        print(reporting.freeze_development(runner.git_head(), determinism))
    elif args.command == "future-test":
        runner.future_test_gate(args.confirmation)
    elif args.command == "locked-test":
        from . import locked_test

        if args.locked_command == "preflight":
            print(locked_test.run_preflight()["status"])
        elif args.locked_command == "authorize":
            print(locked_test.authorize_locked_test()["timestamp_utc"])
        elif args.locked_command == "assign":
            print(locked_test.run_assignments(args.confirmation)["assignment_rows"])
        elif args.locked_command == "evaluate":
            print(locked_test.run_evaluation()["evaluation_timestamp_utc"])
        elif args.locked_command == "figures":
            print(len(locked_test.generate_figures()))
        elif args.locked_command == "reports":
            locked_test.write_reports()
            print("reports")
        elif args.locked_command == "package":
            print(locked_test.package_outputs(args.test_result)["zip_sha256"])
        elif args.locked_command == "all":
            print(locked_test.run_all(args.confirmation)["zip_sha256"])


if __name__ == "__main__":
    main()
