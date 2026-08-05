"""Command-line entry points for the isolated manual-annotation workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import cv2
import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agreement import (  # noqa: E402
    agreement_analysis,
    build_manual_inventory,
    export_consensus,
    initialize_adjudication_database,
)
from models import SCENES  # noqa: E402
from protocol import (  # noqa: E402
    freeze_protocol,
    protocol_is_frozen,
    require_frozen_protocol,
)
from quality import EXPECTED_TEST_COUNTS, validate_queue_sources, write_preflight  # noqa: E402
from queues import (  # noqa: E402
    primary_annotator_queue,
    recording_stratified_pilot,
    validate_primary_queues,
    write_queue,
)
from rendering import render_scene_guide_from_yaml, render_trajectory  # noqa: E402
from source_data import (  # noqa: E402
    attach_sources,
    build_source_catalog,
    load_manifest_and_split,
    load_full_polyline,
    read_video_frame,
    sha256_file,
    write_annotation_access_log,
)
from storage import (  # noqa: E402
    backup_database,
    export_first_pass,
    initialize_database,
    validate_storage_integrity,
)


APP_DIR = Path(__file__).resolve().parent
PUBLICATION_ROOT = APP_DIR.parents[1]
REPO_ROOT = APP_DIR.parents[3]
ANNOTATIONS = PUBLICATION_ROOT / "annotations"
MANIFEST = PUBLICATION_ROOT / "data" / "manifests" / "trajectory_manifest.csv"
SPLIT = PUBLICATION_ROOT / "data" / "splits" / "evaluation_split.csv"
SOURCE_CATALOG = ANNOTATIONS / "provenance" / "trajectory_source_catalog.csv"
ACCESS_LOG = ANNOTATIONS / "provenance" / "annotation_data_access.jsonl"
SCIENTIFIC_FREEZE = PUBLICATION_ROOT / "results" / "development" / "frozen_selection_manifest.json"


def _source_index() -> pd.DataFrame:
    index = load_manifest_and_split(MANIFEST, SPLIT)
    if SOURCE_CATALOG.exists():
        catalog = pd.read_csv(SOURCE_CATALOG, keep_default_na=False)
    else:
        catalog = build_source_catalog(REPO_ROOT, index)
        SOURCE_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(SOURCE_CATALOG, index=False, lineterminator="\n")
    return attach_sources(index, catalog)


def command_build_catalog(_: argparse.Namespace) -> None:
    index = load_manifest_and_split(MANIFEST, SPLIT)
    catalog = build_source_catalog(REPO_ROOT, index)
    SOURCE_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(SOURCE_CATALOG, index=False, lineterminator="\n")
    print(f"Wrote {len(catalog)} recording sources to {SOURCE_CATALOG}")


def command_build_pilot(_: argparse.Namespace) -> None:
    queue = recording_stratified_pilot(_source_index())
    path = ANNOTATIONS / "queues" / "protocol_pilot_queue.csv"
    write_queue(queue, path)
    print(queue.groupby("scene_id").size().to_string())
    print(f"Wrote {len(queue)} rows to {path}")


def command_build_primary(_: argparse.Namespace) -> None:
    frozen = protocol_is_frozen(ANNOTATIONS / "protocol")
    source = _source_index()
    queues = {}
    for annotator in ("annotator_A", "annotator_B"):
        queue = primary_annotator_queue(source, annotator, protocol_frozen=frozen)
        path = ANNOTATIONS / "queues" / f"independent_test_{annotator}.csv"
        write_queue(queue, path)
        queues[annotator] = queue
    validate_primary_queues(queues["annotator_A"], queues["annotator_B"], EXPECTED_TEST_COUNTS)
    print(
        f"Protocol frozen: {frozen}; queue status: {queues['annotator_A']['queue_status'].iat[0]}"
    )
    print(queues["annotator_A"].groupby("scene_id").size().to_string())


def command_prepare_guides(_: argparse.Namespace) -> None:
    catalog = pd.read_csv(SOURCE_CATALOG, keep_default_na=False)
    guides = ANNOTATIONS / "protocol" / "scene_guides"
    guides.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        row = catalog[catalog["scene_id"] == scene].sort_values("recording_id").iloc[0]
        video_path = REPO_ROOT / row["source_recording_file"]
        frame_number = 300
        frame = read_video_frame(video_path, frame_number)
        frame_path = guides / f"{scene}_representative_frame.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        guide_path = guides / f"{scene}.yaml"
        if not guide_path.exists():
            payload = {
                "scene_id": scene,
                "status": "draft_manual_configuration_required",
                "representative_frame": f"protocol/scene_guides/{frame_path.name}",
                "representative_recording_id": row["recording_id"],
                "representative_frame_number": frame_number,
                "approaches": [],
                "valid_entry_approaches": [],
                "valid_exit_approaches": [],
                "maneuver_type_mapping": [],
                "ambiguity_notes": "[MANUAL INPUT REQUIRED]",
            }
            guide_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        render_scene_guide_from_yaml(guide_path, ANNOTATIONS)
        write_annotation_access_log(
            ACCESS_LOG,
            "extract_representative_scene_frame",
            scene,
            row["source_recording_file"],
            1,
            "video_sha256_deferred_until_annotation_use",
        )
        print(f"Prepared draft scene guide: {guide_path}")


def command_render_previews(_: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    pilot = pd.read_csv(ANNOTATIONS / "queues" / "protocol_pilot_queue.csv", keep_default_na=False)
    output_dir = ANNOTATIONS / "preflight" / "previews"
    for scene in SCENES:
        row = pilot[pilot["scene_id"] == scene].iloc[0]
        polyline = load_full_polyline(
            REPO_ROOT,
            row["trajectory_source_path"],
            int(row["source_recording_track_id"]),
            row["trajectory_source_checksum"],
            int(row["frame_start"]),
            int(row["frame_end"]),
        )
        background = read_video_frame(
            REPO_ROOT / row["source_recording_file"],
            int(polyline["frame"].iloc[len(polyline) // 2]),
        )
        path = output_dir / f"{scene}_pilot_trajectory.png"
        figure = render_trajectory(
            polyline, background, output_path=path, title=f"Non-labeling pilot preview: {scene}"
        )
        plt.close(figure)
        write_annotation_access_log(
            ACCESS_LOG,
            "render_non_labeling_pilot_preview",
            scene,
            row["trajectory_source_path"],
            len(polyline),
            row["trajectory_source_checksum"],
        )
        print(f"Rendered {path}")


def command_freeze(_: argparse.Namespace) -> None:
    path, checksum = freeze_protocol(
        ANNOTATIONS / "protocol",
        SCIENTIFIC_FREEZE,
        ANNOTATIONS / "protocol" / "annotation_guideline.md",
    )
    print(f"Frozen {path}: {checksum}")


def command_preflight(_: argparse.Namespace) -> None:
    queue = pd.read_csv(
        ANNOTATIONS / "queues" / "independent_test_annotator_A.csv", keep_default_na=False
    )
    result = validate_queue_sources(REPO_ROOT, queue, ACCESS_LOG)
    summary = write_preflight(
        result,
        ANNOTATIONS / "preflight" / "independent_test_renderability.csv",
        ANNOTATIONS / "preflight" / "independent_test_renderability_summary.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_init_database(args: argparse.Namespace) -> None:
    protocol = require_frozen_protocol(ANNOTATIONS / "protocol")
    database = ANNOTATIONS / "databases" / f"{args.identity}.sqlite"
    if args.identity == "adjudication":
        initialize_adjudication_database(database, protocol["protocol_version"])
    else:
        initialize_database(database, args.identity, protocol["protocol_version"])
    print(f"Initialized empty database: {database}")


def command_backup(args: argparse.Namespace) -> None:
    database = ANNOTATIONS / "databases" / f"{args.identity}.sqlite"
    backup = backup_database(database, ANNOTATIONS / "backups" / args.identity)
    print(f"Backup: {backup}; integrity: {validate_storage_integrity(backup)}")


def command_export(args: argparse.Namespace) -> None:
    database = ANNOTATIONS / "databases" / f"{args.annotator}.sqlite"
    export = ANNOTATIONS / "exports" / f"independent_test_{args.annotator}.csv"
    audit = ANNOTATIONS / "exports" / f"independent_test_{args.annotator}_audit.jsonl"
    from storage import active_annotations

    queue = pd.read_csv(ANNOTATIONS / "queues" / f"independent_test_{args.annotator}.csv")
    active = active_annotations(database, args.annotator)
    if set(active["trajectory_id"]) != set(queue["trajectory_id"]):
        raise ValueError(f"First-pass export requires a complete queue: {len(active)}/{len(queue)}")
    _, checksum = export_first_pass(database, args.annotator, export, audit)
    print(f"Exported {export}: {checksum}")


def command_agreement(_: argparse.Namespace) -> None:
    exports = ANNOTATIONS / "exports"
    a = pd.read_csv(exports / "independent_test_annotator_A.csv")
    b = pd.read_csv(exports / "independent_test_annotator_B.csv")
    outputs = agreement_analysis(a, b)
    report_dir = ANNOTATIONS / "reports" / "agreement"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        if name != "joined":
            frame.to_csv(report_dir / f"{name}.csv", index=True, lineterminator="\n")
    (report_dir / "agreement_summary.md").write_text(
        "# Independent Annotator Agreement\n\n"
        + outputs["summary"].to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(outputs["summary"].to_string(index=False))


def command_consensus(_: argparse.Namespace) -> None:
    exports = ANNOTATIONS / "exports"
    path, checksum = export_consensus(
        ANNOTATIONS / "databases" / "adjudication.sqlite",
        exports / "independent_test_consensus.csv",
        exports / "independent_test_annotator_A.csv",
        exports / "independent_test_annotator_B.csv",
    )
    print(f"Exported {path}: {checksum}")


def command_inventory(args: argparse.Namespace) -> None:
    consensus = pd.read_csv(ANNOTATIONS / "exports" / "independent_test_consensus.csv")
    inventory, distribution = build_manual_inventory(consensus, args.rare_threshold)
    report_dir = ANNOTATIONS / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(report_dir / "manual_maneuver_inventory.csv", index=False, lineterminator="\n")
    distribution.to_csv(
        report_dir / "manual_label_distribution.csv", index=False, lineterminator="\n"
    )
    (report_dir / "manual_maneuver_inventory.md").write_text(
        "# Manual Observed-Maneuver Inventory\n\n"
        f"Derived rare-movement threshold: {args.rare_threshold:.3%}.\n\n"
        + inventory.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote derived inventory with threshold {args.rare_threshold:.3%}")


def command_verify(args: argparse.Namespace) -> None:
    path = Path(args.path)
    sidecar = Path(args.sidecar) if args.sidecar else path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {actual} != {expected}")
    print(f"OK {actual}  {path}")


def command_progress(args: argparse.Namespace) -> None:
    queue = pd.read_csv(ANNOTATIONS / "queues" / f"independent_test_{args.annotator}.csv")
    database = ANNOTATIONS / "databases" / f"{args.annotator}.sqlite"
    from storage import active_annotations

    completed = (
        active_annotations(database, args.annotator)
        if database.exists()
        else pd.DataFrame(columns=["scene_id"])
    )
    elapsed = max(args.elapsed_hours, 0.0)
    rate = len(completed) / elapsed if elapsed > 0 else 0.0
    remaining = len(queue) - len(completed)
    print(f"Completed: {len(completed)}/{len(queue)}; rate: {rate:.1f} annotations/hour")
    print(
        f"Estimated remaining hours: {remaining / rate:.1f}"
        if rate
        else "Estimated remaining hours: unavailable until elapsed time is supplied"
    )
    if rate and args.daily_hours > 0:
        days = math.ceil((remaining / rate) / args.daily_hours)
        print(f"Projected completion date: {date.today() + timedelta(days=days)}")
    for scene in SCENES:
        print(
            f"{scene}: {(completed['scene_id'] == scene).sum()}/{(queue['scene_id'] == scene).sum()}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    commands = {
        "build-source-catalog": command_build_catalog,
        "build-pilot-queue": command_build_pilot,
        "build-primary-queues": command_build_primary,
        "prepare-scene-guides": command_prepare_guides,
        "render-previews": command_render_previews,
        "freeze-protocol": command_freeze,
        "preflight": command_preflight,
        "agreement": command_agreement,
        "export-consensus": command_consensus,
    }
    for name, function in commands.items():
        sub.add_parser(name).set_defaults(function=function)
    p = sub.add_parser("init-database")
    p.add_argument("identity", choices=["annotator_A", "annotator_B", "adjudication"])
    p.set_defaults(function=command_init_database)
    p = sub.add_parser("backup")
    p.add_argument("identity", choices=["annotator_A", "annotator_B", "adjudication"])
    p.set_defaults(function=command_backup)
    p = sub.add_parser("export-first-pass")
    p.add_argument("annotator", choices=["annotator_A", "annotator_B"])
    p.set_defaults(function=command_export)
    p = sub.add_parser("inventory")
    p.add_argument("--rare-threshold", type=float, default=0.01)
    p.set_defaults(function=command_inventory)
    p = sub.add_parser("verify-checksum")
    p.add_argument("path")
    p.add_argument("--sidecar")
    p.set_defaults(function=command_verify)
    p = sub.add_parser("progress")
    p.add_argument("annotator", choices=["annotator_A", "annotator_B"])
    p.add_argument("--elapsed-hours", type=float, default=0.0)
    p.add_argument("--daily-hours", type=float, default=4.0)
    p.set_defaults(function=command_progress)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
