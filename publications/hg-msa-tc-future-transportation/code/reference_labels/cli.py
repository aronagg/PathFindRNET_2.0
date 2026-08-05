"""Command-line entry point for polygon-rule reference-label generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .generator import generate_primary_reference_dataset
    from .protocol import freeze_polygon_protocol, load_frozen_protocol
    from .reporting import (
        build_qa_audit_queue,
        generate_qc_figures,
        run_sensitivity_analysis,
        write_inventory_and_quality_reports,
        write_qa_statement,
    )
except ImportError:  # Direct script execution.
    from generator import generate_primary_reference_dataset
    from protocol import freeze_polygon_protocol, load_frozen_protocol
    from reporting import (
        build_qa_audit_queue,
        generate_qc_figures,
        run_sensitivity_analysis,
        write_inventory_and_quality_reports,
        write_qa_statement,
    )


def _roots() -> tuple[Path, Path]:
    publication_root = Path(__file__).resolve().parents[2]
    repo_root = publication_root.parents[1]
    return repo_root, publication_root


def _paths(publication_root: Path) -> dict[str, Path]:
    reference_dir = publication_root / "annotations/reference_labels"
    protocol_dir = publication_root / "annotations/protocol"
    return {
        "protocol": protocol_dir / "polygon_reference_protocol_v1.yaml",
        "reference_dir": reference_dir,
        "labels": reference_dir / "polygon_rule_reference_labels_all.parquet",
        "docs": publication_root / "docs",
        "figures": reference_dir / "figures",
    }


def _load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Generate primary labels first: {path}")
    return pd.read_parquet(path)


def freeze(repo_root: Path, publication_root: Path, paths: dict[str, Path]) -> None:
    freeze_polygon_protocol(repo_root, publication_root, paths["protocol"])


def generate(repo_root: Path, publication_root: Path, paths: dict[str, Path]) -> pd.DataFrame:
    labels, _ = generate_primary_reference_dataset(repo_root, publication_root, paths["protocol"])
    return labels


def inventories(publication_root: Path, paths: dict[str, Path], labels: pd.DataFrame) -> None:
    protocol = load_frozen_protocol(paths["protocol"])
    write_inventory_and_quality_reports(labels, protocol, paths["reference_dir"], paths["docs"])


def sensitivity(publication_root: Path, paths: dict[str, Path], labels: pd.DataFrame) -> None:
    protocol = load_frozen_protocol(paths["protocol"])
    run_sensitivity_analysis(
        labels,
        publication_root,
        protocol,
        paths["reference_dir"] / "polygon_assignment_sensitivity.csv",
        paths["docs"] / "polygon_assignment_sensitivity_report.md",
    )


def figures(publication_root: Path, paths: dict[str, Path], labels: pd.DataFrame) -> None:
    protocol = load_frozen_protocol(paths["protocol"])
    generate_qc_figures(labels, publication_root, protocol, paths["figures"])
    queue = build_qa_audit_queue(labels, protocol, paths["reference_dir"] / "qa_audit_queue.csv")
    write_qa_statement(paths["reference_dir"] / "qa_audit_queue_report.md", queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "generate", "inventories", "sensitivity", "figures", "all"),
    )
    args = parser.parse_args()
    repo_root, publication_root = _roots()
    paths = _paths(publication_root)
    paths["reference_dir"].mkdir(parents=True, exist_ok=True)
    paths["docs"].mkdir(parents=True, exist_ok=True)

    if args.command == "freeze":
        freeze(repo_root, publication_root, paths)
        return

    if not paths["protocol"].exists():
        raise FileNotFoundError("Freeze the polygon protocol before generating labels.")

    labels = (
        generate(repo_root, publication_root, paths)
        if args.command
        in {
            "generate",
            "all",
        }
        else _load_labels(paths["labels"])
    )
    if args.command in {"inventories", "all"}:
        inventories(publication_root, paths, labels)
    if args.command in {"sensitivity", "all"}:
        sensitivity(publication_root, paths, labels)
    if args.command in {"figures", "all"}:
        figures(publication_root, paths, labels)


if __name__ == "__main__":
    main()
