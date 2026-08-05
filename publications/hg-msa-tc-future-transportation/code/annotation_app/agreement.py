"""Post-annotation agreement, adjudication, consensus, and inventory utilities."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from .models import RAW_ANNOTATION_COLUMNS, validate_annotation_values
    from .storage import connect_database, initialize_database
except ImportError:  # Direct script execution.
    from models import RAW_ANNOTATION_COLUMNS, validate_annotation_values
    from storage import connect_database, initialize_database


AGREEMENT_FIELDS = (
    "manual_maneuver_id",
    "entry_approach",
    "exit_approach",
    "validity",
    "maneuver_type",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _kappa(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if mask.sum() == 0 or len(set(left[mask]) | set(right[mask])) < 2:
        return float("nan")
    return float(cohen_kappa_score(left[mask], right[mask]))


def agreement_analysis(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate agreement without modifying either immutable first-pass export."""
    required = {"scene_id", "trajectory_id", "confidence", *AGREEMENT_FIELDS}
    for name, frame in (("A", a), ("B", b)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Annotator {name} export is missing columns: {missing}")
        if frame["trajectory_id"].duplicated().any():
            raise ValueError(f"Annotator {name} export contains duplicate trajectories.")
    joined = a[list(required)].merge(
        b[list(required)],
        on=["scene_id", "trajectory_id"],
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
    )
    summary_rows = []
    for field in AGREEMENT_FIELDS:
        left, right = joined[f"{field}_A"], joined[f"{field}_B"]
        comparable = left.notna() & right.notna()
        summary_rows.append(
            {
                "field": field,
                "n_comparable": int(comparable.sum()),
                "raw_agreement_pct": float((left[comparable] == right[comparable]).mean() * 100)
                if comparable.any()
                else np.nan,
                "cohen_kappa": _kappa(left, right),
                "missing_A": int(left.isna().sum()),
                "missing_B": int(right.isna().sum()),
            }
        )
        joined[f"agree_{field}"] = left.eq(right) & comparable
    scene_rows = []
    for scene, part in joined.groupby("scene_id", dropna=False):
        for field in AGREEMENT_FIELDS:
            left, right = part[f"{field}_A"], part[f"{field}_B"]
            comparable = left.notna() & right.notna()
            scene_rows.append(
                {
                    "scene_id": scene,
                    "field": field,
                    "n": len(part),
                    "n_comparable": int(comparable.sum()),
                    "raw_agreement_pct": float((left[comparable] == right[comparable]).mean() * 100)
                    if comparable.any()
                    else np.nan,
                    "cohen_kappa": _kappa(left, right),
                }
            )
    by_scene = pd.DataFrame(scene_rows)
    confidence = joined.copy()
    confidence["confidence_stratum"] = (
        confidence["confidence_A"].fillna("missing")
        + "/"
        + confidence["confidence_B"].fillna("missing")
    )
    confidence = (
        confidence.groupby("confidence_stratum")
        .agg(
            n=("trajectory_id", "size"),
            exact_agreement_pct=("agree_manual_maneuver_id", lambda x: float(x.mean() * 100)),
        )
        .reset_index()
    )
    movement = (
        joined.groupby("manual_maneuver_id_A", dropna=False)
        .agg(
            n=("trajectory_id", "size"),
            exact_agreement_pct=("agree_manual_maneuver_id", lambda x: float(x.mean() * 100)),
        )
        .reset_index()
        .rename(columns={"manual_maneuver_id_A": "movement_class_A"})
    )
    matrix = pd.crosstab(
        joined["manual_maneuver_id_A"].fillna("<missing>"),
        joined["manual_maneuver_id_B"].fillna("<missing>"),
        dropna=False,
    )
    disagreements = joined[
        (joined["_merge"] != "both")
        | (~joined["agree_manual_maneuver_id"])
        | (~joined["agree_validity"])
        | joined["confidence_A"].eq("low")
        | joined["confidence_B"].eq("low")
    ].copy()
    return {
        "summary": pd.DataFrame(summary_rows),
        "by_scene": by_scene,
        "by_movement_class": movement,
        "confidence_strata": confidence,
        "disagreement_matrix": matrix,
        "disagreements": disagreements,
        "joined": joined,
    }


def initialize_adjudication_database(path: Path, protocol_version: str) -> None:
    initialize_database(path, "adjudicator", protocol_version, role="adjudicator")
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS adjudications (
                adjudication_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                trajectory_id TEXT NOT NULL UNIQUE,
                annotation_id_A TEXT NOT NULL,
                annotation_id_B TEXT NOT NULL,
                raw_label_A_json TEXT NOT NULL,
                raw_label_B_json TEXT NOT NULL,
                consensus_json TEXT NOT NULL,
                adjudicator_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                protocol_version TEXT NOT NULL
            );
            """
        )


def save_adjudication(
    path: Path,
    raw_a: dict[str, Any],
    raw_b: dict[str, Any],
    consensus: dict[str, Any],
    adjudicator_id: str,
    protocol_version: str,
    reason: str,
) -> str:
    validate_annotation_values(consensus)
    if (
        raw_a["trajectory_id"] != raw_b["trajectory_id"]
        or raw_a["trajectory_id"] != consensus["trajectory_id"]
    ):
        raise ValueError("Adjudication records must refer to one trajectory.")
    adjudication_id = str(uuid.uuid4())
    with connect_database(path) as connection:
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key='protocol_version'"
        ).fetchone()
        if existing is None or existing[0] != protocol_version:
            raise PermissionError("Adjudication protocol version mismatch.")
        connection.execute(
            """INSERT INTO adjudications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                adjudication_id,
                consensus["scene_id"],
                consensus["trajectory_id"],
                raw_a["annotation_id"],
                raw_b["annotation_id"],
                json.dumps(raw_a, sort_keys=True),
                json.dumps(raw_b, sort_keys=True),
                json.dumps(consensus, sort_keys=True),
                adjudicator_id,
                reason,
                utc_timestamp(),
                protocol_version,
            ),
        )
    return adjudication_id


def export_consensus(
    path: Path,
    output_csv: Path,
    raw_a_csv: Path | None = None,
    raw_b_csv: Path | None = None,
) -> tuple[Path, str]:
    if output_csv.exists():
        raise FileExistsError("Consensus export is immutable and already exists.")
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM adjudications ORDER BY scene_id, trajectory_id", connection
        )
    adjudicated = {
        row.trajectory_id: json.loads(row.consensus_json) for row in frame.itertuples(index=False)
    }
    if raw_a_csv is None or raw_b_csv is None:
        output = pd.DataFrame(list(adjudicated.values()))
    else:
        a = pd.read_csv(raw_a_csv, keep_default_na=False)
        b = pd.read_csv(raw_b_csv, keep_default_na=False)
        joined = a.merge(
            b, on=["scene_id", "trajectory_id"], suffixes=("_A", "_B"), validate="one_to_one"
        )
        rows = []
        decision_fields = (*AGREEMENT_FIELDS, "confidence")
        for row in joined.to_dict(orient="records"):
            trajectory_id = row["trajectory_id"]
            needs_adjudication = (
                any(row[f"{field}_A"] != row[f"{field}_B"] for field in decision_fields)
                or row["confidence_A"] == "low"
                or row["confidence_B"] == "low"
            )
            if needs_adjudication:
                if trajectory_id not in adjudicated:
                    raise ValueError(f"Unresolved adjudication case: {trajectory_id}")
                rows.append(adjudicated[trajectory_id])
            else:
                rows.append(
                    {
                        column: row.get(f"{column}_A", row.get(column, ""))
                        for column in RAW_ANNOTATION_COLUMNS
                    }
                )
        output = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, lineterminator="\n")
    checksum = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    output_csv.with_suffix(output_csv.suffix + ".sha256").write_text(
        f"{checksum}  {output_csv.name}\n", encoding="ascii"
    )
    return output_csv, checksum


def build_manual_inventory(
    consensus: pd.DataFrame, rare_threshold: float = 0.01
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive rarity only after consensus; never compares labels with HG outputs."""
    if not 0 <= rare_threshold <= 1:
        raise ValueError("rare_threshold must be in [0, 1].")
    required = {
        "scene_id",
        "manual_maneuver_id",
        "entry_approach",
        "exit_approach",
        "maneuver_type",
        "validity",
    }
    missing = sorted(required.difference(consensus.columns))
    if missing:
        raise ValueError(f"Consensus is missing inventory columns: {missing}")
    valid = consensus[consensus["validity"] == "valid"].copy()
    counts = (
        valid.groupby(
            ["scene_id", "manual_maneuver_id", "entry_approach", "exit_approach", "maneuver_type"]
        )
        .size()
        .rename("valid_trajectory_count")
        .reset_index()
    )
    totals = valid.groupby("scene_id").size().rename("scene_valid_total")
    counts = counts.join(totals, on="scene_id")
    counts["percentage"] = (
        100 * counts["valid_trajectory_count"] / counts["scene_valid_total"].clip(lower=1)
    )
    counts["rare_movement"] = (
        counts["valid_trajectory_count"] / counts["scene_valid_total"].clip(lower=1)
        < rare_threshold
    )
    quality = consensus.groupby(["scene_id", "validity"]).size().unstack(fill_value=0).reset_index()
    quality["observed_manual_maneuver_count"] = (
        quality["scene_id"].map(counts.groupby("scene_id").size()).fillna(0).astype(int)
    )
    return counts, quality
