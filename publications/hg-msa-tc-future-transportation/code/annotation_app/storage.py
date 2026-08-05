"""Append-only SQLite storage, audit, backup, and immutable first-pass export."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from . import APP_VERSION
    from .models import (
        RAW_ANNOTATION_COLUMNS,
        manual_maneuver_id,
        validate_annotation_values,
    )
except ImportError:  # Direct script execution.
    from __init__ import APP_VERSION
    from models import RAW_ANNOTATION_COLUMNS, manual_maneuver_id, validate_annotation_values


SCHEMA_VERSION = "annotation-storage-v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize_database(
    path: Path, annotator_id: str, protocol_version: str, role: str = "annotator"
) -> None:
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                trajectory_id TEXT NOT NULL,
                split TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                entry_approach TEXT NOT NULL,
                exit_approach TEXT NOT NULL,
                manual_maneuver_id TEXT NOT NULL,
                maneuver_type TEXT NOT NULL,
                validity TEXT NOT NULL,
                confidence TEXT NOT NULL,
                annotator_id TEXT NOT NULL,
                annotation_timestamp_utc TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                trajectory_source_checksum TEXT NOT NULL,
                rendering_version TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                revision_number INTEGER NOT NULL,
                supersedes_annotation_id TEXT,
                is_active INTEGER NOT NULL CHECK(is_active IN (0, 1)),
                FOREIGN KEY(supersedes_annotation_id) REFERENCES annotations(annotation_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_annotation
                ON annotations(annotator_id, trajectory_id) WHERE is_active = 1;
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                trajectory_id TEXT,
                old_annotation_id TEXT,
                new_annotation_id TEXT,
                timestamp_utc TEXT NOT NULL,
                app_version TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                batch_id TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS queue_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                annotator_id TEXT NOT NULL,
                trajectory_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS queue_state (
                annotator_id TEXT PRIMARY KEY,
                queue_id TEXT NOT NULL,
                last_position INTEGER NOT NULL DEFAULT 0,
                updated_at_utc TEXT NOT NULL
            );
            """
        )
        existing = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        expected = {
            "schema_version": SCHEMA_VERSION,
            "annotator_id": annotator_id,
            "protocol_version": protocol_version,
            "role": role,
        }
        for key, value in expected.items():
            if key in existing and existing[key] != value:
                raise PermissionError(
                    f"Database identity mismatch for {key}: {existing[key]} != {value}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                (key, value),
            )


def database_identity(path: Path) -> dict[str, str]:
    with connect_database(path) as connection:
        return dict(connection.execute("SELECT key, value FROM metadata").fetchall())


def assert_database_identity(path: Path, annotator_id: str, protocol_version: str) -> None:
    identity = database_identity(path)
    if identity.get("annotator_id") != annotator_id:
        raise PermissionError("Annotator cannot open another annotator's database.")
    if identity.get("protocol_version") != protocol_version:
        raise PermissionError("Annotation database protocol version mismatch.")


def _audit(
    connection: sqlite3.Connection,
    user_id: str,
    action: str,
    protocol_version: str,
    trajectory_id: str | None = None,
    old_annotation_id: str | None = None,
    new_annotation_id: str | None = None,
    batch_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """INSERT INTO audit_log(
            user_id, action, trajectory_id, old_annotation_id, new_annotation_id,
            timestamp_utc, app_version, protocol_version, batch_id, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            action,
            trajectory_id,
            old_annotation_id,
            new_annotation_id,
            utc_timestamp(),
            APP_VERSION,
            protocol_version,
            batch_id,
            json.dumps(details or {}, sort_keys=True),
        ),
    )


def save_annotation(
    path: Path,
    queue_record: dict[str, Any],
    human_values: dict[str, Any],
    annotator_id: str,
    protocol_version: str,
    rendering_version: str,
    batch_id: str | None = None,
) -> str:
    assert_database_identity(path, annotator_id, protocol_version)
    if queue_record["annotator_id"] not in {annotator_id, "protocol_pilot"}:
        raise PermissionError("Queue belongs to a different annotator.")
    record = {
        "scene_id": str(queue_record["scene_id"]),
        "trajectory_id": str(queue_record["trajectory_id"]),
        "split": str(queue_record["split"]),
        "recording_id": str(queue_record["recording_id"]),
        "entry_approach": str(human_values["entry_approach"]),
        "exit_approach": str(human_values["exit_approach"]),
        "maneuver_type": str(human_values["maneuver_type"]),
        "validity": str(human_values["validity"]),
        "confidence": str(human_values["confidence"]),
        "notes": str(human_values.get("notes", "")),
    }
    record["manual_maneuver_id"] = manual_maneuver_id(
        record["scene_id"], record["entry_approach"], record["exit_approach"]
    )
    validate_annotation_values(record)
    annotation_id = str(uuid.uuid4())
    with connect_database(path) as connection:
        active = connection.execute(
            """SELECT annotation_id, revision_number FROM annotations
               WHERE annotator_id=? AND trajectory_id=? AND is_active=1""",
            (annotator_id, record["trajectory_id"]),
        ).fetchone()
        old_id = str(active["annotation_id"]) if active else None
        revision = int(active["revision_number"]) + 1 if active else 1
        if active:
            connection.execute(
                "UPDATE annotations SET is_active=0 WHERE annotation_id=?",
                (old_id,),
            )
        connection.execute(
            """INSERT INTO annotations(
                annotation_id, scene_id, trajectory_id, split, recording_id,
                entry_approach, exit_approach, manual_maneuver_id, maneuver_type,
                validity, confidence, annotator_id, annotation_timestamp_utc,
                protocol_version, trajectory_source_checksum, rendering_version,
                notes, revision_number, supersedes_annotation_id, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                annotation_id,
                record["scene_id"],
                record["trajectory_id"],
                record["split"],
                record["recording_id"],
                record["entry_approach"],
                record["exit_approach"],
                record["manual_maneuver_id"],
                record["maneuver_type"],
                record["validity"],
                record["confidence"],
                annotator_id,
                utc_timestamp(),
                protocol_version,
                str(queue_record["trajectory_source_checksum"]),
                rendering_version,
                record["notes"],
                revision,
                old_id,
            ),
        )
        _audit(
            connection,
            annotator_id,
            "save_annotation" if active is None else "revise_annotation",
            protocol_version,
            record["trajectory_id"],
            old_id,
            annotation_id,
            batch_id,
        )
    return annotation_id


def active_annotations(path: Path, annotator_id: str) -> pd.DataFrame:
    with connect_database(path) as connection:
        return pd.read_sql_query(
            """SELECT * FROM annotations
               WHERE annotator_id=? AND is_active=1 ORDER BY scene_id, trajectory_id""",
            connection,
            params=(annotator_id,),
        )


def annotation_history(path: Path, annotator_id: str, trajectory_id: str) -> pd.DataFrame:
    with connect_database(path) as connection:
        return pd.read_sql_query(
            """SELECT * FROM annotations WHERE annotator_id=? AND trajectory_id=?
               ORDER BY revision_number""",
            connection,
            params=(annotator_id, trajectory_id),
        )


def mark_queue_event(
    path: Path,
    annotator_id: str,
    trajectory_id: str,
    event_type: str,
    details: dict[str, Any] | None = None,
) -> None:
    with connect_database(path) as connection:
        connection.execute(
            """INSERT INTO queue_events(
                annotator_id, trajectory_id, event_type, timestamp_utc, details_json
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                annotator_id,
                trajectory_id,
                event_type,
                utc_timestamp(),
                json.dumps(details or {}, sort_keys=True),
            ),
        )


def record_batch_operation(
    path: Path,
    annotator_id: str,
    protocol_version: str,
    batch_id: str,
    trajectory_ids: list[str],
) -> None:
    assert_database_identity(path, annotator_id, protocol_version)
    with connect_database(path) as connection:
        _audit(
            connection,
            annotator_id,
            "manual_batch_assignment",
            protocol_version,
            batch_id=batch_id,
            details={"trajectory_ids": trajectory_ids, "count": len(trajectory_ids)},
        )


def undo_last_annotation(path: Path, annotator_id: str, protocol_version: str) -> str | None:
    """Deactivate the most recent revision and restore its predecessor, if any."""
    assert_database_identity(path, annotator_id, protocol_version)
    with connect_database(path) as connection:
        current = connection.execute(
            """SELECT annotation_id, trajectory_id, supersedes_annotation_id
               FROM annotations WHERE annotator_id=? AND is_active=1
               ORDER BY annotation_timestamp_utc DESC, rowid DESC LIMIT 1""",
            (annotator_id,),
        ).fetchone()
        if current is None:
            return None
        connection.execute(
            "UPDATE annotations SET is_active=0 WHERE annotation_id=?",
            (current["annotation_id"],),
        )
        restored = current["supersedes_annotation_id"]
        if restored:
            connection.execute(
                "UPDATE annotations SET is_active=1 WHERE annotation_id=?",
                (restored,),
            )
        _audit(
            connection,
            annotator_id,
            "undo_last_annotation",
            protocol_version,
            current["trajectory_id"],
            current["annotation_id"],
            restored,
        )
        return str(current["trajectory_id"])


def queue_state(path: Path, annotator_id: str) -> dict[str, Any] | None:
    with connect_database(path) as connection:
        row = connection.execute(
            "SELECT * FROM queue_state WHERE annotator_id=?", (annotator_id,)
        ).fetchone()
        return dict(row) if row else None


def update_queue_position(path: Path, annotator_id: str, queue_id: str, position: int) -> None:
    with connect_database(path) as connection:
        connection.execute(
            """INSERT INTO queue_state(annotator_id, queue_id, last_position, updated_at_utc)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(annotator_id) DO UPDATE SET
                 queue_id=excluded.queue_id,
                 last_position=excluded.last_position,
                 updated_at_utc=excluded.updated_at_utc""",
            (annotator_id, queue_id, int(position), utc_timestamp()),
        )


def backup_database(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / (f"{path.stem}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.sqlite")
    source_connection = connect_database(path)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()
    with sqlite3.connect(destination) as check:
        result = check.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise IOError(f"Backup integrity check failed: {result}")
    return destination


def export_first_pass(
    path: Path, annotator_id: str, output_csv: Path, audit_jsonl: Path
) -> tuple[Path, str]:
    if output_csv.exists() or audit_jsonl.exists():
        raise FileExistsError("First-pass exports are immutable and already exist.")
    annotations = active_annotations(path, annotator_id)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotations[list(RAW_ANNOTATION_COLUMNS)].to_csv(output_csv, index=False, lineterminator="\n")
    with connect_database(path) as connection:
        audit = pd.read_sql_query("SELECT * FROM audit_log ORDER BY audit_id", connection)
    with audit_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in audit.to_dict(orient="records"):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    checksum = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    output_csv.with_suffix(output_csv.suffix + ".sha256").write_text(
        f"{checksum}  {output_csv.name}\n", encoding="ascii"
    )
    return output_csv, checksum


def validate_storage_integrity(path: Path) -> dict[str, int]:
    with connect_database(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise IOError(f"SQLite integrity failure: {integrity}")
        duplicate_active = connection.execute(
            """SELECT COUNT(*) FROM (
                SELECT annotator_id, trajectory_id, COUNT(*) AS n
                FROM annotations WHERE is_active=1
                GROUP BY annotator_id, trajectory_id HAVING n > 1
            )"""
        ).fetchone()[0]
        return {
            "annotations": connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0],
            "active_annotations": connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE is_active=1"
            ).fetchone()[0],
            "duplicate_active": duplicate_active,
        }
