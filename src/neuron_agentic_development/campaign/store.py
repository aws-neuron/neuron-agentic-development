"""Append-only SQLite event storage and deterministic state projection."""

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    ConflictError,
    IntegrityError,
    NotFoundError,
    canonical_json,
    json_digest,
)


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CampaignStore:
    """The only authoritative campaign-state writer."""

    def __init__(self, project_root, campaign_id):
        self.project_root = Path(project_root).resolve()
        self.campaign_id = campaign_id
        self.directory = (
            self.project_root / "agent_artifacts" / "autobridge" / campaign_id
        )
        self.path = self.directory / "campaign.db"

    @classmethod
    def create(cls, spec):
        store = cls(spec["project_root"], spec["campaign_id"])
        store._reject_legacy_state()
        store.directory.mkdir(parents=True, exist_ok=True)
        if store.path.exists():
            raise ConflictError(f"campaign already exists: {spec['campaign_id']}")
        store._initialize()
        store.append("campaign_initialized", {"spec": spec}, expected_version=0)
        return store

    @classmethod
    def open(cls, project_root, campaign_id):
        store = cls(project_root, campaign_id)
        if not store.path.is_file():
            store._reject_legacy_state()
            raise NotFoundError(f"campaign not found: {campaign_id}")
        return store

    def _reject_legacy_state(self):
        legacy = (
            self.directory / "state.json",
            self.project_root / "agent_artifacts" / "kernel_loop" / self.campaign_id / "loop_state.json",
        )
        if any(path.exists() for path in legacy):
            raise ConflictError(
                "legacy campaign state cannot be resumed or overwritten; preserve its artifacts, "
                "initialize a new campaign ID, and submit fresh specialist evidence"
            )

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    data TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TRIGGER events_no_update
                BEFORE UPDATE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'campaign events are append-only');
                END;
                CREATE TRIGGER events_no_delete
                BEFORE DELETE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'campaign events are append-only');
                END;
                """
            )

    @staticmethod
    def _row(row):
        return {
            "sequence": row["sequence"],
            "event_id": row["event_id"],
            "kind": row["kind"],
            "occurred_at": row["occurred_at"],
            "idempotency_key": row["idempotency_key"],
            "data": json.loads(row["data"]),
            "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"],
        }

    def event_by_idempotency(self, key):
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM events WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return self._row(row) if row else None

    def append(self, kind, data, expected_version, idempotency_key=None):
        serialized = canonical_json(data)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    existing = self._row(existing)
                    if existing["kind"] != kind or canonical_json(
                        existing["data"]
                    ) != serialized:
                        raise ConflictError(
                            f"idempotency key {idempotency_key!r} was reused"
                        )
                    connection.commit()
                    return existing

            previous = connection.execute(
                "SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            version = previous["sequence"] if previous else 0
            if version != expected_version:
                raise ConflictError(
                    f"stale state_version {expected_version}; current version is {version}"
                )
            previous_hash = previous["event_hash"] if previous else ""
            event_id = uuid.uuid4().hex
            occurred_at = _now()
            body = {
                "sequence": version + 1,
                "event_id": event_id,
                "kind": kind,
                "occurred_at": occurred_at,
                "idempotency_key": idempotency_key,
                "data": data,
                "previous_hash": previous_hash,
            }
            event_hash = json_digest(body)
            connection.execute(
                """
                INSERT INTO events (
                    event_id, kind, occurred_at, idempotency_key, data,
                    previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    kind,
                    occurred_at,
                    idempotency_key,
                    serialized,
                    previous_hash,
                    event_hash,
                ),
            )
            connection.commit()
        return {
            **body,
            "idempotency_key": idempotency_key,
            "event_hash": event_hash,
        }

    def events(self):
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY sequence"
            ).fetchall()
        return [self._row(row) for row in rows]

    def verify(self):
        previous_hash = ""
        for expected_sequence, event in enumerate(self.events(), start=1):
            if event["sequence"] != expected_sequence:
                raise IntegrityError("campaign event sequence has a gap")
            if event["previous_hash"] != previous_hash:
                raise IntegrityError(
                    f"campaign hash chain broke at event {expected_sequence}"
                )
            body = {
                "sequence": event["sequence"],
                "event_id": event["event_id"],
                "kind": event["kind"],
                "occurred_at": event["occurred_at"],
                "idempotency_key": event["idempotency_key"],
                "data": event["data"],
                "previous_hash": event["previous_hash"],
            }
            if json_digest(body) != event["event_hash"]:
                raise IntegrityError(
                    f"campaign event {expected_sequence} has an invalid hash"
                )
            previous_hash = event["event_hash"]
        return True

    def state(self):
        events = self.events()
        if not events:
            raise IntegrityError("campaign has no initialization event")
        state = {
            "campaign_id": self.campaign_id,
            "state_version": 0,
            "spec": None,
            "attempts": {},
            "submissions": [],
            "paused": False,
            "pause_after_task": False,
            "final": None,
            "closeout_receipts": [],
        }
        for event in events:
            state["state_version"] = event["sequence"]
            kind = event["kind"]
            data = event["data"]
            if kind == "campaign_initialized":
                state["spec"] = data["spec"]
            elif kind == "attempt_prepared":
                task = data["task"]
                state["attempts"][task["task_id"]] = {
                    "status": "PREPARED",
                    "task": task,
                    "grant": data["grant"],
                    "prepared_sequence": event["sequence"],
                }
            elif kind in ("submission_accepted", "submission_recorded"):
                submission = dict(data["submission"])
                submission["_sequence"] = event["sequence"]
                submission["_accepted"] = kind == "submission_accepted"
                state["submissions"].append(submission)
                attempt = state["attempts"][submission["task_id"]]
                attempt["status"] = (
                    "ACCEPTED"
                    if kind == "submission_accepted"
                    else submission["status"].upper()
                )
                attempt["submission_sequence"] = event["sequence"]
                if state["pause_after_task"]:
                    state["paused"] = True
                    state["pause_after_task"] = False
            elif kind == "attempt_cancelled":
                task_id = data["task_id"]
                state["attempts"][task_id]["status"] = "CANCELLED"
                if state["pause_after_task"]:
                    state["paused"] = True
                    state["pause_after_task"] = False
            elif kind == "attempt_progress":
                progress = data["progress"]
                attempt = state["attempts"][progress["task_id"]]
                attempt["progress"] = progress
                saved = {item["path"]: item for item in attempt.get("progress_artifacts", [])}
                saved.update({item["path"]: item for item in progress["artifacts"]})
                attempt["progress_artifacts"] = list(saved.values())
            elif kind == "campaign_paused":
                if "closeout" in data:
                    state["closeout_receipts"].append(data["closeout"])
                if data["mode"] == "after-task":
                    state["pause_after_task"] = True
                else:
                    state["paused"] = True
                    state["pause_after_task"] = False
                    task_id = data.get("cancelled_task_id")
                    if task_id:
                        state["attempts"][task_id]["status"] = "CANCELLED"
            elif kind == "campaign_resumed":
                state["paused"] = False
                state["pause_after_task"] = False
                state["final"] = None
            elif kind == "campaign_finalized":
                state["final"] = data["final"]
                state["closeout_receipts"].append(data["final"])
            elif kind == "run_closed":
                state["closeout_receipts"].append(data["closeout"])

        if state["spec"] is None:
            raise IntegrityError("campaign initialization event is missing")
        active = [
            attempt
            for attempt in state["attempts"].values()
            if attempt["status"] == "PREPARED"
        ]
        state["active_attempt"] = active[0] if active else None
        if len(active) > 1:
            raise IntegrityError("campaign has more than one active attempt")
        if state["final"]:
            state["status"] = "FINALIZED"
        elif state["paused"] or state["pause_after_task"]:
            state["status"] = "PAUSED"
        elif state["active_attempt"]:
            state["status"] = "RUNNING"
        else:
            state["status"] = "READY"
        return state

    def export_jsonl(self):
        target = self.directory / "campaign.events.jsonl"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for event in self.events():
                stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        return str(target)
