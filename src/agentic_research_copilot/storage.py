from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .schemas import EvidenceItem, MemoryRecord, ResearchJob, ResearchRun


class SQLiteStore:
    """Lightweight durable store inspired by PraisonAI-style persistence and replay."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def load_documents(self) -> list[EvidenceItem]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM documents ORDER BY created_at").fetchall()
        return [EvidenceItem.model_validate_json(row[0]) for row in rows]

    def save_document(self, document: EvidenceItem) -> None:
        payload = document.model_dump_json()
        identity = self._document_identity(document)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents(identity, payload, created_at)
                VALUES (?, ?, COALESCE((SELECT created_at FROM documents WHERE identity = ?), datetime('now')))
                """,
                (identity, payload, identity),
            )

    def load_memory(self) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM memory_records ORDER BY created_at").fetchall()
        return [MemoryRecord.model_validate_json(row[0]) for row in rows]

    def save_memory(self, record: MemoryRecord) -> None:
        payload = record.model_dump_json()
        identity = self._memory_identity(record)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_records(identity, payload, created_at)
                VALUES (?, ?, ?)
                """,
                (identity, payload, record.created_at),
            )

    def load_runs(self) -> list[ResearchRun]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM research_runs ORDER BY started_at DESC").fetchall()
        return [ResearchRun.model_validate_json(row[0]) for row in rows]

    def load_run(self, run_id: str) -> ResearchRun | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchRun.model_validate_json(row[0])

    def load_jobs(self) -> list[ResearchJob]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM research_jobs ORDER BY queued_at DESC").fetchall()
        return [ResearchJob.model_validate_json(row[0]) for row in rows]

    def load_job(self, job_id: str) -> ResearchJob | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM research_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchJob.model_validate_json(row[0])

    def save_job(self, job: ResearchJob) -> None:
        payload = job.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_jobs(job_id, payload, queued_at)
                VALUES (?, ?, ?)
                """,
                (job.job_id, payload, job.queued_at),
            )

    def save_run(self, run: ResearchRun) -> None:
        payload = run.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_runs(run_id, payload, started_at)
                VALUES (?, ?, ?)
                """,
                (run.run_id, payload, run.started_at or ""),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  identity TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                  identity TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                  run_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  started_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_jobs (
                  job_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  queued_at TEXT NOT NULL
                )
                """
            )

    def _document_identity(self, document: EvidenceItem) -> str:
        stable = document.url or f"{document.source}:{document.title}:{document.snippet or document.content or ''}"
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def _memory_identity(self, record: MemoryRecord) -> str:
        stable = f"{record.key}:{record.created_at}:{record.value}"
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
