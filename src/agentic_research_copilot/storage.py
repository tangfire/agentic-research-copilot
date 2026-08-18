from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .schemas import (
    AgentEvent,
    AgentMessage,
    AgentPlanDraft,
    AgentRunStep,
    AgentSession,
    ApprovalRequest,
    ConstraintCoverage,
    EvidenceItem,
    MemoryExtractionResult,
    MemoryItem,
    ResearchJob,
    ResearchRun,
    ResearchSkill,
    WorkspaceProfile,
    ToolInvocation,
)


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
        identity = self.document_identity(document)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents(identity, payload, created_at)
                VALUES (?, ?, COALESCE((SELECT created_at FROM documents WHERE identity = ?), datetime('now')))
                """,
                (identity, payload, identity),
            )

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE identity = ?", (document_id,))
        return cursor.rowcount > 0

    def clear_documents(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents")
        return cursor.rowcount

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

    def load_agent_sessions(self) -> list[AgentSession]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM agent_sessions ORDER BY updated_at DESC").fetchall()
        return [AgentSession.model_validate_json(row[0]) for row in rows]

    def load_agent_session(self, session_id: str) -> AgentSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return AgentSession.model_validate_json(row[0])

    def save_agent_session(self, session: AgentSession) -> None:
        payload = session.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_sessions(session_id, payload, updated_at)
                VALUES (?, ?, ?)
                """,
                (session.session_id, payload, session.updated_at),
            )

    def load_workspaces(self) -> list[WorkspaceProfile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM agent_workspaces ORDER BY updated_at DESC").fetchall()
        return [WorkspaceProfile.model_validate_json(row[0]) for row in rows]

    def load_workspace(self, workspace_id: str) -> WorkspaceProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM agent_workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkspaceProfile.model_validate_json(row[0])

    def save_workspace(self, workspace: WorkspaceProfile) -> None:
        payload = workspace.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_workspaces(workspace_id, payload, updated_at)
                VALUES (?, ?, ?)
                """,
                (workspace.workspace_id, payload, workspace.updated_at),
            )

    def load_agent_messages(self, session_id: str) -> list[AgentMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM agent_messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [AgentMessage.model_validate_json(row[0]) for row in rows]

    def save_agent_message(self, message: AgentMessage) -> None:
        payload = message.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_messages(message_id, session_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (message.message_id, message.session_id, payload, message.created_at),
            )

    def load_agent_steps(self, session_id: str) -> list[AgentRunStep]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM agent_run_steps WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [AgentRunStep.model_validate_json(row[0]) for row in rows]

    def save_agent_step(self, step: AgentRunStep) -> None:
        payload = step.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_run_steps(step_id, session_id, job_id, run_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (step.step_id, step.session_id, step.job_id, step.run_id, payload, step.created_at, step.updated_at),
            )

    def load_tool_invocations(self, session_id: str) -> list[ToolInvocation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM tool_invocations WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [ToolInvocation.model_validate_json(row[0]) for row in rows]

    def save_tool_invocation(self, invocation: ToolInvocation) -> None:
        payload = invocation.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tool_invocations(invocation_id, session_id, run_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    invocation.invocation_id,
                    invocation.session_id,
                    invocation.run_id,
                    payload,
                    invocation.created_at,
                    invocation.updated_at,
                ),
            )

    def load_approval_requests(self, session_id: str) -> list[ApprovalRequest]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM approval_requests WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [ApprovalRequest.model_validate_json(row[0]) for row in rows]

    def load_approval_request(self, approval_id: str) -> ApprovalRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM approval_requests WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return ApprovalRequest.model_validate_json(row[0])

    def save_approval_request(self, approval: ApprovalRequest) -> None:
        payload = approval.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approval_requests(approval_id, session_id, invocation_id, payload, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.approval_id,
                    approval.session_id,
                    approval.invocation_id,
                    payload,
                    approval.created_at,
                    approval.resolved_at,
                ),
            )

    def load_agent_plan_draft(self, session_id: str) -> AgentPlanDraft | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM agent_plan_drafts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return AgentPlanDraft.model_validate_json(row[0])

    def save_agent_plan_draft(self, plan_draft: AgentPlanDraft) -> None:
        payload = plan_draft.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_plan_drafts(session_id, payload, created_at)
                VALUES (?, ?, ?)
                """,
                (plan_draft.session_id, payload, plan_draft.created_at),
            )

    def delete_agent_plan_draft(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM agent_plan_drafts WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def load_memory_items(
        self,
        *,
        scope: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        query = "SELECT payload FROM memory_items"
        conditions: list[str] = []
        params: list[str] = []
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if session_id:
            conditions.append("(session_id = ? OR session_id IS NULL)")
            params.append(session_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [MemoryItem.model_validate_json(row[0]) for row in rows]

    def load_memory_item(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return MemoryItem.model_validate_json(row[0])

    def save_memory_item(self, memory: MemoryItem) -> None:
        payload = memory.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items(memory_id, scope, session_id, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory.memory_id, memory.scope, memory.session_id, payload, memory.updated_at),
            )

    def delete_memory_item(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memory_items WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0

    def load_memory_extraction_results(self, session_id: str) -> list[MemoryExtractionResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM memory_extraction_results WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [MemoryExtractionResult.model_validate_json(row[0]) for row in rows]

    def save_memory_extraction_result(self, result: MemoryExtractionResult) -> None:
        payload = result.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_extraction_results(source_message_id, session_id, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (result.source_message_id, result.session_id, payload, result.created_at),
            )

    def load_constraint_coverage(self, run_id: str) -> list[ConstraintCoverage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM constraint_coverage WHERE run_id = ? ORDER BY constraint_id",
                (run_id,),
            ).fetchall()
        return [ConstraintCoverage.model_validate_json(row[0]) for row in rows]

    def save_constraint_coverage(self, coverage: list[ConstraintCoverage]) -> None:
        if not coverage:
            return
        run_id = coverage[0].run_id
        with self._connect() as conn:
            if run_id:
                conn.execute("DELETE FROM constraint_coverage WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT OR REPLACE INTO constraint_coverage(run_id, constraint_id, payload, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        item.run_id or "",
                        item.constraint_id,
                        item.model_dump_json(),
                        item.created_at,
                    )
                    for item in coverage
                ],
            )

    def clear_runs(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM research_runs")
        return cursor.rowcount

    def clear_jobs(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM research_jobs")
        return cursor.rowcount

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                  session_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_workspaces (
                  workspace_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_messages (
                  message_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_steps (
                  step_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  job_id TEXT,
                  run_id TEXT,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_invocations (
                  invocation_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  run_id TEXT,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                  approval_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  invocation_id TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  resolved_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_plan_drafts (
                  session_id TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_extraction_results (
                  source_message_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS constraint_coverage (
                  run_id TEXT NOT NULL,
                  constraint_id TEXT NOT NULL,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (run_id, constraint_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                  memory_id TEXT PRIMARY KEY,
                  scope TEXT NOT NULL,
                  session_id TEXT,
                  payload TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

    def document_identity(self, document: EvidenceItem) -> str:
        metadata_id = document.metadata.get("document_id")
        if isinstance(metadata_id, str) and metadata_id:
            return metadata_id
        stable = document.url or f"{document.source}:{document.title}:{document.snippet or document.content or ''}"
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()
