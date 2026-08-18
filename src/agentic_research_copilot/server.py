from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import ConversationalResearchAgent
from .constraint_evaluation import extract_constraint_coverage_from_run
from .document_reader import DocumentReadError
from .pipeline import ResearchCopilot
from .schemas import ResearchRequest, WorkspaceProfile


WEB_INDEX_PATH = Path(__file__).resolve().parents[2] / "apps" / "web" / "index.html"


class DocumentInput(BaseModel):
    title: str = Field(min_length=2)
    source: str = Field(min_length=2)
    url: str | None = None
    snippet: str | None = None
    content: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentIngestInput(BaseModel):
    path: str = Field(min_length=1)
    title: str | None = None
    source: str | None = None
    url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentSessionInput(BaseModel):
    title: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentMessageInput(BaseModel):
    content: str = Field(min_length=1)
    depth: Literal["quick", "standard", "deep"] = "standard"
    include_private_docs: bool = True
    max_sections: int = Field(default=4, ge=1, le=8)
    max_revisions: int = Field(default=1, ge=0, le=4)
    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryInput(BaseModel):
    content: str = Field(min_length=1)
    scope: Literal["user", "project", "session"] = "project"
    kind: Literal["preference", "constraint", "decision", "fact", "todo"] = "fact"
    session_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class WorkspaceInput(BaseModel):
    workspace_id: str | None = None
    name: str = Field(min_length=2)
    team_context: str = ""
    default_stack: list[str] = Field(default_factory=list)
    deployment_constraints: list[str] = Field(default_factory=list)
    risk_policy: str = ""
    preferred_sources: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class SkillScriptRunInput(BaseModel):
    payload: dict[str, object] = Field(default_factory=dict)
    timeout_seconds: float | None = None


def create_app() -> FastAPI:
    copilot = ResearchCopilot()
    agent = ConversationalResearchAgent(copilot)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        copilot.close()

    app = FastAPI(title="AI Research Copilot", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        if WEB_INDEX_PATH.exists():
            return WEB_INDEX_PATH.read_text(encoding="utf-8")
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>AI Research Copilot</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 32px; color: #111827; }
              h1 { margin-bottom: 8px; }
              p { line-height: 1.6; }
              a { color: #2563eb; text-decoration: none; }
              a:hover { text-decoration: underline; }
              .card { max-width: 760px; padding: 20px 24px; border: 1px solid #e5e7eb; border-radius: 12px; }
              ul { padding-left: 20px; }
              code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>AI Research Copilot</h1>
              <p>The API is running. Use the links below to explore the research workflow.</p>
              <ul>
                <li><a href="/docs">Interactive API docs</a></li>
                <li><a href="/health">Health check</a></li>
                <li><code>GET /v1/research/runs</code></li>
                <li><code>POST /v1/research/runs</code></li>
                <li><code>POST /v1/research/clarify</code></li>
                <li><code>GET /v1/research/jobs</code></li>
                <li><code>POST /v1/research/jobs</code></li>
                <li><code>POST /v1/research/jobs/{job_id}/cancel</code></li>
                <li><code>GET /v1/research/runs/{run_id}</code></li>
                <li><code>GET /v1/research/runs/{run_id}/checkpoints</code></li>
                <li><code>POST /v1/research/runs/{run_id}/replay</code></li>
                <li><code>GET /v1/documents</code></li>
                <li><code>GET /v1/documents/search</code></li>
                <li><code>POST /v1/documents</code></li>
                <li><code>POST /v1/documents/ingest</code></li>
                <li><code>DELETE /v1/documents/{document_id}</code></li>
                <li><code>DELETE /v1/documents</code></li>
                <li><code>DELETE /v1/research/history</code></li>
                <li><code>GET /v1/telemetry</code></li>
                <li><code>GET /v1/runtime/provider-check</code></li>
              </ul>
            </div>
          </body>
        </html>
        """

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/agent/tools")
    def list_agent_tools():
        return agent.tool_definitions()

    @app.post("/v1/agent/sessions")
    def create_agent_session(payload: AgentSessionInput | None = None):
        payload = payload or AgentSessionInput()
        try:
            return agent.create_session(title=payload.title, workspace_id=payload.workspace_id, metadata=payload.metadata)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

    @app.get("/v1/agent/sessions")
    def list_agent_sessions():
        return agent.list_sessions()

    @app.get("/v1/agent/sessions/{session_id}")
    def get_agent_session(session_id: str):
        bundle = agent.get_session_bundle(session_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="Agent session not found")
        return bundle

    @app.get("/v1/agent/sessions/{session_id}/export")
    def export_agent_session(session_id: str):
        try:
            return agent.export_session_bundle(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.post("/v1/agent/sessions/{session_id}/messages")
    def post_agent_message(session_id: str, payload: AgentMessageInput):
        try:
            return agent.receive_message(
                session_id,
                payload.content,
                depth=payload.depth,
                include_private_docs=payload.include_private_docs,
                max_sections=payload.max_sections,
                max_revisions=payload.max_revisions,
                metadata=payload.metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.post("/v1/agent/sessions/{session_id}/confirm-plan")
    def confirm_agent_plan(session_id: str):
        try:
            return agent.confirm_plan(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/agent/sessions/{session_id}/cancel")
    def cancel_agent_session(session_id: str):
        try:
            return agent.cancel_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.get("/v1/agent/sessions/{session_id}/steps")
    def list_agent_steps(session_id: str):
        try:
            return agent.list_steps(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.get("/v1/agent/sessions/{session_id}/events")
    def list_agent_events(session_id: str, limit: int = Query(default=80, ge=1, le=500)):
        try:
            return agent.list_events(session_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.get("/v1/agent/sessions/{session_id}/tool-invocations")
    def list_agent_tool_invocations(session_id: str):
        try:
            return agent.list_tool_invocations(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.post("/v1/agent/sessions/{session_id}/approvals/{approval_id}/approve")
    def approve_agent_action(session_id: str, approval_id: str):
        try:
            return agent.resolve_approval(session_id, approval_id, approve=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Approval request not found") from exc

    @app.post("/v1/agent/sessions/{session_id}/approvals/{approval_id}/reject")
    def reject_agent_action(session_id: str, approval_id: str):
        try:
            return agent.resolve_approval(session_id, approval_id, approve=False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Approval request not found") from exc

    @app.get("/v1/agent/sessions/{session_id}/memory")
    def get_agent_session_memory(session_id: str):
        if agent.get_session_bundle(session_id) is None:
            raise HTTPException(status_code=404, detail="Agent session not found")
        return agent.list_memory(session_id=session_id)

    @app.get("/v1/agent/sessions/{session_id}/memory/evaluation")
    def get_agent_memory_evaluation(session_id: str):
        try:
            return agent.memory_evaluation(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.post("/v1/memory")
    def add_memory(payload: MemoryInput):
        return agent.add_memory(
            content=payload.content,
            scope=payload.scope,
            kind=payload.kind,
            session_id=payload.session_id,
            metadata=payload.metadata,
        )

    @app.get("/v1/memory")
    def list_memory(
        scope: Literal["user", "project", "session"] | None = None,
        session_id: str | None = None,
    ):
        return agent.list_memory(scope=scope, session_id=session_id)

    @app.delete("/v1/memory/{memory_id}")
    def delete_memory(memory_id: str):
        if not agent.delete_memory(memory_id):
            raise HTTPException(status_code=404, detail="Memory item not found")
        return {"deleted": True, "memory_id": memory_id}

    @app.get("/v1/agent/workspaces")
    def list_agent_workspaces():
        return agent.list_workspaces()

    @app.get("/v1/agent/workspaces/{workspace_id}")
    def get_agent_workspace(workspace_id: str):
        try:
            return agent.get_workspace(workspace_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Workspace not found") from exc

    @app.post("/v1/agent/workspaces")
    def upsert_agent_workspace(payload: WorkspaceInput):
        workspace_id = payload.workspace_id or str(uuid.uuid4())
        existing = None
        try:
            existing = agent.get_workspace(workspace_id)
        except KeyError:
            existing = None
        if existing is not None:
            workspace = existing.model_copy(
                update={
                    "name": payload.name,
                    "team_context": payload.team_context,
                    "default_stack": payload.default_stack,
                    "deployment_constraints": payload.deployment_constraints,
                    "risk_policy": payload.risk_policy,
                    "preferred_sources": payload.preferred_sources,
                    "disabled_tools": payload.disabled_tools,
                    "metadata": {**existing.metadata, **payload.metadata},
                }
            )
        else:
            workspace = WorkspaceProfile(
                workspace_id=workspace_id,
                name=payload.name,
                team_context=payload.team_context,
                default_stack=payload.default_stack,
                deployment_constraints=payload.deployment_constraints,
                risk_policy=payload.risk_policy,
                preferred_sources=payload.preferred_sources,
                disabled_tools=payload.disabled_tools,
                metadata=payload.metadata,
            )
        return agent.save_workspace(workspace)

    @app.get("/v1/agent/skills")
    def list_agent_skills():
        return agent.skill_catalog()

    @app.get("/v1/agent/skills/{skill_id}")
    def get_agent_skill(skill_id: str):
        try:
            return agent.describe_skill(skill_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Skill not found") from exc

    @app.post("/v1/agent/skills/{skill_id}/scripts/{script_name}/run")
    def run_agent_skill_script(skill_id: str, script_name: str, payload: SkillScriptRunInput | None = None):
        payload = payload or SkillScriptRunInput()
        try:
            return agent.run_skill_script(
                skill_id,
                script_name,
                payload=payload.payload,
                timeout_seconds=payload.timeout_seconds,
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Skill or script not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/research/runs")
    def create_run(request: ResearchRequest):
        return copilot.run(request)

    @app.post("/v1/research/clarify")
    def clarify_research(request: ResearchRequest):
        return copilot.clarify(request)

    @app.post("/v1/research/jobs")
    def create_job(request: ResearchRequest):
        job = copilot.submit_job(request)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "run_id": job.run_id,
            "status_url": f"/v1/research/jobs/{job.job_id}/status",
            "result_url": f"/v1/research/jobs/{job.job_id}/result",
            "job": job,
        }

    @app.get("/v1/research/jobs")
    def list_jobs():
        return copilot.list_jobs()

    @app.get("/v1/research/jobs/{job_id}")
    def get_job(job_id: str):
        job = copilot.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/v1/research/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = copilot.cancel_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/v1/research/jobs/{job_id}/status")
    def get_job_status(job_id: str):
        job = copilot.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        run = copilot.get_run(job.run_id) if job.run_id else None
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "topic": job.request.topic,
            "queued_at": job.queued_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "cancel_requested": job.cancel_requested,
            "checkpoint_count": len(run.checkpoints) if run else 0,
            "issue_count": len(run.issues) if run else 0,
            "source_count": run.report.source_count if run and run.report else 0,
        }

    @app.get("/v1/research/jobs/{job_id}/result")
    def get_job_result(job_id: str):
        job = copilot.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed" or not job.run_id:
            return JSONResponse(
                status_code=202,
                content={
                    "job_id": job.job_id,
                    "status": job.status,
                    "detail": "Job is not complete yet",
                },
            )
        run = copilot.get_run(job.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found for completed job")
        return run

    @app.get("/v1/research/runs")
    def list_runs():
        return copilot.list_runs()

    @app.get("/v1/research/runs/{run_id}")
    def get_run(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/v1/research/runs/{run_id}/status")
    def get_run_status(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "topic": run.request.topic,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "duration_ms": run.duration_ms,
            "checkpoint_count": len(run.checkpoints),
            "issue_count": len(run.issues),
            "source_count": run.report.source_count if run.report else 0,
        }

    @app.get("/v1/research/runs/{run_id}/result")
    def get_run_result(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "topic": run.request.topic,
            "report": run.report,
            "evaluation": run.evaluation,
            "issues": run.issues,
            "evidence": run.evidence,
            "retrieval_routes": run.retrieval_routes,
            "role_assignments": run.role_assignments,
            "route_decisions": run.route_decisions,
            "conflicts": run.conflicts,
            "evidence_ledger": run.evidence_ledger,
            "benchmark_summary": run.benchmark_summary,
            "checkpoints": run.checkpoints,
        }

    @app.get("/v1/research/runs/{run_id}/harness")
    def get_run_harness(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "role_assignments": run.role_assignments,
            "route_decisions": run.route_decisions,
            "conflicts": run.conflicts,
            "evidence_ledger": run.evidence_ledger,
            "benchmark_summary": run.benchmark_summary,
            "metadata": run.metadata,
        }

    @app.get("/v1/research/runs/{run_id}/checkpoints")
    def get_checkpoints(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.checkpoints

    @app.get("/v1/research/runs/{run_id}/trace")
    def get_run_trace(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.trace

    @app.get("/v1/research/runs/{run_id}/evaluation")
    def get_run_evaluation(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return run.evaluation

    @app.get("/v1/research/runs/{run_id}/constraint-coverage")
    def get_run_constraint_coverage(run_id: str):
        run = copilot.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        coverage = copilot.storage.load_constraint_coverage(run_id)
        if coverage:
            return coverage
        coverage = extract_constraint_coverage_from_run(run)
        copilot.storage.save_constraint_coverage(coverage)
        return coverage

    @app.post("/v1/research/runs/{run_id}/replay")
    def replay_run(run_id: str):
        run = copilot.replay(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/v1/documents")
    def list_documents():
        return copilot.documents.list()

    @app.get("/v1/documents/search")
    def search_documents(
        q: str = Query(min_length=1),
        limit: int = Query(default=5, ge=1, le=20),
    ):
        results = copilot.documents.search(q, limit=limit)
        return {
            "query": q,
            "result_count": len(results),
            "results": results,
            "corpus_profile": copilot.documents.profile(),
        }

    @app.post("/v1/documents")
    def add_document(payload: DocumentInput):
        return copilot.add_document(
            title=payload.title,
            source=payload.source,
            url=payload.url,
            snippet=payload.snippet,
            content=payload.content,
            metadata=payload.metadata,
        )

    @app.post("/v1/documents/ingest")
    def ingest_document(payload: DocumentIngestInput):
        try:
            documents = copilot.ingest_document_path(
                payload.path,
                title=payload.title,
                source=payload.source,
                url=payload.url,
                metadata=payload.metadata,
            )
        except DocumentReadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "document_count": len(documents),
            "documents": documents,
        }

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: str):
        if not copilot.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "document_id": document_id}

    @app.delete("/v1/documents")
    def clear_documents():
        return copilot.clear_documents()

    @app.delete("/v1/research/history")
    def clear_research_history():
        return copilot.clear_history()

    @app.get("/v1/telemetry")
    def list_telemetry(
        kind: str | None = None,
        run_id: str | None = None,
    ):
        events = copilot.telemetry.all()
        if kind is not None:
            events = [event for event in events if event.kind == kind]
        if run_id is not None:
            events = [event for event in events if event.run_id == run_id]
        return [event.__dict__ for event in events]

    @app.get("/v1/traces")
    def list_traces(
        run_id: str | None = None,
        kind: str | None = None,
    ):
        events = copilot.telemetry.all()
        if run_id is not None:
            events = [event for event in events if event.run_id == run_id]
        if kind is not None:
            events = [event for event in events if event.kind == kind]
        return [event.__dict__ for event in events]

    @app.get("/v1/runtime/config")
    def runtime_config():
        return copilot.runtime_config()

    @app.get("/v1/runtime/provider-check")
    def provider_check():
        return copilot.runtime_config()["provider_readiness"]

    return app
