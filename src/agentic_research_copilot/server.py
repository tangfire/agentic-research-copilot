from __future__ import annotations

from contextlib import asynccontextmanager
import json
from html import escape
from pathlib import Path
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
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


def create_app(copilot: ResearchCopilot | None = None) -> FastAPI:
    copilot = copilot or ResearchCopilot()
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

    @app.delete("/v1/agent/sessions/{session_id}")
    def delete_agent_session(session_id: str):
        deleted = agent.delete_session(session_id)
        if not deleted.get("deleted"):
            raise HTTPException(status_code=404, detail="Agent session not found")
        return deleted

    @app.get("/v1/agent/sessions/{session_id}/steps")
    def list_agent_steps(session_id: str):
        try:
            return agent.list_steps(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc

    @app.get("/v1/agent/sessions/{session_id}/events")
    def list_agent_events(
        session_id: str,
        request: Request,
        limit: int = Query(default=80, ge=1, le=500),
        after_event_id: str | None = None,
        format: Literal["html", "json"] | None = None,
    ):
        try:
            events = agent.list_events(session_id, limit=limit, after_event_id=after_event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Agent session not found") from exc
        accept = (request.headers.get("accept") or "").lower()
        if format != "json" and (format == "html" or ("text/html" in accept and "application/json" not in accept)):
            bundle = agent.get_session_bundle(session_id)
            return HTMLResponse(
                _render_agent_events_page(
                    session_id=session_id,
                    bundle=bundle,
                    events=events,
                    limit=limit,
                    after_event_id=after_event_id,
                )
            )
        return events

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

    @app.get("/v1/observability/status")
    def observability_status():
        return copilot.observability.status()

    return app


def _render_agent_events_page(
    *,
    session_id: str,
    bundle: object | None,
    events: list[object],
    limit: int,
    after_event_id: str | None,
) -> str:
    session = getattr(bundle, "session", None)
    session_title = getattr(session, "title", "") or session_id
    session_status = getattr(session, "status", "") or "unknown"
    workspace_id = getattr(session, "workspace_id", "") or "n/a"
    rendered_events = "\n".join(_render_agent_event_card(event, index=index) for index, event in enumerate(events)) or "<div class='empty'>当前没有事件。</div>"
    rendered_overview = _render_agent_event_overview(events)
    continuation = f"&after_event_id={escape(after_event_id)}" if after_event_id else ""
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>会话事件 - {escape(session_title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f6f8fb;
        --panel: #ffffff;
        --line: #d9e1ec;
        --text: #162033;
        --muted: #5d6b82;
        --accent: #0f766e;
        --accent-soft: #e6fffb;
      }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      }}
      .wrap {{
        max-width: 1120px;
        margin: 0 auto;
        padding: 24px;
      }}
      .head {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        margin-bottom: 20px;
      }}
      .head h1 {{
        margin: 0 0 8px;
        font-size: 24px;
      }}
      .subtle {{
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }}
      .chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }}
      .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--panel);
        color: var(--text);
        font-size: 12px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 14px;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }}
      .event-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 10px;
      }}
      .event-head h2 {{
        margin: 0 0 6px;
        font-size: 18px;
      }}
      .meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        color: var(--muted);
        font-size: 12px;
      }}
      .summary {{
        margin: 0 0 12px;
        line-height: 1.65;
        white-space: pre-wrap;
      }}
      .facts {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin-bottom: 12px;
      }}
      .fact {{
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fafcff;
      }}
      .fact .label {{
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 4px;
      }}
      details {{
        border-top: 1px dashed var(--line);
        padding-top: 10px;
      }}
      summary {{
        cursor: pointer;
        color: var(--accent);
        font-weight: 600;
      }}
      pre {{
        margin: 10px 0 0;
        padding: 12px;
        background: #0f172a;
        color: #dbeafe;
        border-radius: 10px;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
      .footer-link {{
        display: inline-block;
        margin-top: 14px;
        color: var(--accent);
        text-decoration: none;
      }}
      .empty {{
        padding: 20px;
        background: var(--panel);
        border: 1px dashed var(--line);
        border-radius: 12px;
        color: var(--muted);
      }}
      .back-link {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 18px;
        color: var(--accent);
        text-decoration: none;
        font-weight: 650;
      }}
      .back-link:hover {{ text-decoration: underline; }}
      .overview {{
        margin: 18px 0 22px;
        padding: 16px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      }}
      .section-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }}
      .section-head h2 {{ margin: 0; font-size: 16px; }}
      .section-actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
      .action {{
        appearance: none;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        color: var(--accent);
        padding: 7px 10px;
        cursor: pointer;
        font: inherit;
        font-size: 12px;
      }}
      .action:hover {{ background: var(--accent-soft); }}
      .phase-list {{
        display: flex;
        gap: 10px;
        overflow-x: auto;
        padding: 2px 2px 6px;
      }}
      .phase {{
        flex: 1 0 118px;
        min-width: 118px;
        padding: 12px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fafcff;
        color: var(--text);
        text-decoration: none;
      }}
      .phase:hover {{ border-color: var(--accent); background: var(--accent-soft); }}
      .phase-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 11px;
      }}
      .phase-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #aab6c7;
        box-shadow: 0 0 0 4px #edf1f6;
      }}
      .phase.done .phase-dot {{ background: var(--accent); box-shadow: 0 0 0 4px #d8f5ef; }}
      .phase.running .phase-dot {{ background: #d97706; box-shadow: 0 0 0 4px #fff0cf; }}
      .phase.failed .phase-dot {{ background: #dc2626; box-shadow: 0 0 0 4px #fee2e2; }}
      .phase strong {{ display: block; font-size: 13px; }}
      .phase span:last-child {{ display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }}
      .timeline {{
        position: relative;
        display: grid;
        gap: 10px;
        padding-left: 34px;
      }}
      .timeline::before {{
        content: "";
        position: absolute;
        top: 12px;
        bottom: 12px;
        left: 10px;
        width: 2px;
        background: #dbe3ed;
      }}
      .event-node {{
        position: relative;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        overflow: hidden;
      }}
      .event-node::before {{
        content: "";
        position: absolute;
        left: -30px;
        top: 18px;
        width: 12px;
        height: 12px;
        border: 3px solid var(--bg);
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 1px #b7c6d8;
        z-index: 1;
      }}
      .event-node.failed::before {{ background: #dc2626; }}
      .event-node.running::before {{ background: #d97706; }}
      .event-node.pending::before {{ background: #94a3b8; }}
      .node-summary {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        cursor: pointer;
        list-style: none;
      }}
      .node-summary::-webkit-details-marker {{ display: none; }}
      .node-summary:hover {{ background: #fbfdff; }}
      .node-main {{ min-width: 0; }}
      .node-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }}
      .node-title strong {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
      .node-kind {{
        flex: 0 0 auto;
        padding: 3px 7px;
        border-radius: 5px;
        background: #eef3f8;
        color: var(--muted);
        font-size: 11px;
      }}
      .node-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 5px;
        color: var(--muted);
        font-size: 12px;
      }}
      .chevron {{ color: var(--muted); font-size: 18px; transition: transform 0.15s ease; }}
      .event-node[open] .chevron {{ transform: rotate(180deg); }}
      .node-detail {{ padding: 0 16px 16px; border-top: 1px solid #edf1f5; }}
      .detail-summary {{ margin: 14px 0; line-height: 1.65; white-space: pre-wrap; }}
      @media (max-width: 700px) {{
        .wrap {{ padding: 16px 14px 30px; }}
        .head {{ display: block; }}
        .chips {{ margin-top: 14px; }}
        .node-summary {{ grid-template-columns: minmax(0, 1fr) auto; }}
        .node-summary > .pill {{ grid-column: 1; justify-self: start; }}
        .node-summary .chevron {{ grid-column: 2; grid-row: 1; }}
      }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <a class="back-link" href="/">← 返回研究台</a>
      <div class="head">
        <div>
          <h1>会话事件</h1>
          <p class="subtle">{escape(session_title)} 的执行过程。先看上方阶段，再点击下面的节点查看详情。</p>
        </div>
        <div class="chips">
          <span class="pill">会话 {escape(session_id)}</span>
          <span class="pill">状态 {escape(session_status)}</span>
          <span class="pill">Workspace {escape(workspace_id)}</span>
          <span class="pill">limit {limit}</span>
          {f"<span class='pill'>after {escape(after_event_id)}</span>" if after_event_id else ""}
        </div>
      </div>
      <section class="overview">
        <div class="section-head">
          <h2>流程概览</h2>
          <div class="section-actions">
            <button class="action" type="button" data-expand-all>全部展开</button>
            <button class="action" type="button" data-collapse-all>全部收起</button>
          </div>
        </div>
        <div class="phase-list">
          {rendered_overview}
        </div>
      </section>
      <section class="timeline" aria-label="会话事件节点">
        {rendered_events}
      </section>
      <a class="footer-link" href="/v1/agent/sessions/{escape(session_id)}/events?format=json&limit={limit}{continuation}">打开原始 JSON</a>
    </main>
    <script>
      document.querySelectorAll(".event-time").forEach((node) => {{
        const date = new Date(node.dataset.iso || "");
        if (Number.isNaN(date.getTime())) return;
        node.textContent = new Intl.DateTimeFormat("zh-CN", {{
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }}).format(date).replaceAll("/", "-");
      }});
      document.querySelector("[data-expand-all]")?.addEventListener("click", () => {{
        document.querySelectorAll(".event-node").forEach((node) => {{ node.open = true; }});
      }});
      document.querySelector("[data-collapse-all]")?.addEventListener("click", () => {{
        document.querySelectorAll(".event-node").forEach((node) => {{ node.open = false; }});
      }});
    </script>
  </body>
</html>"""


def _render_agent_event_card(event: object, *, index: int) -> str:
    payload = getattr(event, "payload", {}) or {}
    payload_json = escape(json.dumps(payload, ensure_ascii=False, indent=2))
    kind = str(getattr(event, "kind", "event") or "event")
    title = _display_event_title(str(getattr(event, "title", "事件") or "事件"))
    summary = str(getattr(event, "summary", "") or "无摘要")
    actor = str(getattr(event, "actor", "agent") or "agent")
    status = str(getattr(event, "status", "completed") or "completed")
    tool_name = str(getattr(event, "tool_name", "") or "n/a")
    created_at = str(getattr(event, "created_at", "") or "")
    run_id = str(getattr(event, "run_id", "") or "n/a")
    job_id = str(getattr(event, "job_id", "") or "n/a")
    event_id = str(getattr(event, "event_id", "") or "")
    phase = _event_phase(event)
    detail_id = f"event-{event_id or index}"
    node_status = status if status in {"failed", "running", "pending"} else "done"
    return f"""
        <details class="event-node {node_status}" id="{escape(detail_id)}">
          <summary class="node-summary">
            <span class="node-main">
              <span class="node-title"><strong>{escape(title)}</strong><span class="node-kind">{escape(_display_kind(phase))}</span></span>
              <span class="node-meta">
                <span class="event-time" data-iso="{escape(created_at)}">{escape(created_at or "n/a")}</span>
                <span>执行者：{escape(actor)}</span>
                {f"<span>工具：{escape(tool_name)}</span>" if tool_name != "n/a" else ""}
              </span>
            </span>
            <span class="pill">{escape(_display_status(status))}</span>
            <span class="chevron" aria-hidden="true">⌄</span>
          </summary>
          <div class="node-detail">
            <p class="detail-summary">{escape(summary)}</p>
            <div class="facts">
              <div class="fact"><span class="label">运行</span>{escape(run_id)}</div>
              <div class="fact"><span class="label">任务</span>{escape(job_id)}</div>
              <div class="fact"><span class="label">节点 ID</span>{escape(event_id)}</div>
              <div class="fact"><span class="label">事件类型</span>{escape(_display_kind(kind))}</div>
            </div>
            <details>
              <summary>查看原始 JSON</summary>
              <pre>{payload_json}</pre>
            </details>
          </div>
        </details>
    """


def _render_agent_event_overview(events: list[object]) -> str:
    phase_order = ["message", "planning", "tool_call", "retrieval", "research", "report", "verification", "evaluation"]
    grouped: dict[str, list[object]] = {phase: [] for phase in phase_order}
    for event in events:
        phase = _event_phase(event)
        if phase in grouped:
            grouped[phase].append(event)
    rendered: list[str] = []
    for phase in phase_order:
        phase_events = grouped[phase]
        if not phase_events:
            continue
        status = _phase_status(phase_events)
        first_event_id = str(getattr(phase_events[0], "event_id", "") or "")
        target = f"#event-{escape(first_event_id)}" if first_event_id else "#event-timeline"
        rendered.append(
            f"<a class='phase {status}' href='{target}'>"
            f"<div class='phase-top'><span class='phase-dot'></span><span>{len(phase_events)} 个节点</span></div>"
            f"<strong>{escape(_display_kind(phase))}</strong>"
            f"<span>{escape(_display_status(status))}</span>"
            "</a>"
        )
    return "".join(rendered) or "<div class='empty'>当前没有可展示的流程节点。</div>"


def _event_phase(event: object) -> str:
    kind = str(getattr(event, "kind", "message") or "message")
    if kind in {"message", "planning", "tool_call", "retrieval", "research", "report", "verification", "evaluation"}:
        return kind
    event_type = str(getattr(event, "type", "") or "")
    return "tool_call" if event_type == "tool_invocation" else "message"


def _phase_status(events: list[object]) -> str:
    statuses = {str(getattr(event, "status", "completed") or "completed") for event in events}
    if "failed" in statuses:
        return "failed"
    if "running" in statuses or "pending" in statuses:
        return "running"
    return "done"


def _display_event_title(title: str) -> str:
    mapping = {
        "user message": "用户消息",
        "assistant message": "助手消息",
        "User message received": "收到用户消息",
        "Clarification required": "需要补充信息",
        "Research cancellation requested": "研究取消请求",
    }
    if title.startswith("Skill preflight:"):
        return title.replace("Skill preflight:", "Skill 预检：", 1)
    return mapping.get(title, title)


def _display_status(status: str) -> str:
    return {
        "chat": "对话",
        "completed": "完成",
        "running": "运行中",
        "pending": "待处理",
        "failed": "失败",
        "skipped": "跳过",
        "done": "完成",
        "plan": "计划",
        "clarify": "追问",
        "research": "研究",
    }.get(status, status)


def _display_kind(kind: str) -> str:
    return {
        "message": "消息",
        "planning": "规划",
        "tool_call": "工具调用",
        "retrieval": "检索",
        "research": "研究",
        "report": "报告",
        "verification": "验证",
        "evaluation": "评估",
        "approval": "审批",
        "failure": "失败",
        "heartbeat": "心跳",
    }.get(kind, kind)
