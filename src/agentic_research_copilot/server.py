from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .pipeline import ResearchCopilot
from .schemas import ResearchRequest


WEB_INDEX_PATH = Path(__file__).resolve().parents[2] / "apps" / "web" / "index.html"


class DocumentInput(BaseModel):
    title: str = Field(min_length=2)
    source: str = Field(min_length=2)
    url: str | None = None
    snippet: str | None = None
    content: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class MemoryInput(BaseModel):
    key: str = Field(min_length=2)
    value: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    layer: str = "session"
    run_id: str | None = None
    session_id: str | None = None
    topic: str | None = None
    confidence: float = 0.0


def create_app() -> FastAPI:
    copilot = ResearchCopilot()

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
                <li><code>GET /v1/research/jobs</code></li>
                <li><code>POST /v1/research/jobs</code></li>
                <li><code>POST /v1/research/jobs/{job_id}/cancel</code></li>
                <li><code>GET /v1/research/runs/{run_id}</code></li>
                <li><code>GET /v1/research/runs/{run_id}/checkpoints</code></li>
                <li><code>POST /v1/research/runs/{run_id}/replay</code></li>
                <li><code>GET /v1/documents</code></li>
                <li><code>POST /v1/documents</code></li>
                <li><code>GET /v1/memory</code></li>
                <li><code>GET /v1/memory/governance</code></li>
                <li><code>POST /v1/memory</code></li>
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

    @app.post("/v1/research/runs")
    def create_run(request: ResearchRequest):
        return copilot.run(request)

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
            "checkpoints": run.checkpoints,
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

    @app.post("/v1/research/runs/{run_id}/replay")
    def replay_run(run_id: str):
        run = copilot.replay(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/v1/documents")
    def list_documents():
        return copilot.documents.list()

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

    @app.get("/v1/memory")
    def list_memory(
        layer: str | None = None,
        topic: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        tag: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
    ):
        return copilot.memory.list(
            layer=layer,
            topic=topic,
            run_id=run_id,
            session_id=session_id,
            tag=tag,
            limit=limit,
        )

    @app.get("/v1/memory/governance")
    def memory_governance():
        return copilot.memory.governance_report()

    @app.post("/v1/memory")
    def add_memory(payload: MemoryInput):
        return copilot.add_memory(
            key=payload.key,
            value=payload.value,
            tags=payload.tags,
            metadata=payload.metadata,
            layer=payload.layer,
            run_id=payload.run_id,
            session_id=payload.session_id,
            topic=payload.topic,
            confidence=payload.confidence,
        )

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


app = create_app()
