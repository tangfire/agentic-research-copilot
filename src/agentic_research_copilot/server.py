from __future__ import annotations

from fastapi import FastAPI

from .pipeline import ResearchCopilot
from .schemas import ResearchRequest


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Research Copilot", version="0.1.0")
    copilot = ResearchCopilot()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/research/runs")
    def create_run(request: ResearchRequest):
        return copilot.run(request)

    @app.get("/v1/memory")
    def list_memory():
        return copilot.memory.list()

    @app.get("/v1/telemetry")
    def list_telemetry():
        return [event.__dict__ for event in copilot.telemetry.all()]

    return app


app = create_app()

