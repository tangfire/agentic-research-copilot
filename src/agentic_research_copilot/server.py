from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .pipeline import ResearchCopilot
from .schemas import ResearchRequest


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic Research Copilot", version="0.1.0")
    copilot = ResearchCopilot()

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Agentic Research Copilot</title>
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
              <h1>Agentic Research Copilot</h1>
              <p>The API is running. Use the links below to explore the service.</p>
              <ul>
                <li><a href="/docs">Interactive API docs</a></li>
                <li><a href="/health">Health check</a></li>
                <li><code>POST /v1/research/runs</code></li>
                <li><code>GET /v1/memory</code></li>
                <li><code>GET /v1/telemetry</code></li>
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

    @app.get("/v1/memory")
    def list_memory():
        return copilot.memory.list()

    @app.get("/v1/telemetry")
    def list_telemetry():
        return [event.__dict__ for event in copilot.telemetry.all()]

    return app


app = create_app()
