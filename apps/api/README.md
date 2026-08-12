# API

FastAPI backend for the local Agentic Research Runtime.

It intentionally stays small. The API exists to expose the research loop and its artifacts, not to make the project look like a generic CRUD system.

Main surfaces:

- research runs and asynchronous jobs
- document ingest and local corpus search
- agent orchestration through the pipeline
- report generation and replay
- trace, evaluation, telemetry, and runtime config

There are no memory endpoints in the current core. Project memory and the old local workbench MCP server were removed so the backend remains focused on evidence-grounded research runs.
