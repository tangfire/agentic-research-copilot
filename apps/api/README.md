# API

FastAPI backend for the local Agentic Research Runtime.

It intentionally stays small. The API exists to expose the research loop and its artifacts, not to make the project look like a generic CRUD system.

Main surfaces:

- research runs and asynchronous jobs
- document ingest and local corpus search
- conversational agent sessions, memory, workspaces, skills, and plan confirmation
- specialist worker routing through `RepoSignalAgent`, `ArchitectureFitAgent`, and `OpsRiskAgent`
- report generation and replay
- trace, evaluation, telemetry, and runtime config

The old local workbench MCP server is intentionally gone. MCP is now only an external evidence boundary. Memory endpoints are part of the current agent workbench layer:

- `POST /v1/memory`
- `GET /v1/memory`
- `DELETE /v1/memory/{memory_id}`

The lower-level `/v1/research/*` APIs remain focused on evidence-grounded research runs.
