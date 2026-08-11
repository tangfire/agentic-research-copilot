# AI Research Copilot

AI Research Copilot is a single-node technical research and engineering-intelligence assistant. It turns an open-ended technical question into a planned, citation-backed, traceable research report by choosing between web search, GitHub MCP, and local Agentic RAG evidence.

This repo should be presented as a large-model application project for technical research, not as a generic CRUD system, private-data assistant, GitHub-only analyzer, or MCP wrapper. The useful learning target is the full research runtime: planning, agent supervision, search/tool routing, GitHub developer evidence, Agentic RAG, source reading, verification, evaluation, and replay.

See [Product Positioning](docs/product-positioning.md) for the intended product boundary.

## Current Boundary

The current core path is:

```text
clarify -> plan -> supervise research -> search/read/retrieve -> synthesize -> verify/evaluate -> persist trace
```

The core deliberately does **not** include a project memory module anymore. The previous layered memory store was removed because it made the interview story look broader than the current experiments justified.

The local research-workbench MCP server was also removed. MCP is now kept as an external tool boundary through `mcp_tools.py`: when `ARC_MCP_SERVER_URL` and `ARC_MCP_TOOLS` are configured, the researcher may call allowlisted external tools with structured arguments and convert their results into evidence. The recommended demo integration is GitHub MCP for repository, code, issue, pull request, and release evidence.

## What The System Does

1. `Clarifier` checks whether the user request is specific enough.
2. `Planner` writes a research brief and decomposes the topic into focused plan items.
3. `ResearchSupervisor` emits ODR-style `think_tool`, `ConductResearch`, and `ResearchComplete` tool calls.
4. `Researcher` runs a bounded tool loop for each delegated unit: `web_search`, optional external `mcp_tool`, or completion.
5. `Retriever` grounds uploaded documents with child chunk retrieval, parent/neighbor expansion, dense retrieval, BM25, graph signal fusion, and reranking.
6. `Reporter` writes topic-specific sections from notes and evidence. It does not use fixed demo sections.
7. `Verifier` and `RAGEvaluator` check citation coverage, evidence sufficiency, source diversity, context precision, and unsupported sections.
8. `RunLedger`, SQLite storage, telemetry, and LangGraph checkpoints make the run inspectable and replayable.

## Technology Fit

- `LangGraph`: fits the conditional research workflow: plan, delegate, run tools, verify, revise, and finalize.
- `FastAPI`: provides a small local product API for jobs, documents, runs, traces, evaluation, and runtime config.
- `Qdrant`: stores dense vectors for contextual grounding.
- `SQLite FTS5/BM25`: adds lexical recall for exact terms, paper names, component names, and metrics.
- `LightRAG-inspired graph signal`: extracts entities and relationships, then fuses graph hits with dense/BM25 candidates before rerank.
- `Qwen/DashScope rerank`: reorders fused candidates with a query-aware reranker in real-provider mode.
- `Celery/Redis`: optional single-node API/worker separation for strict demo runs, not a distributed platform claim.
- `MCP`: optional external tool interface. GitHub MCP is the preferred extension because it adds developer source-of-truth evidence that Tavily-style web search and local RAG do not cover as precisely.
- `OpenAI-compatible providers`: let the same contracts work with DeepSeek, Qwen/DashScope-compatible endpoints, OpenAI-style APIs, and deterministic test doubles.

## MCP Recommendation

Do not connect a full "research assistant" MCP as the default external tool. That duplicates this project's planner/supervisor/reporter and makes the architecture look confused.

Best fit for this repo:

- First choice: [GitHub MCP Server](https://github.com/github/github-mcp-server) through the official remote read-only endpoint. This adds repository, code, issue, PR, and release evidence for technical research topics.
- Second choice: a paper/search MCP only when the demo topic genuinely needs scholarly metadata outside the existing search provider.
- Avoid: MCP servers marketed as complete deep-research agents or generic web-search duplicates. They overlap with the core product instead of extending it.

Future direction: this project itself is a reasonable MCP Server candidate, but that should be a separate outward-facing facade exposing tools like `run_research`, `search_local_corpus`, and `inspect_research_run`. It should not reintroduce the removed local workbench that called the app from inside itself.

## API Surface

- `POST /v1/research/clarify`
- `POST /v1/research/runs`
- `GET /v1/research/runs`
- `GET /v1/research/runs/{run_id}`
- `GET /v1/research/runs/{run_id}/trace`
- `GET /v1/research/runs/{run_id}/evaluation`
- `POST /v1/research/runs/{run_id}/replay`
- `POST /v1/research/jobs`
- `GET /v1/research/jobs/{job_id}/status`
- `GET /v1/research/jobs/{job_id}/result`
- `POST /v1/documents`
- `POST /v1/documents/ingest`
- `GET /v1/documents/search`
- `GET /v1/runtime/config`
- `GET /v1/runtime/provider-check`
- `DELETE /v1/research/history`

There are no `/v1/memory` endpoints in the current core.

## Configuration

Start from `.env.example`.

Important settings:

```text
ARC_STRICT_PROVIDERS=true
ARC_MODEL_PROVIDER=openai_compatible
ARC_MODEL_BASE_URL=...
ARC_MODEL_API_KEY=...
ARC_EMBEDDING_PROVIDER=openai_compatible
ARC_EMBEDDING_BASE_URL=...
ARC_EMBEDDING_API_KEY=...
ARC_SEARCH_PROVIDER=tavily
ARC_SEARCH_API_KEY=...
ARC_QDRANT_URL=http://127.0.0.1:6333
ARC_RERANK_PROVIDER=dashscope
ARC_RERANK_API_KEY=...
```

Optional external MCP:

```text
ARC_MCP_ENABLED=true
ARC_MCP_SERVER_URL=https://api.githubcopilot.com/mcp/readonly
ARC_MCP_TOOLS=search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release
ARC_MCP_AUTH_REQUIRED=true
ARC_MCP_AUTH_TOKEN=<github-token>
ARC_MCP_PROMPT=Use GitHub MCP for repository, code, issue, pull request, and release evidence; use Tavily for broader web context.
```

## Run And Test

Install:

```powershell
pip install -e .[dev]
```

Run tests:

```powershell
pytest
```

Start API:

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

## Study Path

Read in this order:

1. `docs/architecture.md`
2. `docs/source-map.md`
3. `docs/learning/zh/ai_research_copilot_learning_guide_zh.md`
4. `src/agentic_research_copilot/schemas.py`
5. `src/agentic_research_copilot/providers.py`
6. `src/agentic_research_copilot/graph_runtime.py`
7. `src/agentic_research_copilot/pipeline.py`
8. `src/agentic_research_copilot/agents`
9. `src/agentic_research_copilot/retrieval/store.py`

## Interview Framing

The strongest story is:

> I built a single-node AI research copilot inspired by Open Deep Research. The core difficulty is not CRUD; it is turning an open-ended question into a supervised research graph with structured planning, bounded tool use, hybrid retrieval, citation-locked report generation, and replayable evaluation artifacts.

Do not overclaim distributed execution, enterprise memory, browser automation, or a general agent platform. The project is strongest when described as an inspectable research runtime with a credible Agentic RAG stack.
