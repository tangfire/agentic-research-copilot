# Source Map

## Primary Reference: Open Deep Research

The main architectural reference is Open Deep Research. The repo adapts these ideas:

- LangGraph-style state graph for long-running research.
- A clarification gate before expensive research starts.
- A supervisor that emits `think_tool`, `ConductResearch`, and `ResearchComplete`.
- Parallel research units when a plan has independent sub-questions.
- Bounded researcher loops over search/tool calls.
- Compression of source material before report synthesis.
- Citation-backed final report generation.
- Evaluation of source quality, groundedness, usefulness, and trajectory outside the hot path.
- MCP as a configurable external tool allowlist with tool catalogs and structured tool arguments.

This repo does not import Open Deep Research at runtime. It uses ODR as a learning target and adapts the shape into local runtime modules.

## Secondary Reference: PraisonAI

PraisonAI remains only a secondary reference for:

- reader registry ideas
- persistence/replay vocabulary
- agent handoff vocabulary
- observability and run-ledger concepts

It should not be presented as the primary memory reference. The current memory layer is closer to a small Mem0-inspired session/user/project store and is implemented directly in SQLite.

## Memory Reference: Mem0

Mem0 is used as a conceptual reference for memory layering:

- user-level preferences
- project/team constraints
- session facts
- retrieval of relevant memories before planning

No Mem0 SDK is imported at runtime. The local implementation lives in `agent.py`, `schemas.py`, and `storage.py`, and project-scope memory is synced into the local document store for retrieval.

## Conceptual RAG Reference: LightRAG

LightRAG is used as a conceptual reference for graph-enhanced retrieval:

- extract canonical entities
- extract relationships
- separate local entity signals from global relation signals
- fuse graph hits with vector/keyword retrieval

No LightRAG code is copied. The local graph layer is a bounded retrieval signal inside `retrieval/store.py`.

## Original Modules

- `schemas.py`: structured contracts for requests, plan items, routes, evidence, reports, trace, and evaluation.
- `agent.py`: conversational session facade, memory extraction, plan confirmation, and research job binding.
- `providers.py`: OpenAI-compatible real model provider, structured JSON calls, embeddings, graph extraction, source compression, and report composition.
- `dev_fixtures.py`: explicit fixture provider for tests and local regression scripts; it is not a product runtime provider.
- `graph_runtime.py`: LangGraph orchestration path.
- `pipeline.py`: application facade that wires providers, retrieval, agents, storage, jobs, telemetry, and runtime config.
- `agents/planner.py`: delegates planning to the model provider.
- `agents/supervisor.py`: normalizes ODR-style supervisor tool calls.
- `agents/researcher.py`: bounded search/MCP/research-complete loop for each research unit, including structured MCP arguments and iteration trace metadata.
- `agents/reporter.py`: builds the final report through the model provider.
- `agents/verifier.py`: checks report quality and revision need.
- `retrieval/store.py`: contextual retrieval, child chunks, parent/neighbor expansion, Qdrant dense search, SQLite BM25, graph fusion, and rerank.
- `mcp_tools.py`: external MCP client adapter with allowlist filtering, GitHub read-only URL handling, tool catalog extraction, structured argument calls, and query fallback.
- `search.py`: search provider adapters behind a common tool contract.
- `source_reader.py`: provider raw-content extraction/compression.
- `document_reader.py`: local file parsing before indexing.
- `evaluation.py`: proxy RAG and citation metrics.
- `storage.py`: SQLite persistence for documents, jobs, runs, agent sessions, messages, plan drafts, memory, and trace artifacts.
- `ledger.py`: in-process run/job ledgers.
- `telemetry.py`: trace events for handoffs, tools, checkpoints, evaluation, and failures.

Removed from the current core:

- `memory/store.py`
- the former local workbench MCP server
- local workbench MCP defaults

Reintroduced in the agent layer:

- `POST /v1/agent/sessions`
- `POST /v1/agent/sessions/{session_id}/messages`
- `POST /v1/agent/sessions/{session_id}/confirm-plan`
- `GET /v1/agent/sessions/{session_id}/memory`
- `POST /v1/memory`
- `GET /v1/memory`
- `DELETE /v1/memory/{memory_id}`

## Study Order

1. Read `docs/product-positioning.md` to understand the runtime boundary.
2. Read `agent.py` to understand the conversation, memory, and confirmation layer.
3. Read `schemas.py` to understand the data contracts.
4. Read `providers.py` to see where real LLM calls happen.
5. Read `graph_runtime.py` to understand the runtime graph.
6. Read `pipeline.py` to see how the application is assembled.
7. Read `agents/` to understand which code is thin orchestration and which decisions are delegated to the provider.
8. Read `retrieval/store.py` for the Agentic RAG implementation.
9. Read `evaluation.py`, `storage.py`, and `telemetry.py` for replay and quality evidence.
10. Read `mcp_tools.py` only after the main chain is clear; GitHub MCP is a developer evidence channel, not the whole runtime.

## Interview Checkpoint

You should be able to explain:

- why this is not plain top-k RAG
- why the supervisor uses structured tool calls
- how `RetrievalRoute` controls external/internal/hybrid evidence
- how child retrieval plus parent/neighbor expansion differs from a naive chunk return
- why GitHub MCP is an external developer evidence source rather than a replacement for Tavily or local RAG
- how `mcp_tool_name` and `mcp_tool_args` flow from provider decision to MCP evidence and trace
- why memory is implemented as a thin local agent layer rather than an enterprise personalization system
- what would be required to expose this project as a clean MCP Server later
- why this project should be discussed as an inspectable runtime experiment rather than a commercial Codex replacement
