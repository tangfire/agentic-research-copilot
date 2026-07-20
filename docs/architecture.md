# Architecture

## Goal

Build an AI research copilot that can supervise a run, plan, search, retrieve contextual evidence, recall memory, verify claims, evaluate RAG quality, and report with citations.

The product is best described as **AI Research Copilot for complex questions**.
The technical layer uses LangGraph + Agentic RAG: it avoids a plain
`question -> top-k chunks -> answer` pipeline by making planning, tool selection,
query rewrite, evidence sufficiency, citation checks, and revision explicit
runtime states.

## System Flow

```mermaid
flowchart LR
  UI["Web UI"] --> API["FastAPI"]
  API --> Graph["LangGraph StateGraph"]
  Graph --> Supervisor["Supervisor"]
  Supervisor --> Planner["Planner Agent"]
  Supervisor --> Research["Researcher / Retriever"]
  Planner --> Routes["Routing + Tool Selection"]
  Routes --> ParallelResearch["Concurrent Research Workers"]
  ParallelResearch --> Research
  Routes --> Retriever["Qdrant Grounding Layer"]
  Research --> Evidence["Evidence Store"]
  Retriever --> Evidence
  Evidence --> Verifier["Verifier Agent"]
  Verifier --> Reporter["Report Generator"]
  Reporter --> Storage["Run / Artifact Store"]
  API --> Memory["Layered Memory Service"]
  API --> Obs["Observability / Cost / Replay"]
  Obs --> Storage
```

## Core Modules

### Product Surfaces

- Portal: submit research questions, inspect the generated plan, routes, report, and citations.
- Admin: inspect sources, memory, telemetry, checkpoints, and runtime config.
- The first UI is a dependency-light static research workspace served by FastAPI so the AI core remains the center of the project.
- The workspace surfaces provider readiness, job progress, report review, route inspection, memory governance, and trace timeline without adding a heavy frontend build step.
- A heavier frontend can be added later when streaming jobs, authentication, collaborative editing, or richer report review becomes necessary.

### Orchestration Runtime

- The default runtime is a LangGraph `StateGraph`, matching the graph-first direction used by Open Deep Research.
- The graph nodes are `supervisor_start`, `memory_recall`, `planner`, `parallel_research`, `reporter`, `verifier_evaluator`, `revision_prepare`, `memory_write`, and `finalize`.
- The graph compiles with a single-node LangGraph SQLite checkpointer by default; it falls back to in-process `MemorySaver` only when strict provider mode is disabled.
- The older custom workflow remains available through `ARC_ORCHESTRATION_RUNTIME=custom` for comparison and offline fallback.
- `ARC_STRICT_PROVIDERS=true` turns the local app into a real-provider demo: missing model/search/embedding/rerank/Qdrant/checkpointer configuration fails startup instead of silently downgrading.
- This repo does not import Open Deep Research as a runtime dependency; it uses the same graph-shaped design pattern with product-specific nodes and schemas.

### Job Manager

- Accepts research jobs separately from final run artifacts.
- Tracks `queued`, `running`, `completed`, `failed`, and `cancelled` states.
- Executes offline/test research jobs through a single-worker background queue.
- Submits strict local demo jobs to Celery over Redis for a single-node API/worker split.
- If `ARC_JOB_QUEUE_BACKEND=celery` is used in strict provider mode, enqueue failures fail the job instead of falling back to the in-process worker.
- Celery mode requires `ARC_QDRANT_URL`; embedded Qdrant paths are single-process only and are not used for API/worker separation.
- Persists job and run records in SQLite and refreshes status reads from SQLite, so a local API process can observe updates written by a separate worker process.
- Records attempts, retry errors, cancellation requests, and timeout metadata.
- Records handoffs, trace events, and failure states so a run can be inspected after completion.
- This is intentionally not described as a production distributed scheduler; Celery/Redis is optional process separation for personal deployment, while multi-worker production scheduling remains out of scope.

### Planner Agent

- Normalizes the user request.
- Breaks it into sub-questions.
- Decides whether clarification is needed.
- Emits a schema-backed planning contract.

### Research Agents

- Run plan-item research concurrently with a configurable worker budget.
- Use a pluggable search registry inspired by Open Deep Research: Tavily, Exa, Perplexity, arXiv, PubMed, Linkup, OpenAI native web search, Anthropic native web search, plus DuckDuckGo, Brave, and SerpAPI adapters.
- Summarize evidence with citations.
- Return structured findings instead of raw text only.

### External vs Internal Evidence

- External search is used for fresh web evidence, papers, and public references.
- Internal grounding is used for uploaded documents, project notes, and prior runs.
- A corpus profile summarizes what uploaded/project sources exist so the router can decide whether internal grounding is actually available.
- The planner/router can route a task to one or both sources.
- Each route records selected tools, rewritten web/internal queries, minimum evidence thresholds, minimum source diversity, and sufficiency criteria.
- The verifier compares the evidence channels instead of assuming they say the same thing.
- The reporter keeps source attribution separate so the final answer stays traceable.
- Qdrant backs the internal embedding index, while the route contract stays stable.

### Grounding Layer

- Indexes uploaded PDFs, notes, and prior project artifacts.
- Performs contextual chunking, Qdrant dense/sparse retrieval, RRF/DBSF fusion, and reranking.
- Preserves source attribution so each report section can be traced back.
- Works as the internal evidence channel, distinct from web search.

### Why RAG fits this project

- The product is evidence-driven, not a free-form creative generator.
- Answers should be grounded in uploaded documents, prior runs, memory, and web evidence.
- Retrieval is used as a grounding layer for the planner, researcher, and reporter.
- Memory stores preferences and distilled conclusions; retrieval brings back supporting evidence.
- Fine-tuning is a poor fit because the source set changes and provenance matters.
- Plain keyword RAG is too weak here, so the repo uses contextual chunking, Qdrant named dense/sparse vectors, fusion, and reranking first.
- Agentic RAG adds query rewrite, tool selection, multi-query retrieval, sufficiency scoring, and revision-triggering quality gates.
- GraphRAG, RAPTOR, or knowledge-graph extraction are deliberately out of scope for v1; they add weight without being necessary for a deep-research copilot.
- For larger corpora, the same contract can be swapped to pgvector or another hybrid retrieval backend without changing the orchestration flow.
- The first hybrid implementation uses Qdrant dense vector search plus sparse token vectors and fuses results with `RRF` or `DBSF` before applying a pluggable reranker.
- The default reranker calls Qwen/DashScope `qwen3-rerank` when an API key is configured. Deterministic `rule_diversity_chunk_bonus` is reserved for offline runs and tests, and is disabled by `ARC_STRICT_PROVIDERS=true`.
- For DashScope, the reranker accepts the generic `https://dashscope.aliyuncs.com/compatible-mode/v1` base URL and maps it to the rerank service endpoint internally, because the generic compatible-mode endpoint does not expose `/reranks` directly.

### Memory Service

- Stores session notes, canonical facts, and topic summaries.
- Supports short-term context and long-term memory.
- Uses explicit write and recall rules instead of a flat key/value store.
- Ranks memory recall with lexical matching, embedding-assisted semantic similarity, confidence, memory layer, and governance status.
- Adds governance metadata to canonical memory: conflicts are retained, marked `needs_review`, and exposed through a governance report instead of being overwritten silently.
- This follows the useful subset of PraisonAI's memory direction without turning the project into a generic memory platform. PraisonAI also exposes broader short/long/entity/user memory, quality-aware search, session stores, and knowledge retrieval; this repo keeps only the product-specific pieces needed for deep research.

### Model Provider

- Exposes an OpenAI-compatible chat and embeddings adapter.
- Uses deterministic test doubles by default for CI and offline development.
- Allows chat-only providers to use deterministic local embeddings in offline mode, or a separate OpenAI-compatible embedding endpoint for strict real-provider demos.
- Keeps planner, verifier, and reporter outputs schema-backed.

### Verifier Agent

- Checks citation completeness.
- Detects contradictions or missing evidence.
- Flags weak claims before final report generation.

### Report Generator

- Writes the final answer or report.
- Keeps source references attached to each section.
- Follows the Open Deep Research final-report pattern: compressed findings are
  synthesized by the report model, while `citation_indexes` are mapped back to
  existing evidence so the model cannot invent source objects.

### Observability Layer

- Tracks token usage, tool calls, handoffs, and latency.
- Stores traces, failures, and replay inputs.
- Helps explain why a run succeeded or failed.
- Persists run-level checkpoints in SQLite and supports a single-node LangGraph SQLite checkpointer for graph execution. Full distributed resume remains out of scope; the README/API calls the current behavior single-node checkpointing plus durable trace/replay.
- Exposes `GET /v1/runtime/provider-check` so demos can verify real-provider readiness without exposing secret values.

### RAG Evaluation Layer

- Scores plan coverage, retrieval hit rate, contextual retrieval contribution, evidence sufficiency, tool-selection coverage, query-rewrite count, source quality, context precision, context recall, faithfulness proxy, citation precision, citation source coverage, source diversity, and unsupported sections.
- Emits an `evaluation` trace event and a `rag.evaluated` checkpoint for each run.
- Feeds weak citation or retrieval quality back into the supervisor revision loop.
- The inspected Open Deep Research reference keeps source quality as an evaluator concern rather than a separate runtime source-filtering policy. This repo follows that choice and does not add a standalone source quality policy layer for v1.
- `scripts/run_llm_judge_eval.py` provides an optional Open Deep Research-style
  LLM-as-judge artifact for demo reports, scoring research depth, source quality,
  analytical rigor, structure, groundedness, and completeness.
- `scripts/run_ragas_eval.py` provides an optional Ragas artifact over saved demo
  evidence when the `.[eval]` extra is installed.
- Runtime RAG metrics remain lightweight local proxies. They are designed to make retrieval and citation failure modes visible without adding benchmark cost to every request.

## Interfaces

- `POST /v1/research/jobs`: submit an asynchronous research job and receive a job envelope.
- `GET /v1/research/jobs`: inspect submitted jobs.
- `GET /v1/research/jobs/{job_id}/status`: inspect queued/running/completed/failed/cancelled state.
- `GET /v1/research/jobs/{job_id}/result`: fetch the completed run artifact.
- `POST /v1/research/jobs/{job_id}/cancel`: request cancellation for queued/running jobs.
- `POST /v1/research/runs`: synchronous run endpoint for tests and simple clients.
- `GET /v1/research/runs/{run_id}/status`: inspect completed run timing and source count.
- `GET /v1/research/runs/{run_id}/result`: inspect report, issues, evidence, routes, and checkpoints.
- `GET /v1/research/runs/{run_id}/evaluation`: inspect RAG and citation quality gates.
- `GET /v1/research/runs/{run_id}/trace`: inspect the handoff and tool trace for a run.
- `GET /v1/memory?layer=&topic=&run_id=&session_id=&tag=`: filter memory records.
- `GET /v1/memory/governance`: inspect canonical memory conflicts and review-required records.
- `GET /v1/runtime/config`: inspect agents, tools, routing, storage, and quality gates.

## Data Model

- `research_session`
- `research_job`
- `research_task`
- `evidence_item`
- `memory_item`
- `report_version`
- `run_trace`
- `run_ledger`

## Inspiration Split

- `open_deep_research`: LangGraph StateGraph orchestration, supervisor/researcher split, planning, parallel research, citations, report generation
- `PraisonAI`: memory, handoff, observability, evaluation, workflow patterns

## MVP Sequence

1. Search-only report generation.
2. Add contextual document grounding.
3. Add memory and user preferences.
4. Add verification and replay.
5. Add evaluation and scoring.
6. Wire real search and model providers for demo runs.
7. Harden the LangGraph runtime with richer interruption/resume and streaming trace support.
