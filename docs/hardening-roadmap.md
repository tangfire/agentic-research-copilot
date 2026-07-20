# Hardening Roadmap

## Current Position

This repository should be presented as an **AI Research Copilot for complex questions**:
it plans research, routes between external search/contextual retrieval/memory, verifies
citations, evaluates RAG quality, and persists inspectable traces.

It should not be described as a production-grade generic agent platform. The current
architecture is intentionally product-specific: strong enough for an interview demo,
small enough to explain and maintain.

## Reference Check

### Open Deep Research

The local Open Deep Research reference is strongest in these areas:

- LangGraph-first research orchestration.
- Supervisor/researcher split with parallel delegated research.
- Tool-calling research loop with reflection.
- Compression of research findings before final report writing.
- Citation-backed final reports.
- Evaluation with overall quality, source quality, groundedness, correctness,
  completeness, relevance, and structure.

The inspected reference treats `source_quality_score` as an evaluation concern. It does
not add a separate runtime source-quality filter/policy layer that blocks search results
before report generation. This repo follows that shape for v1: source quality is scored
and surfaced in traces/evaluation, while provider ranking and user-selected sources stay
outside a hard-coded local policy.

### PraisonAI

The local PraisonAI reference is strongest in these areas:

- Broad memory runtime with short-term, long-term, entity, and user memory.
- Quality-aware memory writes and searches.
- Rerank hooks on retrieval/memory search.
- Session, checkpoint, replay, trace, and persistence concepts.
- A broad SDK/plugin shape for many agent use cases.

This repo should not copy PraisonAI as a runtime dependency or broaden into that kind of
SDK. The useful subset is memory governance, quality-aware recall, trace/replay concepts,
and the pluggable reranker interface.

## Next Optimizations

### P0: Expand The Evaluation Dataset

Status: implemented for v1.

Why: a tiny dataset proves the script works, but it is thin when an interviewer asks
how regressions are caught. The repo now keeps a 12-case deterministic regression set
and emits an eval report so retrieval, citation, source-quality, and memory behavior can
be inspected without claiming a full external benchmark.

Target coverage:

- product positioning
- LangGraph runtime and checkpoint boundary
- routing/tool selection/query rewrite
- dense/sparse Qdrant hybrid retrieval
- citation and evidence verification
- memory governance and conflict review
- source-quality evaluation boundary
- job queue/trace/replay boundary
- model/search provider configuration
- revision loop failure handling

### P1: Add A Reranker Interface

Status: implemented for v1.

Why: current retrieval already uses Qdrant dense/sparse fusion plus a pluggable reranker
interface. The default runtime attempts a real Qwen/DashScope rerank call. Deterministic
diversity/chunk-position scoring remains available for tests/offline runs, while
`ARC_STRICT_PROVIDERS=true` disables rerank fallback for demos.

- default Qwen/DashScope `qwen3-rerank` when credentials are configured
- deterministic `rule_diversity_chunk_bonus` fallback for tests and offline runs
- strict real-provider mode that raises on missing or failed reranker calls
- optional cross-encoder reranker later
- trace metadata that records which reranker was used

This borrows PraisonAI's rerank-hook idea without pretending a local cross-encoder is
already running.

### P1: Keep Source Quality As Evaluation, Not Runtime Filtering

Status: intentionally keep as-is for v1.

Why: Open Deep Research's local reference scores source quality in evaluators rather than
hard-filtering search results at runtime. Hard filters are risky for a general research
assistant because they can silently remove useful primary sources, niche pages, or
freshly published material.

Better v1 behavior:

- show `source_quality_score`
- include weak-source notes in evaluation
- prefer `arxiv`, `pubmed`, Tavily, Exa, Perplexity, Brave, or SerpAPI for demos
- keep final reports citation-backed
- do not add runtime source scoring/filtering unless the product later needs a
  domain-specific source policy
- document that strict domain allow/deny lists are future product policy, not core agent
  architecture

### P1: LLM-Driven Final Report Synthesis

Status: implemented.

Why: Open Deep Research does not leave the final report as a template-only
assembly. It compresses research findings and then asks a final report model to
synthesize a comprehensive answer with citations. This repo mirrors that shape
inside the structured report contract: the report model emits section drafts and
citation indexes, while the app maps those indexes back to existing evidence objects.

- LLM-generated `ReporterSectionDraft` objects
- citation indexes instead of model-invented URLs
- fallback to deterministic/template sections for tests and offline mode
- trace still records reporter model usage

### P1: Optional LLM-As-Judge Eval Artifact

Status: implemented as a demo/eval script.

Why: Open Deep Research evaluates with Deep Research Bench and LLM-as-judge
criteria. This repo keeps low-cost proxy evals in the runtime and adds
`scripts/run_llm_judge_eval.py` for interview/demo artifacts. It is deliberately
not run on every request.

- research depth
- source quality
- analytical rigor
- structure
- groundedness
- completeness

### P1: Optional Ragas Eval Artifact

Status: implemented as an optional eval script.

Why: Ragas is useful for learning and interview artifacts, but it should not be a
serving dependency for every research request. The repo now exposes
`scripts/run_ragas_eval.py`, which reads a saved demo report and trace evidence,
builds a single-case Ragas dataset, and writes `examples/ragas-report.json` when
the `.[eval]` extra is installed. Runtime evaluation still uses deterministic proxy
metrics so tests stay stable and local runs remain cheap.

Do not describe this as a full benchmark unless a larger labeled dataset is added.

### P2: Persistent LangGraph Checkpointer

Status: implemented as a single-node option with strict-mode guardrails.

Why: local/personal deployment benefits from a file-backed checkpointer, but it should
not become a distributed-platform claim. The graph uses a LangGraph SQLite checkpointer
by default and still falls back to `MemorySaver` defensively when strict provider
mode is disabled. In strict demo mode, missing or failed SQLite checkpointing fails
startup. Durable run artifacts, traces, and app-level checkpoints are always
persisted by the app in SQLite.

Do not describe this as full distributed durable execution; describe it as single-node
checkpointing plus durable trace/replay.

### P2: Production Queue And Streaming

Status: implemented for strict single-node worker separation.

Why: Redis/Celery can be useful when the API process and worker process should be
separated on one machine. The strict real-provider config now uses Celery over
Redis, validates broker/result backend configuration, requires Qdrant server mode
for API/worker separation, persists job/run records in SQLite, and refreshes status
reads from SQLite so the API can observe worker-written state. The in-process queue
remains available for offline tests. Streaming tokens, auth, multi-tenancy, rate
limits, and human review are still deferred because this is a personal research
copilot, not a SaaS platform.

### P1: Strict Real-Provider Demo Mode

Status: implemented.

Why: the resume/interview demo should prove the app can call real remote model,
embedding, search, rerank, Qdrant, and checkpointing services. Test doubles are still
valuable for CI, but they should not silently hide missing keys during demos.

- `ARC_STRICT_PROVIDERS=true`
- startup validation for chat, embedding, search, rerank, Qdrant, and LangGraph
  checkpoint configuration
- strict search wrapper that raises when a real provider returns no evidence
- strict rerank and Qdrant paths that raise instead of falling back to local rules
- `GET /v1/runtime/provider-check` and `scripts/check_real_providers.py`
- `scripts/start_real.ps1` for local strict startup without committing secrets

## Resume-Safe Boundary

Strong phrasing:

> Built a LangGraph-based AI Research Copilot that decomposes complex questions, routes
> between external search/contextual retrieval/memory, uses Qdrant dense+sparse hybrid
> RAG, verifies citations, evaluates RAG quality, and persists traceable research reports.

Avoid:

- "production-grade distributed agent platform"
- "distributed LangGraph durable execution"
- "full Ragas benchmark" unless a larger labeled dataset is added
- "cross-encoder reranking" until that provider is actually wired
- "runtime source-quality policy" because v1 keeps source quality in evaluation
