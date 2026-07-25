# Source Map

## Reuse Plan

### From `open_deep_research` (primary reference)

- `langgraph.json`: graph entrypoint wiring for the Deep Researcher graph.
- `src/open_deep_research/deep_researcher.py`: LangGraph `StateGraph` runtime with main graph, supervisor subgraph, researcher subgraph, parallel research delegation, compression, and final report generation.
- `src/open_deep_research/state.py`: state model for messages, research briefs, raw notes, compressed notes, and final output payloads.
- `src/open_deep_research/prompts.py`: clarification, research brief, supervisor, researcher, compression, and report prompts.
- `src/open_deep_research/configuration.py`: runtime model, search, clarification, concurrency, and iteration configuration.
- `src/open_deep_research/utils.py`: search tools, native web search helpers, MCP tools, token-limit handling, note extraction, summarization, and tool wiring.
- `src/legacy/utils.py`: historical `split_and_rerank` path for chunking provider raw content before selecting query-relevant passages.
- `tests/evaluators.py`: source-quality, groundedness, helpfulness, and trajectory evaluators used as offline quality checks.
- No standalone runtime source-quality policy/filter was found in the inspected Open Deep Research reference; source quality is treated as an evaluation concern.
- Supervisor autonomy note: the inspected ODR graph lets the LLM decide between `think_tool`, `ConductResearch`, and `ResearchComplete`. This repo mirrors that runtime boundary with a local `research_supervisor` node; each `ConductResearch` call carries selected tools, query rewrites, grounding mode, and sufficiency criteria. Deterministic route hints are retained only for offline tests and defensive fallback inputs.

### From `PraisonAI` (secondary reference)

- `src/praisonai-ts/src/memory/memory.ts`: memory abstraction and retrieval interface
- `src/praisonai-ts/src/memory/file-memory.ts`: persistent memory and replay-friendly file storage
- `src/praisonai-agents/praisonaiagents/memory/memory.py`: short/long/entity/user memory operations, quality-aware search, and memory context construction
- `src/praisonai-agents/praisonaiagents/knowledge/knowledge.py`: knowledge indexing/search shape used as a reference for retrieval-backed context
- `src/praisonai-agents/praisonaiagents/knowledge/readers.py`: reader registry/protocol shape for future source-reader adapters
- `src/praisonai/praisonai/adapters/readers.py`: MarkItDown/Text/URL/Directory reader shape used as a reference for local document parsing boundaries
- `src/praisonai-agents/praisonaiagents/knowledge/chunking.py`: chunking strategy registry used as a reference for document ingestion
- `src/praisonai-agents/praisonaiagents/rag/context.py`: token-aware context assembly and chunk deduplication
- `src/praisonai-ts/src/observability/index.ts`: trace / telemetry abstraction
- `src/praisonai/praisonai/acp/session.py`: persistent sessions and resume flow
- `src/praisonai-bot/praisonai_bot/bots/_outbox.py`: durable queue / persistence pattern
- `ARCHITECTURE.md`: replay, checkpointing, and run ledger concepts

### From `LightRAG` (conceptual RAG reference)

- Paper: `https://arxiv.org/abs/2410.05779`
- Repository: `https://github.com/HKUDS/LightRAG`
- Conceptual ideas used here: graph-enhanced text indexing, entity/relation
  signals, vector + graph retrieval fusion, and lightweight incremental updates.
- No LightRAG runtime code is copied into this repo. The local implementation is
  a product-specific, lightweight entity co-occurrence index fused into the
  existing contextual-retrieval + Qdrant dense + SQLite FTS5/BM25 + rerank grounding layer.

### Original modules in this repo

- public API
- `providers.py`: OpenAI-compatible model adapter and deterministic test doubles
- `source_reader.py`: provider raw-content reading with `extract`, `model_compress`, and `chunk_rerank_compress` strategies, including chunk rerank plus neighbor-window expansion before compression
- `document_reader.py`: local text/Markdown/HTML reader with heading-aware section metadata, plus optional PyMuPDF PDF block/table-aware page segmentation before grounding retrieval
- `workflow.py`: query building, note compression, and source formatting
- `search.py`: Tavily, Exa, Perplexity, arXiv, PubMed, Linkup, OpenAI native web search, Anthropic native web search, DuckDuckGo, Brave, and SerpAPI adapters behind one tool contract
- `provider_validation.py`: strict real-provider readiness checks for demos without exposing secret values
- `storage.py`: persistence glue for memory, documents, and runs
- `ledger.py`: checkpoint ledger and replay-friendly run records
- `routing.py`: offline/test route hints and fallback corpus-profile based decisions
- `graph_runtime.py`: LangGraph runtime that wires supervisor, memory, planner, research_supervisor, parallel research, reporter, verifier/evaluator, memory-write, and finalization nodes
- `pipeline.py`: product facade, runtime selection, supervisor orchestration fallback, `ConductResearch` route materialization, concurrent plan-item research, revision guards, and cross-module glue
- `evaluation.py`: local Agentic RAG, sufficiency, tool-selection, source-quality, and citation quality checks
- `scripts/run_llm_judge_eval.py`: optional Open Deep Research-style LLM-as-judge evaluation artifact for saved demo reports
- `scripts/run_ragas_eval.py`: optional Ragas evaluation artifact over saved report and trace evidence
- `scripts/start_real.ps1`: strict real-provider startup that validates providers, prepares Redis/Qdrant, starts a Celery worker, and serves the workspace
- `retrieval/store.py`: parent-child chunking, indexing-time contextual retrieval prefixes, LightRAG-inspired entity/relation graph indexing, Qdrant dense vectors, SQLite FTS5/BM25 keyword indexing, RRF/DBSF fusion, local fallback fusion, graph-score fusion, parent/neighbor context expansion, and reranking contract
- `retrieval/fulltext.py`: single-node SQLite FTS5 keyword index using SQLite's `bm25()` ranking so lexical recall is a real BM25 path rather than a pseudo hashed-token vector
- `retrieval/rerank.py`: Qwen/DashScope reranker plus offline rule rerank fallback, disabled by strict demo mode
- `memory/store.py`: layered session, canonical fact, and summary memory with embedding-assisted recall and governance metadata
- report composition and source index formatting
- deployment and local dev setup
- integration glue between memory, retrieval, report generation, and search

## How to study this repo

1. Read the `open_deep_research` entries first; ODR is the main target for learning the research workflow.
2. Read `graph_runtime.py` next; it is the local LangGraph adaptation of the Open Deep Research-style graph.
3. Read the `PraisonAI` entries only as secondary material for memory, readers, persistence, and observability.
4. Trace the corresponding modules in this repo:
   - `workflow.py` for planning and note compression
   - `document_reader.py` and `retrieval/store.py` for parsing, segmenting, chunking, indexing, and reranking
   - `graph_runtime.py` for active LangGraph orchestration
   - `pipeline.py` for orchestration and checkpoints
   - `storage.py` for local persistence
   - `telemetry.py` and `ledger.py` for run tracking
