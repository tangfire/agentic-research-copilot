# Hardening Roadmap

## Current Position

This repository should be presented as an **AI Research Copilot for complex questions**:
it plans research, searches and reads sources, routes between contextual
retrieval/memory when useful, verifies citations, evaluates quality, and persists
inspectable traces.

Short positioning:

> An AI Research Copilot for complex questions that uses planning, search,
> reading, retrieval augmentation, memory recall, citation verification, and
> evaluation replay to generate traceable research reports.

The main product path is **planning -> search/reading -> synthesis ->
verification/evaluation -> replay**. RAG is a contextual grounding, already-read
evidence cache, and memory-recall layer; it is not the only or primary path.

It should not be described as a production-grade generic agent platform. The current
architecture is intentionally product-specific: strong enough for an interview demo,
small enough to explain and maintain.

## Technology Fit Review

Current verdict: the technology choices are appropriate for the project. The repo
does not appear to be an unrelated stack wrapped in agent/RAG terminology. The
important caveat is presentation: describe the system as a single-node AI
Research Copilot, not as an enterprise research platform.

| Technology | Why it fits | Boundary to keep honest |
| --- | --- | --- |
| LangGraph | The workflow has stateful planning, delegation, bounded loops, verification, revision, memory writes, and replay. | Single-node SQLite checkpointing, not distributed durable execution. |
| FastAPI | The product needs inspectable job, run, document, memory, telemetry, and runtime-config APIs. | Thin local API, not a full SaaS backend. |
| Celery/Redis | Useful for separating API and worker processes during strict real-provider demos. | Single-node worker separation, not production multi-tenant scheduling. |
| SQLite | Good for local run ledgers, memory, documents, job state, telemetry, and checkpoint files. | Local persistence, not horizontal database scaling. |
| Qdrant | Provides a real dense-vector backend for grounding documents. | Retrieval backend only; it does not make the project an enterprise search platform by itself. |
| SQLite FTS5/BM25 | Adds exact term and acronym recall that dense embeddings often miss. | Single-node lexical index, not Elasticsearch/OpenSearch. |
| Reranker | Reorders fused dense/BM25/graph candidates by query relevance. | Qwen/DashScope rerank, not a locally trained cross-encoder unless that is added later. |
| MCP | Configurable tool boundary for grounding search, memory recall, run/eval inspection, and readiness checks. | Local workbench server, not an enterprise MCP gateway. |
| Provider raw-content reader | Reads and compresses source content beyond snippets. | Provider reading layer, not browser automation or crawling. |
| PyMuPDF document reader | Preserves useful PDF page/block/table metadata for grounding. | Not OCR or full document intelligence. |
| LightRAG-inspired graph signal | Adds entity/relation candidate expansion before rerank. | Lightweight graph signal, not a full LightRAG/GraphRAG runtime. |
| Proxy evaluation + optional judge/Ragas | Makes quality failures visible without adding heavy benchmark cost to every run. | Demo/eval artifacts, not a large public benchmark. |

This makes the resume story defensible: the architecture is built around a real
research workflow, and the heavier pieces each support a specific failure mode in
plain RAG. The next risk is not the stack choice; it is an underprepared demo
with an empty corpus, no memory records, or no saved trace/evaluation artifacts.

## July 2026 Resume Demo Status

The resume demo gap has been reduced. `scripts/prepare_resume_demo_assets.py`
now prepares a bounded federated-learning corpus, memory records, and completed
run artifacts under `examples/resume-demo/`. The latest strict-provider run used
OpenAI-compatible chat, Qwen/DashScope embeddings and rerank, Tavily, Qdrant,
SQLite FTS5/BM25, Celery/Redis, LangGraph, and the local MCP workbench.

Current stable assets:

- 5 paper excerpts from 5 federated-learning sources
- 3 project memory records
- MCP readiness 10/10
- 2 completed research runs with saved reports, traces, evaluations, and replay
  data

The detailed experiment log is in `docs/resume-demo-runbook.md`. Keep that
runbook current whenever a demo failure leads to a code, prompt, config, or
evaluation change.

## Reference Check

### Open Deep Research

The local Open Deep Research reference is strongest in these areas:

- LangGraph-first research orchestration.
- Supervisor/researcher split with parallel delegated research.
- Tool-calling research loop with reflection.
- Configurable MCP tool loading through `mcp_config.url` and `mcp_config.tools`.
- Tavily search with provider-returned `raw_content` that is compressed before
  final report writing.
- Compression of research findings before final report writing.
- Citation-backed final reports.
- Evaluation with overall quality, source quality, groundedness, correctness,
  completeness, relevance, and structure.

The inspected reference treats `source_quality_score` as an evaluation concern. It does
not add a separate runtime source-quality filter/policy layer that blocks search results
before report generation. This repo follows that shape for v1: source quality is scored
and surfaced in traces/evaluation, while provider ranking and user-selected sources stay
outside a hard-coded local policy.

ODR's supervisor uses an LLM tool loop with `think_tool`, `ConductResearch`, and
`ResearchComplete`. This repo now mirrors that boundary in the runtime: the
research supervisor emits those tool-call-shaped decisions, and each
`ConductResearch` call carries selected tools, query rewrites, grounding mode,
and sufficiency criteria. Deterministic route hints remain only for offline tests
and defensive fallback behavior.

ODR's MCP integration is not a fixed list of built-in servers. The inspected
mainline exposes `MCPConfig(url, tools, auth_required)` and `mcp_prompt`, then
loads only the configured tool allowlist through `MultiServerMCPClient`. This
repo follows that mechanism with `ARC_MCP_SERVER_URL`, `ARC_MCP_TOOLS`, optional
bearer auth, and trace-visible MCP evidence records.

### PraisonAI

PraisonAI is a secondary reference, not the main project target. The useful inspected
ideas are strongest in these areas:

- Broad memory runtime with short-term, long-term, entity, and user memory.
- Reader registry and document ingestion concepts for file/URL sources.
- MarkItDown-backed document conversion and chunking strategies for knowledge ingestion.
- Quality-aware memory writes and searches.
- Rerank hooks on retrieval/memory search.
- Session, checkpoint, replay, trace, and persistence concepts.
- A broad SDK/plugin shape for many agent use cases.

This repo should not copy PraisonAI as a runtime dependency or broaden into that kind of
SDK. The useful subset is memory governance, quality-aware recall, trace/replay concepts,
the pluggable reranker interface, and a reader/parser extension shape.

## ODR-Aligned Boundary Ledger

These are the boundaries to keep clear when presenting the project:

| Area | ODR reference shape | This repo's current shape | How to present it |
| --- | --- | --- | --- |
| Product path | Plan/research/compress/report/evaluate | Plan, search/read, retrieve, synthesize, verify/evaluate, replay | ODR-style AI Research Copilot, not plain RAG |
| Supervisor/tool loop | LLM supervisor delegates with `think_tool`, `ConductResearch`, and `ResearchComplete` | LLM planner plus ODR-style research supervisor; `ConductResearch` carries tools, queries, mode, and sufficiency criteria | Main orchestration is ODR-aligned; deterministic hints are fallback/test scaffolding |
| Researcher tool loop | Researcher model can call search, `think_tool`, configured MCP tools, or complete | Bounded model-driven researcher action schema over `think_tool`, `web_search`, configured `mcp_tool`, and `ResearchComplete` | ODR-shaped but budgeted for a local project |
| MCP tools | Configured MCP server URL plus explicit tool allowlist; no fixed default server list | `ARC_MCP_SERVER_URL` + comma-separated `ARC_MCP_TOOLS`; optional bearer auth and MCP prompt; local workbench server exposes grounding search, memory recall, run/eval inspection, and readiness checks | Say it mirrors ODR's registry boundary, and the local server is a controlled experiment workbench |
| Web reader | Tavily/provider `raw_content` and compression | Tavily/provider `raw_content` with extract/model/chunk-rerank compression plus neighbor expansion | ODR-level v1 reading boundary, not browser automation |
| Source quality | Evaluation/judge surface, not runtime blocking | Runtime metrics and optional judge artifacts expose weak sources | Do not claim permanent high-quality source filtering |
| Report synthesis | Model-generated report from compressed findings | Citation-aware model sections with backend-locked evidence IDs | Strong; this is one of the main ODR-aligned upgrades |
| Local RAG | Not the primary ODR path | Added product-specific grounding with document reader and Qdrant | RAG is grounding/cache/memory, not the main deep-research engine |
| PDF/document reader | Not an enterprise document platform | Text/Markdown/HTML plus block/table-aware PDF page metadata before chunking | Credible v1 reader; future OCR/scanned-document/layout work is honest roadmap |
| Evaluation | Deep Research Bench / LLM judge style | Proxy metrics, optional LLM judge artifact, optional Ragas artifact | Demo artifact, not a full public benchmark |
| Deployment | Local graph/checkpoint examples | Single-node FastAPI/Celery/Redis/SQLite/Qdrant | Personal/local deployment; do not call it distributed |

Demo artifact warning: old `examples/llm-judge-report.json` and
`examples/ragas-report.json` may contain realistic weak spots such as shallow analysis,
mixed source quality, weak source candidates, or mediocre faithfulness/context precision.
That does not automatically mean the architecture is wrong; search APIs can return mixed
sources and small demos are sensitive to question choice. Before interviews, regenerate a
cleaner demo using questions that naturally retrieve papers, official docs, technical
reports, standards, or primary-source pages.

For MCP demos, prefer the local `arc-research-workbench` server instead of random
public MCP servers. It makes the tool path reliable and useful for actual
experiments: the researcher can call `search_grounding_corpus` for ingested
document evidence, `recall_project_memory` for session/summary/canonical memory,
`inspect_research_runs` for trace/evaluation replay, and `check_demo_readiness`
before an interview demo. Optional tools (`search_reference_corpus`,
`inspect_runtime_config`, and `recommend_demo_questions`) remain available for
architecture study and demo preparation. This demonstrates MCP as a tool
registration layer without pretending ODR ships a fixed server bundle.

## Next Optimizations

### P0: Prepare Real Demo Corpus, Memory, And Trace Artifacts

Status: required before important interviews.

Why: the runtime can be technically ready while the demonstration is still weak.
If the grounding corpus is empty and memory has no records, MCP readiness and
hybrid retrieval can load correctly but the system will not visibly prove its
research value.

Minimum interview pack:

- ingest 5-10 high-quality sources, preferably official docs, papers, technical
  reports, standards, or primary-source pages
- run 2-3 complex questions that naturally require planning, web search, local
  grounding, memory recall, and citation verification
- keep one run that calls the local MCP workbench
- save the final report, source index, trace, evaluation, and optional LLM judge
  or Ragas artifact
- record what the run demonstrates: planner output, `ConductResearch`, search
  provider evidence, source-reader compression, Qdrant/BM25/rerank metadata,
  memory writes, verifier/evaluator notes, and replay

This is the highest-value next step because it turns the architecture into a
credible product demonstration.

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
- indexing-time contextual retrieval prefixing
- Qdrant dense retrieval plus SQLite FTS5/BM25 keyword retrieval
- parent-child retrieval with parent/neighbor context expansion
- LightRAG-inspired entity/relation graph signal fused before reranking
- citation and evidence verification
- memory governance and conflict review
- source-quality evaluation boundary
- job queue/trace/replay boundary
- model/search provider configuration
- revision loop failure handling

### P1: Add A Reranker Interface

Status: implemented for v1.

Why: current retrieval already uses indexing-time contextual retrieval prefixes, Qdrant dense retrieval, SQLite FTS5/BM25 keyword retrieval, and a pluggable reranker
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

### P1: Provider Raw-Content Reading

Status: implemented for Tavily with extraction and optional model compression.

Why: deep research should not rely only on search snippets. Open Deep Research's
Tavily tool requests raw page content and compresses it before final synthesis.
This repo follows that boundary without adding a separate browser stack in v1:
Tavily can request `raw_content`, the source reader converts it into citation-ready
evidence, and report synthesis receives compact evidence instead of raw pages.

- `ARC_SEARCH_INCLUDE_RAW_CONTENT=true`
- `ARC_SOURCE_READER_ENABLED=true`
- `ARC_SOURCE_READER_STRATEGY=extract | model_compress | chunk_rerank_compress`
- `ARC_SOURCE_READER_MAX_CHARS=50000`
- `ARC_SOURCE_READER_EXCERPT_CHARS=1600`
- `ARC_RESEARCH_MAX_ITERATIONS=3`
- `extract` uses deterministic query-relevant sentence selection for tests/offline runs
- `model_compress` calls the configured chat model and returns a structured
  `summary/key_excerpts/relevance/limitations` contract
- `chunk_rerank_compress` follows the Open Deep Research legacy direction:
  split raw content into overlapping chunks, rerank candidate chunks by query
  relevance, expand selected chunks with a configurable neighbor window, stitch
  the expanded context in source order, then run structured model compression
- trace/evidence metadata records `read_strategy`, raw-content size, compression model
  usage, chunk counts, neighbor expansion, rerank method, compressed size, and
  excerpt size when available

This is still a provider reading layer, not a full browser/PDF reader. If the product
needs stronger document fidelity later, add a dedicated source-reader adapter for HTML
and PDFs behind the same contract instead of mixing parser logic into the researcher.

### P1: Local Document Reader And Chunking Boundary

Status: implemented for local text/Markdown/HTML files, heading-aware section
segmentation, and optional PyMuPDF-backed PDF block/table-aware page parsing.

Why: internal RAG quality depends heavily on parsing and chunking. A strong
retrieval stack cannot recover evidence that was lost during ingestion, so the
local grounding path keeps parsing, segmenting, chunking, indexing, retrieval,
and report synthesis as explicit stages.

- `POST /v1/documents/ingest` parses a local file path into grounding documents
- text, CSV/JSON/XML/YAML-like files are read as normalized text
- Markdown headings are used to create section segments before vector chunking
- HTML is converted to visible text with script/style noise removed; headings are
  preserved as Markdown-style section boundaries before segmentation
- Markdown/HTML section segments preserve `section_heading`, `section_level`,
  `section_path`, `section_path_parts`, `section_index`, and `section_count`
- PDFs use PyMuPDF when `.[documents]` is installed
- PDF pages become separate document segments with `page_number`, `page_count`,
  `segment_kind=page`, and parser metadata before vector chunking
- PDF parsing prefers PyMuPDF block extraction for reading-order recovery and
  falls back to plain page text when block extraction is unavailable
- PDF metadata includes `pdf_text_parse_method`, `text_block_count`,
  `line_count`, `heading_hints`, page width/height, and page rotation when
  available
- PyMuPDF `find_tables()` is used opportunistically; detected tables are
  converted into compact Markdown-like text and table metadata records
  `table_count`, `table_cell_count`, and `has_tables`
- `DocumentStore` still owns paragraph-aware child chunking, chunk overlap,
  indexing-time contextual retrieval prefixing, title/source/chunk metadata injection, Qdrant dense indexing,
  SQLite FTS5/BM25 keyword indexing, a lightweight entity/relation graph index, RRF/DBSF fusion,
  graph-score fusion, reranking, and same-document parent/neighbor context
  expansion after child retrieval
- long-paragraph sentence splitting uses explicit English and Chinese punctuation
  boundaries instead of a corrupted regex literal

This is now a credible v1 local knowledge reader, but it should still be described
honestly: it is not a full enterprise document intelligence pipeline. Future
hardening can add OCR for scanned PDFs, figure captions, stronger layout-aware
chunking, table-position citations, and stronger page/section citation rendering.

### P1: LightRAG-Inspired Graph-Augmented Retrieval

Status: implemented as a lightweight graph signal inside the existing grounding
layer.

Why: plain vector top-k retrieval can miss short proper nouns, component names,
and cross-chunk relationships. LightRAG is useful as a reference because it
pushes graph-enhanced indexing and retrieval instead of relying only on semantic
similarity. This repo keeps the implementation smaller and product-specific:
local document chunks are still the primary retrievable units, but ingestion also
extracts entity labels from each child chunk and records chunk/entity and
entity/entity co-occurrence edges.

Current behavior:

- extracts lightweight entities from chunk titles, sources, and raw chunk text
- avoids indexing contextual wrapper labels such as `Document`, `Metadata`, or
  `Excerpt` as graph entities
- keeps chunk -> entity, entity -> chunk, and entity -> neighbor entity indexes in
  memory alongside the Qdrant/local vector index
- expands a query through matched entities and neighboring relation entities
- fuses `graph_score` into the dense/BM25 candidate set before reranking
- preserves graph metadata such as `graph_query_entities`,
  `graph_matched_entities`, `graph_expanded_entities`, and
  `graph_augmented_retrieval` for trace and demo inspection

How to present it:

> The project does not claim to be a full LightRAG clone. It adopts the useful
> idea of graph-enhanced retrieval: dense vectors provide semantic recall, BM25
> keyword retrieval preserves exact terms, the lightweight entity/relation graph improves
> cross-chunk recall, and the reranker decides the final evidence order.

Future hardening can add LLM-based entity/relation extraction, persistent graph
storage, relation typing, graph traversal depth control, and offline retrieval
ablation experiments. Those are useful research directions, but not required for
the current single-node AI Research Copilot demo.

## Resume-Safe Boundary

Strong phrasing:

> Built a LangGraph-based AI Research Copilot that decomposes complex questions, routes
> between external search/reading, contextual retrieval, and memory, uses Qdrant
> dense+BM25 hybrid grounding, verifies citations, evaluates RAG/source quality,
> and persists traceable research reports.

Avoid:

- "production-grade distributed agent platform"
- "distributed LangGraph durable execution"
- "full Ragas benchmark" unless a larger labeled dataset is added
- "cross-encoder reranking" until that provider is actually wired
- "runtime source-quality policy" because v1 keeps source quality in evaluation
- "fully autonomous browser-scale research platform" because v1 uses ODR-style supervisor delegation but not browser automation or distributed crawling
- "browser-level source reader" because v1 uses provider raw content and compression
- "enterprise PDF/OCR/document-intelligence parser" because v1 preserves page/block/table metadata but does not solve scanned PDFs or complex layout intelligence
