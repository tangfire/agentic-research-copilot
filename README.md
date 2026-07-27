# AI Research Copilot

Deep research assistant built from MIT-licensed open-source references and original product-specific glue code.

The product is not a chatbot or a generic agent framework. It turns complex
questions into citation-backed research reports by planning, searching, recalling
context, verifying evidence, evaluating RAG quality, and preserving inspectable
run traces.

The product positioning is **AI Research Copilot for complex questions**:
users ask a complex question, and the system plans a research path, searches
external sources, recalls context, verifies evidence, and produces a cited,
traceable, reviewable research report.

One-line positioning:

> An AI Research Copilot for complex questions that uses planning, search,
> reading, retrieval augmentation, memory recall, citation verification, and
> evaluation replay to generate traceable research reports.

The main product path is **planning -> search/reading -> synthesis ->
verification/evaluation -> replay**. RAG is not the only or primary path; it is
used for contextual grounding, cached evidence from already-read material, and
memory recall.

The technical layer uses **LangGraph + Agentic RAG**. It does not run plain
`question -> top-k -> answer` RAG: it plans sub-questions, rewrites queries,
uses an ODR-style research supervisor to delegate tool-backed research units,
checks evidence sufficiency, and triggers a revision loop when the answer is not
adequately supported.

It turns a user question into a research artifact by combining:

- supervisor-driven planning, reflection, and research delegation
- concurrent research over plan items
- ODR-style `think_tool`, `ConductResearch`, and `ResearchComplete` decisions
- a bounded researcher loop per delegated unit: choose `think_tool`, `web_search`, configured `mcp_tool`, or `ResearchComplete`, then compress findings
- explicit tool selection for `web_search`, `vector_retrieval`, `memory_recall`, and configured MCP tools
- query rewrite / multi-query retrieval plans per research unit
- Open Deep Research-style web reading: Tavily can return `raw_content`, and the source reader turns it into citation-ready evidence through query-aware extraction, model compression, or chunk-rerank compression with neighbor expansion
- local document ingestion for text/Markdown/HTML and optional block/table-aware PDF page parsing before Qdrant-backed contextual retrieval
- LightRAG-inspired graph-augmented retrieval: entity/relation signals are fused with parent-child dense/BM25 retrieval before reranking
- Open Deep Research-style final synthesis: compressed findings are rewritten by the report model while citations remain mapped to existing evidence
- short-term and long-term memory
- agent handoff and verification
- RAG/citation evaluation, observability, cost tracking, and replay

## Why this project

- It is a strong way to learn agentic research, grounding, memory, and tool calling.
- It deliberately uses Open Deep Research as the primary reference design, so the study path is clear: understand ODR's supervisor/researcher/report/eval loop, then inspect how this repo adapts that shape into a local product.
- It is easier to explain in interviews than a generic chat bot.
- It can be packaged honestly as a derivative/assembly project with clear attribution.
- It has a stable demo path: complex question + project context + external search -> plan -> retrieve -> verify -> report -> memory -> trace.

## Product Flow

1. `Supervisor` starts the run and records trace/checkpoint state.
2. `Memory` recalls session, canonical, and summary records.
3. `Planner` decomposes the research question into plan items.
4. `ResearchSupervisor` reflects with `think_tool`, delegates plan items through `ConductResearch`, and records `ResearchComplete` criteria.
5. Each `ConductResearch` call carries selected tools, query rewrites, evidence thresholds, and external/internal/hybrid grounding mode.
6. `Researcher` runs delegated research units concurrently. Each unit uses a bounded ODR-style tool loop: the model chooses between reflection, web search, configured MCP tools, and completion; provider raw content is read/compressed when available; evidence/source sufficiency and stopping reason are recorded in the trace.
7. `Retriever` uses child-chunk retrieval, parent/neighbor context expansion, Qdrant dense retrieval, SQLite FTS5/BM25 keyword retrieval, RRF/DBSF fusion, and reranking for uploaded context and already-read grounding material.
8. `Reporter` generates source-indexed report sections.
9. `Verifier` checks citations, coverage, confidence, and weak claims.
10. `Evaluator` records RAG/citation quality metrics.
11. `Supervisor` either accepts the answer, writes memory, or triggers a revision loop.

## Current Stack And Upgrade Path

- Backend: Python 3.11+, FastAPI, LangGraph-backed research workflow orchestration
- Jobs: single-worker background queue for offline tests, plus Celery/Redis for strict single-node worker separation; queued/running/completed/failed/cancelled states, retry metadata, and cancellation records
- Search/reading: offline-safe local mode by default, with Open Deep Research-style providers (`tavily`, `exa`, `perplexity`, `arxiv`, `pubmed`, `linkup`, `openai_web`, `anthropic_web`) plus practical adapters (`duckduckgo`, `brave`, `serpapi`) behind the same tool contract; Tavily can request raw page content and the source reader compresses it into source-backed evidence excerpts; strict demo mode requires a configured real provider
- Document ingestion: local file reader for text/Markdown/HTML and optional PyMuPDF-backed PDF parsing; Markdown/HTML headings are preserved as section segments, and PDF pages are stored as separate block/table-aware segments with `page_number`/`page_count`, layout, heading-hint, and table metadata before vector chunking
- Retrieval: paragraph-aware child chunking + Anthropic-style indexing-time contextual retrieval prefixes + parent-child context expansion + LightRAG-inspired entity/relation graph signal + Qdrant-backed dense vectors + SQLite FTS5/BM25 keyword index + RRF/DBSF fusion + Qwen/DashScope reranking, with local fallbacks reserved for tests/offline mode and disabled by `ARC_STRICT_PROVIDERS=true`
- Memory: layered session, canonical fact, and summary records persisted in SQLite, with embedding-assisted recall and conflict governance
- Model runtime: OpenAI-compatible chat/embedding adapter with deterministic test doubles
- UI: dependency-light research workspace served by FastAPI, with provider readiness, job progress, report review, route inspection, and trace timeline in one local console
- Infra: Docker Compose for local service orchestration
- Evaluation: proxy RAG gates for plan coverage, retrieval hit rate, source quality, citation precision, source coverage, and unsupported sections, plus optional Open Deep Research-style judge and Ragas artifacts

## Technical Design Fit

The stack is intentionally matched to the product problem. It is not a generic
chatbot wrapped in fashionable agent/RAG terminology.

- `LangGraph` fits because the core workflow is stateful and conditional:
  clarify, plan, delegate research, collect evidence, synthesize, verify,
  revise, write memory, and persist trace/replay artifacts. A linear chain would
  hide those states.
- `FastAPI` fits because the product needs a small, inspectable local API for
  jobs, runs, documents, memory, telemetry, and runtime readiness.
- `Celery` + `Redis` fits only as single-node API/worker separation for strict
  real-provider demos. The repo does not claim distributed scheduling,
  multi-tenant isolation, or SaaS operations.
- `SQLite` fits for local run ledgers, memory, jobs, telemetry, and LangGraph
  checkpoint files. It keeps the demo reproducible and inspectable without
  pretending to be a distributed database.
- `Qdrant` fits as the dense-vector grounding backend, while SQLite FTS5/BM25
  keeps exact term recall visible. The hybrid path is useful for research
  questions where component names, dates, paper titles, and metrics matter.
- The reranker fits because dense/BM25/graph fusion creates a candidate set, but
  final evidence order still needs query-aware relevance scoring.
- MCP fits as a configurable tool boundary. The local workbench server exposes
  useful project capabilities to the researcher: grounding search, memory recall,
  run/evaluation inspection, and readiness checks. It is not positioned as an
  enterprise MCP gateway.
- Provider `raw_content` reading fits the v1 deep-research boundary: it reads
  and compresses source content into citation-ready evidence without taking on a
  full browser automation stack.

The product should be presented as a single-node AI Research Copilot. It is
credible as an agentic research and Agentic RAG learning project, but it should
not be described as a production distributed research platform.

## Open Deep Research Alignment

This repo is designed as an Open Deep Research learning and adaptation project,
not a generic agent SDK. The main shape intentionally follows the inspected ODR
reference:

- LangGraph-style supervisor/researcher/report graph
- ODR-style clarification gate before research starts
- complex question decomposition and concurrent research units
- ODR's bias toward single-agent simplicity unless the question has clear independent research directions
- bounded researcher search/read/reflect loops before evidence compression
- provider `raw_content` reading and compression before final synthesis
- MCP compatibility through a configurable `url` + `tools` allowlist, matching ODR's `mcp_config` shape instead of hard-coding server names
- citation-backed report generation with source indexes
- source quality, groundedness, completeness, and structure evaluated in demo/eval artifacts instead of a runtime source-blocking policy

There are deliberate product-specific differences:

- The runtime mirrors ODR's supervisor boundary: an LLM-backed
  `ResearchSupervisor` emits `think_tool`, `ConductResearch`, and
  `ResearchComplete` decisions. Each `ConductResearch` call carries evidence
  tools, query rewrites, grounding mode, and sufficiency criteria. The
  deterministic provider only fills the same schema for offline tests/fallbacks.
- Multi-agent execution is conditional, not cosmetic. The supervisor can delegate
  several independent research units when the plan has parallelizable subtopics,
  but the implementation keeps a single-researcher path for simple questions. This
  follows ODR's "bias toward single agent unless parallelization is clear" rule.
- Each delegated external research unit now records a bounded researcher loop:
  query, new evidence count, source count, sufficiency gaps, reflection, next
  query, and completion reason. This makes the multi-agent claim inspectable in
  run notes, checkpoints, and trace replay.
- External web reading follows ODR's practical v1 boundary: Tavily/provider
  `raw_content` is compressed into evidence. It is not a full browser automation
  reader.
- Local document reading is an extension for this product's grounding layer:
  text/Markdown/HTML/PDF parsing feeds Qdrant retrieval. PDF support preserves
  page, block, layout, heading-hint, and table metadata when PyMuPDF can extract
  it, but it is not an enterprise OCR or scanned-document intelligence system.
- Evaluation produces runtime proxy metrics, optional Ragas artifacts, and
  optional LLM-as-judge artifacts. It is not a large public benchmark unless a
  larger labeled dataset is added.
- Celery/Redis/SQLite/Qdrant are used as a single-node personal deployment
  shape. Do not describe this as a distributed research platform.

Source quality intentionally stays in the evaluator layer for v1. The inspected
Open Deep Research reference scores source quality through evaluation rather than
hard-filtering search results at runtime. This repo follows that boundary: weak
sources are surfaced through metrics, trace, and revision notes, while the search
loop remains able to preserve niche or fresh sources when they are relevant.

Source reading is deliberately scoped. For external web evidence, the v1 default
is a provider reading layer: search providers can return raw content, and the
source reader converts it into compact evidence through rule-based query extraction
(`extract`), structured model compression (`model_compress`), or Open Deep
Research legacy-style chunk reranking plus compression (`chunk_rerank_compress`).
For long provider pages, `chunk_rerank_compress` retrieves precise child chunks,
expands each hit with a configurable neighbor window, stitches the expanded
context in source order, and only then asks the model to compress the evidence.
This handles the common boundary failure where the policy premise lands in one
chunk while the concrete number or conclusion lands in the next.
That puts the web-reading path at the same practical v1 boundary as the inspected
Open Deep Research reference: search discovers sources, provider `raw_content`
acts as the reading layer, and compressed evidence feeds report synthesis. It is
not advertised as a full browser automation stack.

Local grounding has a separate reader boundary. `POST /v1/documents/ingest` can
read local text, Markdown, HTML, and PDF files. Plain text-like files become one
document segment. Markdown and HTML keep heading structure by emitting section
segments with `section_heading`, `section_level`, `section_path`,
`section_path_parts`, `section_index`, and `section_count` metadata. PDFs are
parsed with optional PyMuPDF and split into page segments so `page_number`,
`page_count`, `text_block_count`, `heading_hints`, `table_count`,
`table_cell_count`, page dimensions, and rotation survive into retrieval
metadata. When PyMuPDF table detection succeeds, detected tables are converted
into compact Markdown-like text and appended to the page segment before
chunking. After parsing, `DocumentStore`
performs paragraph-aware child chunking with overlap, generates an indexing-time
contextual retrieval prefix for each child chunk, embeds the prefixed chunk,
writes dense vectors into Qdrant, writes the same prefixed chunk into a SQLite
FTS5 keyword index, fuses dense and BM25 results with
RRF/DBSF, and reranks the candidates. The returned evidence uses a parent-child
pattern: precise child chunks are retrieved, then the same document/section/page
parent context plus neighboring child chunks are returned for synthesis. The
store also maintains a LightRAG-inspired lightweight entity co-occurrence graph:
query entities retrieve directly matched chunks and nearby relation-neighbor
chunks, then the graph score is fused with dense/BM25 candidates before
reranking. Parsing, chunking, graph indexing, retrieval, and report synthesis
stay separated so the reader can be strengthened later without rewriting the
research graph.

Install the document parsing extra when PDF ingestion is needed:

```bash
pip install -e .[documents]
```

## Local Config

- `ARC_STORAGE_PATH=.arc/agentic_research.db`
- `ARC_LOAD_DOTENV=true`
- `ARC_ORCHESTRATION_RUNTIME=langgraph`
- `ARC_STRICT_PROVIDERS=false`
- `ARC_SEARCH_PROVIDER=none`
- `ARC_SEARCH_API_KEY=`
- `ARC_SEARCH_BASE_URL=`
- `ARC_SEARCH_MODEL=`
- `ARC_SEARCH_DEPTH=basic`
- `ARC_SEARCH_MAX_RESULTS=5`
- `ARC_SEARCH_TIMEOUT_SECONDS=8`
- `ARC_SEARCH_INCLUDE_RAW_CONTENT=true`
- `ARC_MCP_ENABLED=true`
- `ARC_MCP_SERVER_URL=http://127.0.0.1:8765`
- `ARC_MCP_TOOLS=search_grounding_corpus,recall_project_memory,inspect_research_runs,check_demo_readiness`
- `ARC_MCP_AUTH_REQUIRED=false`
- `ARC_MCP_AUTH_TOKEN=`
- `ARC_MCP_PROMPT=Use MCP workspace tools when ingested grounding documents, project memory, prior run traces/evaluation, or demo readiness checks can improve the research answer.`
- `ARC_MCP_TRANSPORT=streamable_http`
- `ARC_MCP_TIMEOUT_SECONDS=20`
- `ARC_MCP_DEMO_PORT=8765`
- `ARC_MCP_DEMO_API_BASE=http://127.0.0.1:8010`
- `ARC_MCP_DEMO_ROOTS=`
- `ARC_SOURCE_READER_ENABLED=true`
- `ARC_SOURCE_READER_STRATEGY=extract`
- `ARC_SOURCE_READER_MAX_CHARS=50000`
- `ARC_SOURCE_READER_EXCERPT_CHARS=1600`
- `ARC_RESEARCH_MAX_WORKERS=4`
- `ARC_RESEARCH_MAX_ITERATIONS=3`
- `ARC_JOB_MAX_ATTEMPTS=2`
- `ARC_JOB_TIMEOUT_SECONDS=120`
- `ARC_JOB_QUEUE_BACKEND=in_process`
- `ARC_CELERY_BROKER_URL=redis://localhost:6379/0`
- `ARC_CELERY_RESULT_BACKEND=redis://localhost:6379/1`
- `ARC_MODEL_PROVIDER=deterministic`
- `ARC_MODEL_BASE_URL=`
- `ARC_MODEL_CHAT_MODEL=gpt-4o-mini`
- `ARC_MODEL_EMBEDDING_MODEL=text-embedding-3-small`
- `ARC_EMBEDDING_PROVIDER=model`
- `ARC_EMBEDDING_BASE_URL=`
- `ARC_EMBEDDING_API_KEY=`
- `ARC_EMBEDDING_MODEL=qwen3.7-text-embedding`
- `ARC_EMBEDDING_DIMENSIONS=256`
- `ARC_QDRANT_URL=http://localhost:6333`
- `ARC_QDRANT_COLLECTION=arc_documents`
- `ARC_QDRANT_LOCATION=:memory:`
- `ARC_QDRANT_PREFER_LOCAL=true`
- `ARC_MAX_REVISIONS=2`
- `ARC_RAG_MAX_QUERY_REWRITES=2`
- `ARC_RAG_MIN_EVIDENCE_PER_ITEM=2`
- `ARC_RAG_MIN_SOURCE_DIVERSITY=2`
- `ARC_RAG_HYBRID_FUSION=rrf`
- `ARC_RAG_GRAPH_ENABLED=true`
- `ARC_RAG_GRAPH_MAX_ENTITIES_PER_CHUNK=12`
- `ARC_RAG_GRAPH_NEIGHBOR_LIMIT=8`
- `ARC_RERANK_PROVIDER=dashscope`
- `ARC_RERANK_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `ARC_RERANK_API_KEY=`
- `ARC_RERANK_MODEL=qwen3-rerank`
- `ARC_RERANK_TIMEOUT_SECONDS=15`
- `ARC_RERANK_CANDIDATE_LIMIT=24`
- `ARC_LANGGRAPH_CHECKPOINTER=sqlite`
- `ARC_LANGGRAPH_CHECKPOINT_PATH=.arc/langgraph_checkpoints.sqlite`

Set `ARC_SEARCH_PROVIDER=duckduckgo`, `arxiv`, or `pubmed` for no-key search sources.
Set `ARC_SEARCH_PROVIDER=tavily`, `exa`, `perplexity`, `linkup`, `openai_web`,
`anthropic_web`, `brave`, or `serpapi` with `ARC_SEARCH_API_KEY` when you want a
production-grade search provider. Use `ARC_SEARCH_MODEL` for answer-engine providers
such as Perplexity, OpenAI native web search, or Anthropic native web search.
Keep `ARC_SEARCH_DEPTH=basic` for Tavily demos to minimize credit usage; switch to
`advanced` only when you intentionally want a deeper search.
Keep `ARC_SEARCH_INCLUDE_RAW_CONTENT=true` for deep-research demos. This mirrors
Open Deep Research's Tavily path: search discovers candidate sources, provider
raw content acts as the reading layer, and the researcher compresses relevant
excerpts before final synthesis. Disable it only when cost or provider payload
size matters more than source-level evidence quality.

MCP is enabled as a capability by default, following Open Deep Research's
configuration boundary. ODR does not hard-code a fixed set of MCP servers; it
loads tools from a configured `mcp_config.url` and `mcp_config.tools` allowlist.
This repo mirrors that shape with `ARC_MCP_SERVER_URL` and comma-separated
`ARC_MCP_TOOLS`.

For local demos, the repo ships a streamable HTTP research-workbench MCP server:
`python -m agentic_research_copilot.research_mcp_server`. `start_real.ps1`
starts it automatically unless `-NoMcp` is passed. The default workbench tools are:

- `search_grounding_corpus`: queries `/v1/documents/search` so MCP can use the same contextual retrieval stack as the app: Qdrant dense retrieval, SQLite FTS5/BM25, parent-child expansion, graph signal, and rerank metadata.
- `recall_project_memory`: queries `/v1/memory/search` and returns session/summary/canonical memories with governance signals.
- `inspect_research_runs`: reads recent runs plus trace/evaluation summaries so a researcher can reuse prior findings and inspect replay artifacts.
- `check_demo_readiness`: checks provider readiness, MCP loading, local documents, memory, and completed runs before an interview/demo.

Optional inspection tools are also registered but are not part of the default
allowlist: `search_reference_corpus` for local ODR/PraisonAI/source-map lookup,
`inspect_runtime_config` for full runtime configuration inspection, and
`recommend_demo_questions` for a reproducible demo playbook.

Use `ARC_MCP_AUTH_REQUIRED=true` plus `ARC_MCP_AUTH_TOKEN` for bearer-token MCP
servers, and `ARC_MCP_PROMPT` to tell the researcher when those tools are useful.

Use `ARC_JOB_QUEUE_BACKEND=celery` when you want the API and worker separated on
one local node. In that mode, use `ARC_QDRANT_URL=http://localhost:6333` and
`ARC_QDRANT_PREFER_LOCAL=false`; embedded Qdrant paths are single-process only.

For chat-only OpenAI-compatible providers such as DeepSeek, set
`ARC_MODEL_PROVIDER=openai_compatible`, `ARC_MODEL_BASE_URL=https://api.deepseek.com`,
`ARC_MODEL_CHAT_MODEL=deepseek-chat`, and `ARC_EMBEDDING_PROVIDER=deterministic`.
For a separate OpenAI-compatible embedding provider, set `ARC_EMBEDDING_PROVIDER=openai_compatible`,
`ARC_EMBEDDING_BASE_URL`, `ARC_EMBEDDING_API_KEY`, and `ARC_EMBEDDING_MODEL`.
For Qwen AI Platform or DashScope, use `ARC_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`,
`ARC_EMBEDDING_MODEL=qwen3.7-text-embedding`, and `ARC_EMBEDDING_DIMENSIONS=256`.
For strict real-provider demos, do not use deterministic embeddings; use a separate
embedding provider such as Qwen/DashScope.

Real reranking defaults to Qwen/DashScope through `ARC_RERANK_PROVIDER=dashscope`.
Set `ARC_RERANK_API_KEY` from your shell or secret manager, not in committed files.
If `ARC_RERANK_API_KEY` is not set, the app also accepts `DASHSCOPE_API_KEY` and
falls back to deterministic local reranking for offline runs and tests.
Use `ARC_RERANK_MODEL=qwen3-rerank`; point `ARC_RERANK_BASE_URL` either at the
full `/compatible-api/v1/reranks` endpoint or the service base URL documented by
DashScope. The adapter also accepts `https://dashscope.aliyuncs.com/compatible-mode/v1`
and maps it to DashScope's rerank service endpoint internally, because the generic
compatible-mode URL does not expose `/reranks` directly.

The app loads a project-root `.env` by default. `.env` is ignored by git, so real
keys should live there or in your shell environment. Common aliases are accepted:
`DASHSCOPE_API_KEY` / `QWEN_API_KEY` for Qwen-compatible model, embedding, and rerank
adapters, `OPENAI_API_KEY` for OpenAI-compatible providers, `DEEPSEEK_API_KEY` for
DeepSeek chat, and provider-specific search aliases such as `TAVILY_API_KEY`,
`EXA_API_KEY`, `PERPLEXITY_API_KEY`, `BRAVE_API_KEY`, and `SERPAPI_API_KEY`.

## Real Provider Demo Mode

Use strict mode when you want a local demo that calls the real remote services
instead of silent test fallbacks:

```powershell
Copy-Item .env.real.example .env
# Fill the blank key/model values in .env or set the same values in your shell.
python scripts/check_real_providers.py
.\scripts\start_real.ps1 -Port 8000
```

Example local profile using an OpenAI-compatible relay for chat, DashScope/Qwen
for embeddings/rerank, Tavily for search, and local Docker Qdrant:

```env
ARC_STRICT_PROVIDERS=true
ARC_MODEL_PROVIDER=openai_compatible
ARC_MODEL_BASE_URL=https://relay.novelcat.cloud/v1
ARC_MODEL_CHAT_MODEL=<your-chat-model>
ARC_MODEL_API_KEY=<set-in-shell-or-secret-manager>
ARC_EMBEDDING_PROVIDER=openai_compatible
ARC_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ARC_EMBEDDING_MODEL=qwen3.7-text-embedding
ARC_EMBEDDING_API_KEY=<set-in-shell-or-secret-manager>
ARC_SEARCH_PROVIDER=tavily
ARC_SEARCH_API_KEY=<set-in-shell-or-secret-manager>
ARC_SEARCH_INCLUDE_RAW_CONTENT=true
ARC_MCP_ENABLED=true
ARC_MCP_SERVER_URL=http://127.0.0.1:8765
ARC_MCP_TOOLS=search_grounding_corpus,recall_project_memory,inspect_research_runs,check_demo_readiness
ARC_MCP_AUTH_REQUIRED=false
ARC_MCP_AUTH_TOKEN=<set-in-shell-or-secret-manager-if-required>
ARC_MCP_PROMPT=Use MCP workspace tools when ingested grounding documents, project memory, prior run traces/evaluation, or demo readiness checks can improve the research answer.
ARC_MCP_DEMO_PORT=8765
ARC_MCP_DEMO_API_BASE=http://127.0.0.1:8000
ARC_SOURCE_READER_ENABLED=true
ARC_SOURCE_READER_STRATEGY=chunk_rerank_compress
ARC_SOURCE_READER_MAX_CHARS=50000
ARC_SOURCE_READER_EXCERPT_CHARS=1600
ARC_SOURCE_READER_CHUNK_CONTEXT_WINDOW=1
ARC_RESEARCH_MAX_ITERATIONS=3
ARC_RERANK_PROVIDER=dashscope
ARC_RERANK_MODEL=qwen3-rerank
ARC_RERANK_API_KEY=<set-in-shell-or-secret-manager>
ARC_QDRANT_URL=http://localhost:6333
```

If Docker/Qdrant is not available locally, use a persistent embedded Qdrant path
instead of the in-memory fallback:

```env
ARC_QDRANT_URL=
ARC_QDRANT_LOCATION=.arc/qdrant-real
ARC_QDRANT_PREFER_LOCAL=true
```

`scripts/start_real.ps1` sets `ARC_STRICT_PROVIDERS=true`, defaults search to
Tavily, embedding/rerank to Qwen/DashScope, MCP to the local workbench MCP server,
and Qdrant to `http://localhost:6333` when those variables are not already set.
It does not contain secrets. The app will refuse to start if any required real
provider is missing.

Use `.\scripts\start_real.ps1 -Smoke` only when you intentionally want to make a
small remote embedding, search, and rerank call before starting the server. This
is useful for demo verification, but it consumes provider quota.

Inspect `GET /v1/runtime/provider-check` or the Config tab to confirm which
providers are active. The response only reports whether keys/base URLs are
configured; it never returns secret values.

## Run Locally

```bash
pip install -e .[dev]
uvicorn agentic_research_copilot.server:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the research workspace or `http://127.0.0.1:8000/docs`
for the API contract.

For strict real-provider startup on Windows, prefer the bundled script:

```powershell
.\scripts\start_real.ps1 -Port 8010 -Smoke
```

The script validates real providers, starts Redis/Qdrant when needed, starts a
Celery worker, and then serves the FastAPI workspace. Use `-InProcess` only when
you intentionally want the simpler single-process queue.

For manual single-node Celery/Redis queueing, install the queue extra and run a
worker beside the API:

```bash
pip install -e .[queue]
set ARC_JOB_QUEUE_BACKEND=celery
set ARC_QDRANT_URL=http://localhost:6333
set ARC_QDRANT_PREFER_LOCAL=false
python -m celery -A agentic_research_copilot.celery_app worker --loglevel=INFO --pool=solo
```

The code default `ARC_JOB_QUEUE_BACKEND=in_process` remains simpler for offline tests.
The strict local demo config uses Celery/Redis.

Use `ARC_ORCHESTRATION_RUNTIME=custom` only when you intentionally want to compare
the older custom workflow fallback against the LangGraph runtime.

Install `.[ai]` only if you want to experiment with the LangChain/OpenAI ecosystem
extras beyond the current OpenAI-compatible provider adapter.

The root web workspace includes:

- Portal: ask a research question and inspect report output.
- Runs: inspect plan steps, handoffs, and job history.
- Sources: add raw document snippets, ingest local files, delete, or clear project documents and grounding context.
- Memory: add and filter layered memory records.
- Traces: inspect telemetry events, run traces, and checkpoints.
- Config: inspect runtime agents, tools, routing, providers, and quality gates.

Maintenance endpoints:

- `GET /v1/documents/search?q=...` searches the ingested grounding corpus through the same hybrid retrieval stack used by MCP.
- `POST /v1/documents/ingest` parses a local text/Markdown/HTML/PDF file into grounding documents.
- `DELETE /v1/documents/{document_id}` removes one document and its vector chunks.
- `DELETE /v1/documents` clears the local document corpus and rebuilds the retrieval index as empty.
- `GET /v1/memory/search?q=...` recalls layered memory with governance metadata for MCP and manual inspection.
- `DELETE /v1/research/history` clears run/job/telemetry history while preserving documents and memory.
- `DELETE /v1/research/history?include_memory=true` also clears layered memory.

`ARC_SEED_REFERENCE_KNOWLEDGE=false` keeps first launch clean. Set it to `true`
only for offline demos or tests that intentionally need built-in project
grounding documents.

## Demo Artifacts

Use `scripts/capture_demo.py` to capture a resume/interview demo run without
writing secrets into the repository. The script reads provider configuration
from environment variables and writes:

- `examples/demo-report.md`
- `examples/demo-trace.json`

Use `scripts/run_llm_judge_eval.py` when you want an Open Deep Research-style
LLM-as-judge artifact for the saved demo report:

```bash
python scripts/run_llm_judge_eval.py --report examples/demo-report.md --context examples/demo-trace.json
```

This writes `examples/llm-judge-report.json` with research depth, source quality,
analytical rigor, structure, groundedness, and completeness scores. It is an
offline/demo evaluation step, not a default runtime cost.

Use `scripts/run_ragas_eval.py` when you want an optional Ragas artifact over the
same saved demo evidence:

```bash
pip install -e .[eval]
python scripts/run_ragas_eval.py --report examples/demo-report.md --trace examples/demo-trace.json
```

Or run the same evaluation in Docker without adding Ragas to the main app runtime:

```bash
docker compose --profile eval run --rm eval
```

This writes `examples/ragas-report.json`. It is kept outside the runtime path
because Ragas is a benchmark/evaluation dependency, not a product serving dependency.

The committed demo artifacts are examples, not permanent quality claims. Search
providers can return mixed sources, and older artifacts may show realistic weak
spots such as shallow analysis, mixed source quality, or mediocre
faithfulness/context precision. Before interviews, capture a fresh demo with a
question that naturally retrieves papers, official docs, technical reports, or
primary-source pages, then regenerate both the LLM judge and Ragas reports.

Recommended demo configuration:

- Chat: OpenAI-compatible relay such as DeepSeek/Qwen/OpenAI-compatible providers.
- Embeddings: Qwen AI Platform or DashScope `qwen3.7-text-embedding` with 256 dimensions.
- Search: `arxiv` for a no-key academic smoke, or Tavily/Exa/Perplexity/Brave/SerpAPI when a production search key is available.

Useful API paths:

- `POST /v1/research/jobs`
- `GET /v1/research/jobs`
- `GET /v1/research/jobs/{job_id}/status`
- `GET /v1/research/jobs/{job_id}/result`
- `POST /v1/research/jobs/{job_id}/cancel`
- `POST /v1/research/clarify`
- `POST /v1/research/runs`
- `GET /v1/research/runs/{run_id}/status`
- `GET /v1/research/runs/{run_id}/result`
- `GET /v1/research/runs/{run_id}/evaluation`
- `GET /v1/research/runs/{run_id}/trace`
- `GET /v1/memory/governance`
- `GET /v1/runtime/config`

## Planned Layout

```text
agentic-research-copilot/
  apps/
    api/
    web/
  docs/
  infra/
  src/
    agentic_research_copilot/
```

## Current Status

- Clean-room design and scaffold are in place.
- The API is runnable and exposes research jobs, status/result views, memory, documents, telemetry, config, and run history.
- A local web workspace is available at `/` for the portal and admin inspection flows.
- The pipeline now runs through a LangGraph `StateGraph` with supervisor, memory, planner, research_supervisor, parallel research, reporter, verifier/evaluator, memory-write, and finalization nodes.
- It includes ODR-style supervisor tool calls, supervisor-driven handoffs, Agentic RAG routes derived from `ConductResearch`, query rewrites, tool selection, evidence sufficiency checks, concurrent plan-item research, provider raw-content source reading, optional model compression, LightRAG-inspired entity/relation graph augmentation, parent-child retrieval, Qdrant dense retrieval, SQLite FTS5/BM25 keyword retrieval, Qwen/DashScope reranking, strict real-provider demo mode, layered memory, semantic memory recall, memory governance, structured contracts, single-node LangGraph SQLite checkpoint support, SQLite replay traces, RAG/citation/source evaluation, and report generation.
- Strict real-provider runs are supported with an OpenAI-compatible chat provider, Qwen/DashScope embeddings and reranking, Tavily search, Qdrant, Celery/Redis, LangGraph SQLite checkpointing, and the local MCP workbench.
- Demo artifacts should be regenerated before interviews from a populated grounding corpus and at least one completed memory-writing run. An empty corpus or empty memory store means the system is technically ready but the demonstration is not yet convincing.
- The local Open Deep Research reference does not implement a standalone runtime source-quality filter. This repo follows that shape: source quality is scored in evaluation and traces, while search provider ranking/filtering is left to the configured provider.
- Evaluation follows a lightweight Ragas-style direction with local proxy metrics for context precision, context recall, and faithfulness. `scripts/run_ragas_eval.py` can produce an actual optional Ragas artifact when `.[eval]` is installed, but runtime scoring remains lightweight.
- The main route selector is the ODR-style `ResearchSupervisor`; deterministic route hints remain only as fallback/test scaffolding and are not the product's primary decision layer.
- The web reader is provider-raw-content based, and the PDF reader is page/block/table-metadata based. Present them as credible v1 readers, not browser automation or enterprise OCR/document intelligence.

Current demo readiness checklist:

- Real providers configured: chat, embedding, search, rerank.
- Runtime configured: LangGraph SQLite checkpointing, Qdrant, Celery/Redis.
- MCP configured: streamable HTTP workbench with grounding, memory, run/eval, and readiness tools.
- Still required for a strong interview demo: ingest high-quality documents, run at least one successful research job, write/recall memory, and save trace/evaluation artifacts.

## Resume-safe phrasing

Use wording like:

> Built an agentic deep research workspace by assembling and extending MIT-licensed open-source agent and research frameworks.

Or, more specifically:

> Built a LangGraph-based agentic research copilot that decomposes complex questions, uses an ODR-style supervisor to delegate tool-backed research units, verifies citations, evaluates RAG quality, and persists traceable research reports.

Interview explanation:

> I built Agentic RAG rather than plain top-k RAG: the planner decomposes the question, the research supervisor reflects and emits `ConductResearch` calls with selected tools and query rewrites, the retriever/searcher collect evidence, the verifier/evaluator check citation coverage and evidence sufficiency, and the LangGraph supervisor triggers revision when support is weak.

Advanced RAG explanation:

> The grounding layer combines Contextual Retrieval, Hybrid Search, Rerank, Query Rewrite, Parent-Child retrieval, and a LightRAG-inspired graph signal. Each child chunk receives an indexing-time context prefix before it is written to Qdrant dense vectors and a real SQLite FTS5/BM25 keyword index; entity/relation graph hits are fused into the candidate set, a reranker orders the evidence, and the final evidence expands back to parent/neighbor context for report synthesis. This is graph-augmented RAG inside an ODR-style research workflow, not a standalone GraphRAG framework.

Do not present upstream code as your original work.

## Source Map

- `open_deep_research` (primary): LangGraph research workflow structure, supervisor/researcher split, search-provider abstraction, raw-content compression, citation-backed reports, and LLM/evaluator-style demo artifacts
- `praisonaiagents` (secondary): memory, reader registry, knowledge ingestion, persistence, observability, session/replay patterns
- this repo: product shape, API, storage model, run ledger, checkpointing, deployment glue

The repo uses Open Deep Research as the main learning target and reference
design, with PraisonAI only as a secondary source for memory/reader/replay ideas.
Neither reference is imported as a runtime dependency. That keeps the product
narrow: a deep-research copilot instead of a generic agent framework.

Memory design note: PraisonAI has a broader memory runtime with short/long/entity/user
memory, quality-aware search, session persistence, and knowledge retrieval. This repo
does not copy that whole platform. It implements the parts that fit the product:
session/summary/canonical memory, embedding-assisted recall, confidence-aware ranking,
canonical conflict review, and run/session provenance.

## Study Route

1. Start with `docs/source-map.md`.
2. Then read `source_reader.py`, `workflow.py`, `pipeline.py`, and `storage.py`.
3. Read `docs/hardening-roadmap.md` for the current resume-safe boundaries and next hardening priorities.
4. Read `docs/interview-notes.zh-CN.md` for Chinese interview talking points and design trade-offs.
5. Finally inspect the API and the run output to understand how the pieces fit together.
