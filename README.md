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

The technical layer uses **LangGraph + Agentic RAG**. It does not run plain
`question -> top-k -> answer` RAG: it plans sub-questions, rewrites queries,
selects tools, routes between external search and contextual retrieval, checks
evidence sufficiency, and triggers a revision loop when the answer is not
adequately supported.

It turns a user question into a research artifact by combining:

- supervisor-driven planning and decomposition
- concurrent research over plan items
- retrieval routing between external search, uploaded context, and hybrid evidence
- explicit tool selection for `web_search`, `vector_retrieval`, and `memory_recall`
- query rewrite / multi-query retrieval plans per research unit
- Qdrant-backed contextual document retrieval with dense/sparse fusion and reranking
- Open Deep Research-style final synthesis: compressed findings are rewritten by the report model while citations remain mapped to existing evidence
- short-term and long-term memory
- agent handoff and verification
- RAG/citation evaluation, observability, cost tracking, and replay

## Why this project

- It is a strong way to learn agentic research, grounding, memory, and tool calling.
- It is easier to explain in interviews than a generic chat bot.
- It can be packaged honestly as a derivative/assembly project with clear attribution.
- It has a stable demo path: complex question + project context + external search -> plan -> retrieve -> verify -> report -> memory -> trace.

## Product Flow

1. `Supervisor` starts the run and records trace/checkpoint state.
2. `Memory` recalls session, canonical, and summary records.
3. `Planner` decomposes the research question into plan items.
4. `Router` chooses external search, internal retrieval, or hybrid routes.
5. `Router` emits selected tools, query rewrites, and evidence sufficiency thresholds.
6. `Researcher` runs plan-item research concurrently.
7. `Retriever` uses Qdrant dense/sparse hybrid retrieval, RRF/DBSF fusion, and reranking for uploaded context.
8. `Reporter` generates source-indexed report sections.
9. `Verifier` checks citations, coverage, confidence, and weak claims.
10. `Evaluator` records RAG/citation quality metrics.
11. `Supervisor` either accepts the answer, writes memory, or triggers a revision loop.

## Current Stack And Upgrade Path

- Backend: Python 3.11+, FastAPI, LangGraph-backed research workflow orchestration
- Jobs: single-worker background queue for offline tests, plus Celery/Redis for strict single-node worker separation; queued/running/completed/failed/cancelled states, retry metadata, and cancellation records
- Search: offline-safe local mode by default, with Open Deep Research-style providers (`tavily`, `exa`, `perplexity`, `arxiv`, `pubmed`, `linkup`, `openai_web`, `anthropic_web`) plus practical adapters (`duckduckgo`, `brave`, `serpapi`) behind the same tool contract; strict demo mode requires a configured real provider
- Retrieval: Qdrant-backed dense/sparse named vectors + RRF/DBSF fusion + Qwen/DashScope reranking, with local fallbacks reserved for tests/offline mode and disabled by `ARC_STRICT_PROVIDERS=true`
- Memory: layered session, canonical fact, and summary records persisted in SQLite, with embedding-assisted recall and conflict governance
- Model runtime: OpenAI-compatible chat/embedding adapter with deterministic test doubles
- UI: dependency-light research workspace served by FastAPI, with provider readiness, job progress, report review, route inspection, and trace timeline in one local console
- Infra: Docker Compose for local service orchestration
- Evaluation: proxy RAG gates for plan coverage, retrieval hit rate, source quality, citation precision, source coverage, and unsupported sections, plus optional Open Deep Research-style judge and Ragas artifacts

Source quality intentionally stays in the evaluator layer for v1. The inspected
Open Deep Research reference scores source quality through evaluation rather than
hard-filtering search results at runtime. This repo follows that boundary: weak
sources are surfaced through metrics, trace, and revision notes, while the search
loop remains able to preserve niche or fresh sources when they are relevant.

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
- `ARC_RESEARCH_MAX_WORKERS=4`
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
Tavily, embedding/rerank to Qwen/DashScope, and Qdrant to `http://localhost:6333`
when those variables are not already set. It does not contain secrets. The app
will refuse to start if any required real provider is missing.

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
- Sources: add project documents and grounding context.
- Memory: add and filter layered memory records.
- Traces: inspect telemetry events, run traces, and checkpoints.
- Config: inspect runtime agents, tools, routing, providers, and quality gates.

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
- The pipeline now runs through a LangGraph `StateGraph` with supervisor, memory, planner, parallel research, reporter, verifier/evaluator, memory-write, and finalization nodes.
- It includes supervisor-driven handoffs, explicit Agentic RAG routes, query rewrites, tool selection, evidence sufficiency checks, concurrent plan-item research, Qdrant dense/sparse hybrid retrieval, Qwen/DashScope reranking, strict real-provider demo mode, layered memory, semantic memory recall, memory governance, structured contracts, single-node LangGraph SQLite checkpoint support, SQLite replay traces, RAG/citation/source evaluation, and report generation.
- Demo artifacts have been captured with an OpenAI-compatible chat provider, Qwen embeddings, and Tavily search.
- The local Open Deep Research reference does not implement a standalone runtime source-quality filter. This repo follows that shape: source quality is scored in evaluation and traces, while search provider ranking/filtering is left to the configured provider.
- Evaluation follows a lightweight Ragas-style direction with local proxy metrics for context precision, context recall, and faithfulness. `scripts/run_ragas_eval.py` can produce an actual optional Ragas artifact when `.[eval]` is installed, but runtime scoring remains lightweight.

## Resume-safe phrasing

Use wording like:

> Built an agentic deep research workspace by assembling and extending MIT-licensed open-source agent and research frameworks.

Or, more specifically:

> Built a LangGraph-based agentic research copilot that decomposes complex questions, routes between external search and contextual RAG, verifies citations, evaluates RAG quality, and persists traceable research reports.

Interview explanation:

> I built Agentic RAG rather than plain top-k RAG: the planner decomposes the question, the router selects tools and rewrites queries for each research unit, the retriever/searcher collect evidence, the verifier/evaluator check citation coverage and evidence sufficiency, and the LangGraph supervisor triggers revision when support is weak.

Do not present upstream code as your original work.

## Source Map

- `open_deep_research`: LangGraph research workflow structure, supervisor/researcher split, search-provider abstraction, and citation-backed reports
- `praisonaiagents`: memory, persistence, observability, session/replay patterns
- this repo: product shape, API, storage model, run ledger, checkpointing, deployment glue

The repo uses PraisonAI and Open Deep Research as reference designs, not as runtime
dependencies. That keeps the product narrow: a deep-research copilot instead of a
generic agent framework.

Memory design note: PraisonAI has a broader memory runtime with short/long/entity/user
memory, quality-aware search, session persistence, and knowledge retrieval. This repo
does not copy that whole platform. It implements the parts that fit the product:
session/summary/canonical memory, embedding-assisted recall, confidence-aware ranking,
canonical conflict review, and run/session provenance.

## Study Route

1. Start with `docs/source-map.md`.
2. Then read `workflow.py`, `pipeline.py`, and `storage.py`.
3. Read `docs/hardening-roadmap.md` for the current resume-safe boundaries and next hardening priorities.
4. Finally inspect the API and the run output to understand how the pieces fit together.
