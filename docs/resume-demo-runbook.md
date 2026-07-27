# Resume Demo Runbook

This runbook records the July 2026 resume demo experiment for AI Research
Copilot. It is intentionally written as an interview artifact: what was missing,
what failed under real providers, what was fixed, and what can now be shown.

## Goal

The project previously had a strong architecture but weak demo assets: the
strict-provider stack was configured, while the grounding corpus, memory, traces,
and evaluation artifacts were too empty to make the project feel real. The goal
of this experiment was to create stable assets that exercise agent planning,
RAG, memory, MCP, citation verification, evaluation, and replay.

## Runtime Profile

The completed run used:

- API: `http://127.0.0.1:8010`
- Orchestration: `langgraph`
- Queue: `celery`
- Vector backend: `qdrant_dense`
- Keyword backend: `sqlite_fts5_bm25`
- Reranker: `dashscope_qwen3_rerank`
- MCP workbench: loaded
- Strict provider readiness: passed

For an isolated demo run, use a separate storage file and Qdrant collection so
old experiments do not change startup time or retrieval results:

```powershell
$env:ARC_STORAGE_PATH = ".arc/resume_demo_run.db"
$env:ARC_QDRANT_COLLECTION = "arc_resume_demo_20260727"
$env:ARC_LANGGRAPH_CHECKPOINT_PATH = ".arc/langgraph_resume_demo.sqlite"
$env:ARC_RESEARCH_MAX_ITERATIONS = "2"
$env:ARC_RAG_MAX_QUERY_REWRITES = "1"
$env:ARC_JOB_TIMEOUT_SECONDS = "240"
.\scripts\start_real.ps1 -Port 8010 -NoInfra
```

Then prepare the corpus, memory, and run artifacts:

```powershell
python scripts\prepare_resume_demo_assets.py --source-dir "<local-reference-paper-folder>"
```

Use `--skip-runs` when you only want to seed corpus and memory.

## Demo Assets

The reproducible script is `scripts/prepare_resume_demo_assets.py`. Pass the
paper folder through `--source-dir` or `ARC_RESUME_DEMO_SOURCE_DIR`; committed
artifacts keep only stable `resume-demo-papers/<file>` source ids instead of a
machine-local absolute path. The script seeds controlled excerpts from five
federated learning papers:

- FedAvg: `mcmahan17a.pdf`
- pFedMe: personalized federated learning with Moreau envelopes
- FedRolex: model-heterogeneous FL with rolling sub-model extraction
- Personalized FL via heterogeneous model reassembly
- FedAUX: unlabeled auxiliary data for FL

It also seeds three memory records:

- `resume_demo:positioning`
- `resume_demo:corpus_scope`
- `resume_demo:demo_goal`

Final readiness:

- Corpus: 5 documents from 5 sources
- Memory: 3 records, including 1 canonical record
- MCP readiness: 10/10 checks passed
- Completed run artifacts: 2

Generated assets live under `examples/resume-demo/`:

- `demo-summary.json`
- `demo-summary.md`
- `fl-heterogeneity-comparison.*`
- `fl-personalization-design-memo.*`

Each completed run has a saved `run.json`, `trace.json`, `evaluation.json`, and
human-readable `report.md`.

## Completed Runs

| Slug | Status | Sources | Trace events | Checkpoints | Evaluation |
| --- | --- | ---: | ---: | ---: | --- |
| `fl-heterogeneity-comparison` | completed | 6 | 44 | 12 | passed |
| `fl-personalization-design-memo` | completed | 6 | 53 | 14 | passed |

These runs demonstrate:

- LangGraph planning and checkpointed execution
- Supervisor and researcher tool decisions
- Web search/source reading through Tavily
- Local corpus retrieval through Qdrant + SQLite FTS5/BM25
- Qwen/DashScope reranking
- Memory recall/write behavior
- Citation-locked report generation
- Evaluation and replay artifacts
- MCP readiness and workbench inspection

## Problems And Fixes

### Full PDF Ingestion Was Too Slow

Problem: Full synchronous PDF ingestion through `/v1/documents/ingest` was too
slow in strict provider mode. A single full paper could take several minutes
because page parsing, contextualization, embeddings, and Qdrant writes all used
real providers.

Fix: The demo script seeds bounded excerpts through `/v1/documents`. This still
exercises real embedding, BM25, Qdrant, graph signals, and reranking, but avoids
making interview prep depend on full-PDF indexing latency. Full ingestion remains
available; the demo path is intentionally bounded and reproducible.

Interview framing: this is a classic production trade-off. Keep the full
pipeline, but create a deterministic demo seed path so experiments are reliable.

### BOM-Prefixed `.env` Hid Strict Mode

Problem: The local `.env` file started with a UTF-8 BOM. `python-dotenv` treated
the first key as `\ufeffARC_STRICT_PROVIDERS`, so direct `load_settings()` calls
could miss `ARC_STRICT_PROVIDERS=true`.

Fix: `settings.py` now reads dotenv files with `encoding="utf-8-sig"`, and
`tests/test_settings.py` covers BOM-prefixed dotenv files.

Interview framing: provider readiness cannot depend on editor-specific encoding
behavior. Config parsing was made explicit and tested.

### Celery Jobs Stayed `queued` During Worker Warmup

Problem: A Celery worker constructs `ResearchCopilot` before executing a job.
Restoring and indexing an existing corpus can take minutes in strict mode, so
the API showed the job as `queued` even though the worker had already received
the task and was doing real model/embedding/Qdrant work.

Fix: `celery_app.py` now performs a lightweight SQLite status update as soon as
the task is received, then reuses a worker-level `ResearchCopilot` instance. The
worker refreshes state before each task so newly added documents/memory are
picked up incrementally instead of rebuilding the full corpus every time.

Interview framing: queue state and worker warmup are separate concerns. The fix
improves observability first, then reduces repeated cold-start cost.

### LLM Structured Output Used `null` For List Fields

Problem: Real model output returned `null` for supervisor list fields such as
`selected_tools`, `web_queries`, `internal_queries`, and
`sufficiency_criteria`. The strict Pydantic contract expected arrays, causing
valid research jobs to fail before route fallback logic could run.

Fix: `schemas.py` now normalizes `null` list fields to empty lists for
`SupervisorToolCall` and `SupervisorDecisionContract`, with coverage in
`tests/test_schemas.py`.

Interview framing: real model structured output is messy. Contracts should stay
strict about shape, but tolerate common provider serialization quirks when safe.

### Quality Gates Rejected A Weak Architecture Demo

Problem: An early architecture-focused demo question failed with:
`Plan coverage is weak` and `Evidence sufficiency is weak`. The system did
produce a run artifact, but the evaluator correctly marked it failed because the
available FL paper corpus was not enough evidence for every architecture claim.

Fix: Keep the failed run as a useful evaluation story, but change the stable
resume demo to corpus-grounded FL questions. Architecture claims are documented
from runtime config and source code, while research reports stay grounded in the
paper corpus.

Interview framing: this is a strength, not an embarrassment. The evaluator
blocked a report when evidence did not match the question.

### Real Providers Expose Latency And Query Noise

Problem: Strict runs surfaced real-world latency and a few Tavily `400` responses
for generated queries. Later generated queries succeeded, and completed runs
still produced sufficient evidence.

Fix: The demo profile reduces iterations and query rewrites for stability:
`ARC_RESEARCH_MAX_ITERATIONS=2` and `ARC_RAG_MAX_QUERY_REWRITES=1`. Future work
should add stronger search-query validation and a cold-start index cache.

Interview framing: this shows why traces matter. Provider errors, retries,
latency, and evaluator outcomes are visible instead of hidden.

## Resume Talking Points

Use this project as an AI Research Copilot rather than a private-data product.
The core story is:

- Designed a LangGraph agentic research workflow with planning, supervisor
  delegation, source reading, retrieval, reporting, verification, evaluation,
  memory writes, and replay.
- Built an Agentic RAG grounding layer with Qdrant dense vectors, SQLite
  FTS5/BM25, parent/neighbor context expansion, LightRAG-inspired graph signals,
  and Qwen/DashScope reranking.
- Added real-provider strict mode over OpenAI-compatible chat, Qwen/DashScope
  embeddings/rerank, Tavily search, Qdrant, Celery/Redis, and MCP tools.
- Produced reproducible demo artifacts: 5-paper corpus, 3 memory records, 2
  completed runs, trace/evaluation/report files, and MCP readiness 10/10.
- Found and fixed real integration issues: BOM dotenv parsing, Celery state
  visibility during worker warmup, repeated worker reindexing, and `null` list
  fields in structured LLM output.

## Remaining Boundaries

Keep these boundaries honest:

- Single-node Celery/Redis/SQLite/Qdrant, not a distributed platform.
- Controlled excerpts for stable demo, not a large production corpus.
- Proxy evaluation and saved artifacts, not a public benchmark.
- PyMuPDF text extraction, not OCR or enterprise document intelligence.
- Local MCP workbench, not an enterprise MCP gateway.
