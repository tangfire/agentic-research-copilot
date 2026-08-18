# Agentic Research Runtime

`AI Research Copilot` is the repository and API product name, but the project should now be presented as a **Conversational Research Agent Runtime with HITL, tool policy, memory evaluation, workspace control, and constraint coverage**: a single-node AI engineering experiment for understanding how Deep Research / Codex-like systems can be built, configured with local knowledge, inspected, evaluated, and replayed.

The goal is not to beat mature general-purpose agents. The goal is to implement the engineering skeleton behind a citation-grounded research agent: session state, memory, interactive planning, human confirmation, step visibility, tool policy, approval artifacts, evidence routing, GitHub MCP integration, local Agentic RAG, report synthesis, constraint coverage, verification, evaluation, and trace replay.

See [Product Positioning](docs/product-positioning.md), [Architecture](docs/architecture.md), [Research Desk v3 Architecture](docs/research-desk-v3-architecture.zh-CN.md), [OpenClaw / Hermes Design Notes](docs/openclaw-hermes-design-notes.zh-CN.md), [Agent Maturity Pack](docs/agent-maturity-pack.zh-CN.md), [Tool Loop And HITL](docs/tool-loop-and-hitl.zh-CN.md), [Memory And Constraint Evaluation](docs/memory-and-constraint-eval.zh-CN.md), [Agent Reference Stack](docs/agent-reference-stack.zh-CN.md), [Autumn Recruiting Playbook](docs/autumn-recruiting-playbook.zh-CN.md), [Interview Question Bank](docs/interview-question-bank.zh-CN.md), [Demo Script](docs/demo-script.zh-CN.md), [Usage Guide](docs/usage-guide.zh-CN.md), [Hardening Roadmap](docs/hardening-roadmap.md), and [Chinese Interview Notes](docs/interview-notes.zh-CN.md) for the intended project boundary.

## Honest Positioning

This project is strongest when described as:

> A learning-by-building AI engineering lab for complex technical research. It turns an open-ended engineering question into a planned research graph, routes evidence across web search, GitHub MCP, and local documents, then produces a citation-backed report with verifier, evaluator, trace, and replay artifacts.

Do not pitch it as:

- a commercial replacement for Codex, ChatGPT Deep Research, or OpenAI Deep Research
- a generic chatbot without research artifacts
- a private-data assistant
- a GitHub-only analyzer
- an MCP marketplace or wrapper around another deep-research agent
- a full agent SDK or distributed SaaS platform

The interview value is the runtime design and the observable artifacts, not the claim that this small project is smarter than a frontier product.

## Current Boundary

The current core path is:

```text
chat session -> memory -> clarify/plan -> confirm -> step stream -> tool policy/approval -> supervise research -> search/read/retrieve -> synthesize -> constraint gate -> verify/evaluate -> persist trace
```

The conversational layer owns `AgentSession`, `AgentMessage`, `AgentPlanDraft`, `AgentRunStep`, `AgentToolDefinition`, `ToolInvocation`, `ApprovalRequest`, `MemoryExtractionResult`, `ConstraintCoverage`, and SQLite-backed memory. The research runtime underneath stays focused on `ResearchRequest -> ResearchRun`: planning, tool loop, retrieval, report, evaluation, trace, and replay. This keeps the product boundary clean: the agent layer handles conversation, confirmation, policy, and session-visible observability, while the existing runtime handles research execution.

The local research-workbench MCP server was also removed. MCP is now kept as an external tool boundary through `mcp_tools.py`: when `ARC_MCP_SERVER_URL` and `ARC_MCP_TOOLS` are configured, the researcher may call allowlisted external tools with structured arguments and convert their results into evidence. The recommended demo integration is GitHub MCP for repository, code, issue, pull request, and release evidence.

The conversational layer now also carries a workspace control plane and a small skill/playbook catalog. That lets the product explain not only "what research is running" but also "which team context, which playbook, and which guardrails shaped this run."

The skill layer is now backed by local skill packs under `skills/`: each pack can ship a `skill.json` manifest, a `SKILL.md` instruction file, and optional whitelist-only scripts that run through a controlled JSON stdin/stdout boundary. This is intentionally smaller than a full plugin marketplace, but it is real enough to demonstrate discoverable skills, instruction loading, preflight hooks, and safe local execution.

The v4 harness keeps the multi-agent story narrow: `RepoSignalAgent`, `ArchitectureFitAgent`, and `OpsRiskAgent` are stable specialist lanes for open-source adoption review. They do not replace the underlying planner/researcher/reporter; they make route decisions, evidence ownership, conflicts, and benchmark summaries visible in the run artifact and session export.

## What The System Does

1. `ConversationalResearchAgent` stores chat sessions, extracts memory, loads relevant constraints, and decides whether to clarify or draft a plan.
2. `AgentPlanDraft` presents a readable research brief, plan items, assumptions, and success criteria. It requires user confirmation.
3. `AgentRunStep` records message, planning, approval, research, report, verification, evaluation, and failure stages so the workbench can inspect the run while it progresses.
4. `AgentToolDefinition`, `ToolInvocation`, and `ApprovalRequest` make tool status, MCP auth gaps, and approval decisions explicit instead of hidden in code.
5. `ResearchCopilot` starts only after confirmation and turns the plan into a `ResearchRequest`.
6. `Clarifier` checks whether the user request is specific enough.
7. `Planner` writes a research brief and decomposes the topic into focused plan items.
8. `ResearchSupervisor` emits ODR-style `think_tool`, `ConductResearch`, and `ResearchComplete` tool calls.
9. `Researcher` runs a bounded tool loop for each delegated unit: `web_search`, optional external `mcp_tool`, or completion.
10. `Retriever` grounds uploaded documents and project memory with child chunk retrieval, parent/neighbor expansion, dense retrieval, BM25, graph signal fusion, and reranking.
11. `Reporter` writes topic-specific sections from notes and evidence. It does not use fixed demo sections.
12. `ConstraintCoverage` checks hard project constraints against report sections and evidence, adding warnings or failing evaluation when coverage is too weak.
13. `MultiAgentHarness` maps plan items to `RepoSignalAgent`, `ArchitectureFitAgent`, and `OpsRiskAgent`, then writes role assignments, route decisions, conflicts, an evidence ledger, and benchmark proxy metrics.
14. `Verifier` and `RAGEvaluator` check citation coverage, evidence sufficiency, source diversity, context precision, and unsupported sections.
15. `RunLedger`, SQLite storage, telemetry, and LangGraph checkpoints make the run inspectable; frozen replay reuses saved artifacts instead of re-calling live tools.

## Best Demo Modes

Use one of these narrow modes instead of presenting the app as an everything assistant:

1. **Open-source Due Diligence**
   Input a GitHub repository or `owner/repo`, then inspect architecture, implementation files, issue risks, PR activity, release signals, ecosystem context, and adoption concerns.
2. **Technical Decision Memo**
   Compare two libraries, architectures, or retrieval/agent designs, then generate an ADR-style report with citations, evidence gaps, and confidence notes.
3. **Local Corpus Research**
   Ingest papers, notes, or architecture docs, then show child chunk hits, parent/neighbor context expansion, dense/BM25/graph fusion, rerank, final citations, and evaluation metrics.

For autumn recruiting, the first mode is easiest to explain because GitHub MCP, web search, and local adoption notes each have a clear evidence role.

## Realistic Adoption Memo Lab

The most product-like local experiment is:

> Input a repository, a decision question, and a saved team context pack; output a citation-backed technical adoption memo with trace and evaluation metrics.

Run:

```powershell
python scripts/run_adoption_memo_experiment.py --clean
```

The default lab run now uses the configured real provider stack: real chat model, external search, real embeddings, real rerank, local persistent Qdrant, trace, and evaluation. Deterministic mode is kept only for offline regression tests:

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode deterministic
```

For GitHub MCP evidence, add `--use-mcp`. The run will fail fast when GitHub MCP auth is missing instead of silently falling back to web-only evidence:

```powershell
python scripts/run_adoption_memo_experiment.py --clean --mode real --use-mcp
```

Smoke-test GitHub MCP before a full run:

```powershell
python scripts/check_github_mcp.py
```

The lab seeds fictional but realistic small-team constraints from `examples/adoption-lab/team-context/`, reviews `langchain-ai/langgraph` for a Python/FastAPI platform team, and writes report, trace, evaluation, and analysis artifacts to `examples/adoption-lab/outputs/`.

See [Adoption Memo Lab](docs/adoption-memo-lab.zh-CN.md) for the Chinese walkthrough.

## What Still Needs To Be Hardened

The agent session, memory, confirmation gate, step stream, tool policy, approval artifacts, memory quality view, constraint coverage gate, static workbench, and public APIs now exist. The next useful work is stronger proof artifacts and deeper real-provider demos:

1. Build 2-3 repeatable demo corpora and topics.
2. Generate saved report, trace, route, source-index, and evaluation bundles for each demo.
3. Add a stable GitHub MCP smoke demo with a documented fallback when auth or network access is unavailable.
4. Expand the eval dataset with labeled retrieval/citation expectations.
5. Add or document a run-bundle export command for resume and interview review.
6. Improve memory extraction with an LLM-backed extractor once there is a small evaluation set.
7. Upgrade approval from observable HITL to durable graph interrupt/resume if the project later needs true paused tool execution.
8. Polish report, evidence, quality gates, and trace replay rather than adding unrelated product surfaces.

These items are tracked in [Hardening Roadmap](docs/hardening-roadmap.md).

## Technology Fit

- `LangGraph`: fits the conditional research workflow: plan, delegate, run tools, verify, revise, and finalize.
- `FastAPI`: provides a small local API for jobs, documents, runs, traces, evaluation, replay, and runtime config.
- `SQLite`: persists sessions, messages, plan drafts, memory, jobs, runs, and trace metadata for a single-user local workbench.
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
- Avoid: MCP servers marketed as complete deep-research agents or generic web-search duplicates. They overlap with the core runtime instead of extending it.

Future direction: this project itself is a reasonable MCP Server candidate, but that should be a separate outward-facing facade exposing tools like `run_research`, `search_local_corpus`, and `inspect_research_run`. It should not reintroduce the removed local workbench that called the app from inside itself.

## API Surface

- `POST /v1/agent/sessions`
- `GET /v1/agent/sessions`
- `GET /v1/agent/sessions/{session_id}`
- `POST /v1/agent/sessions/{session_id}/messages`
- `POST /v1/agent/sessions/{session_id}/confirm-plan`
- `POST /v1/agent/sessions/{session_id}/cancel`
- `GET /v1/agent/sessions/{session_id}/memory`
- `GET /v1/agent/sessions/{session_id}/memory/evaluation`
- `GET /v1/agent/sessions/{session_id}/steps`
- `GET /v1/agent/sessions/{session_id}/events`
- `GET /v1/agent/sessions/{session_id}/tool-invocations`
- `POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/approve`
- `POST /v1/agent/sessions/{session_id}/approvals/{approval_id}/reject`
- `GET /v1/agent/tools`
- `POST /v1/memory`
- `GET /v1/memory`
- `DELETE /v1/memory/{memory_id}`
- `POST /v1/research/clarify`
- `POST /v1/research/runs`
- `GET /v1/research/runs`
- `GET /v1/research/runs/{run_id}`
- `GET /v1/research/runs/{run_id}/trace`
- `GET /v1/research/runs/{run_id}/evaluation`
- `GET /v1/research/runs/{run_id}/harness`
- `GET /v1/research/runs/{run_id}/constraint-coverage`
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

The `/v1/research/*` APIs remain the lower-level runtime. The `/v1/agent/*` and `/v1/memory` APIs are the conversational workbench layer on top.

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

`ARC_MCP_AUTH_TOKEN` may also be supplied through `GH_TOKEN`, `GITHUB_TOKEN`, or `GITHUB_PERSONAL_ACCESS_TOKEN`. The adoption memo runner forces the GitHub read-only endpoint and GitHub tool allowlist when `--mode real --use-mcp` is used, so stale local MCP tools are not counted as GitHub evidence.

## Run And Test

Install:

```powershell
pip install -e .[dev]
```

Run tests:

```powershell
pytest
```

Run the harness benchmark:

```powershell
python scripts/run_harness_benchmark.py --clean --max-tasks 24
```

Start API:

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

Open the local workbench at `http://127.0.0.1:8000/`.

## Quick Use

打开页面后只需要记住这条路径：

```text
新建会话 -> 输入问题和团队约束 -> 发送 -> 查看计划 -> 确认并开始研究 -> 查看结果
```

页面默认只展示下一步、计划、记忆和结果。路由、工具、Skill、trace 和质量明细都收在“高级信息”里。

完整的中文操作说明见 [Usage Guide](docs/usage-guide.zh-CN.md)。

## Study Path

Read in this order:

1. `docs/product-positioning.md`
2. `docs/architecture.md`
3. `docs/agent-maturity-pack.zh-CN.md`
4. `docs/tool-loop-and-hitl.zh-CN.md`
5. `docs/memory-and-constraint-eval.zh-CN.md`
6. `docs/interview-question-bank.zh-CN.md`
7. `docs/source-map.md`
8. `docs/learning/zh/ai_research_copilot_learning_guide_zh.md`
9. `src/agentic_research_copilot/agent.py`
10. `src/agentic_research_copilot/schemas.py`
11. `src/agentic_research_copilot/providers.py`
12. `src/agentic_research_copilot/graph_runtime.py`
13. `src/agentic_research_copilot/pipeline.py`
14. `src/agentic_research_copilot/agents`
15. `src/agentic_research_copilot/retrieval/store.py`

## Interview Framing

The strongest story is:

> I built a single-node Agentic Research Runtime inspired by Open Deep Research. The core difficulty is not CRUD or chat; it is turning an open-ended technical question into a supervised research graph with structured planning, bounded tool use, hybrid retrieval, citation-locked report generation, verifier/evaluator quality gates, and replayable trace artifacts.

After the agent maturity upgrade:

> I added the missing agent product layer: sessions, memory, human confirmation, step visibility, tool policy, approval artifacts, memory evaluation, constraint coverage, and a workbench UI. A user can save team constraints once, discuss a technical adoption question, inspect the generated plan, confirm it, then receive the report, evidence, trace, quality gates, and evaluation in the same session.

If asked "why not just use Codex?", answer:

> Mature products are absolutely stronger as end-user assistants. This project is not trying to replace them. It is a learning-by-building implementation of the mechanisms behind that class of systems: stateful orchestration, tool routing, approval boundaries, evidence contracts, local retrieval, citation grounding, constraint coverage, evaluation, and replay.

Do not overclaim distributed execution, enterprise memory, browser automation, multi-user SaaS, or a general agent platform. The project is strongest when described as an inspectable conversational research runtime with a credible Agentic RAG stack.
