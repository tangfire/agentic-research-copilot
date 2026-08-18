# Product Positioning

## One Sentence

AI Research Copilot is best presented as a **Conversational Research Agent Runtime**: a single-node AI engineering experiment that turns a chat session, saved team constraints, and an open-ended technical question into a confirmed, citation-backed, traceable research run using web search, GitHub MCP, and local Agentic RAG evidence.

## Honest Framing

This is not a strong commercial product thesis yet. A mature agent such as Codex or Deep Research will usually be a better end-user tool.

The credible value of this repo is different:

- it implements the runtime mechanics behind a research agent
- it adds the product-facing agent layer: sessions, memory, plan confirmation, and job binding
- it exposes plans, tool decisions, routes, evidence, citations, evaluation, and trace artifacts
- it gives the author concrete engineering material to discuss in interviews
- it keeps features narrow enough that implementation quality can be inspected

Use the project as a learning-by-building lab, not as a claim that a student-built copilot beats frontier products.

## Target Scenarios

### Open-source Due Diligence

Input: a GitHub URL or `owner/repo` plus an adoption context.

Output: architecture notes, implementation evidence, issue/PR/release risks, ecosystem context, adoption recommendation, citations, source index, evaluation, and trace.

This is the recommended autumn-recruiting demo because every evidence channel has a natural role:

- GitHub MCP: repository, code, issues, pull requests, releases
- Web search: docs, ecosystem context, comparisons, external reports
- Local RAG: optional adoption criteria, team constraints, prior notes, architecture docs

### Technical Decision Memo

Input: an engineering comparison such as "LangGraph vs AutoGen for a bounded research agent" or "Qdrant vs Milvus for a small local RAG system".

Output: ADR-style recommendation with cited trade-offs, risks, evidence gaps, and confidence.

### Local Corpus Research

Input: papers, notes, docs, or architecture files plus a focused question.

Output: grounded report with visible retrieval routes, child chunk hits, parent/neighbor expansion, dense/BM25/graph fusion, rerank, citations, and evaluation.

## Most Realistic Product Scenario

The strongest deployable scenario is:

> A technical adoption memo workbench for small engineering teams.

The team stores recurring constraints once: stack, deployment boundary, risk tolerance, adoption checklist, evaluation requirements, and graph-orchestration rules. A user then inputs a repository and decision question. The runtime retrieves those local constraints, searches public evidence, optionally calls GitHub MCP, produces a cited adoption memo, and saves trace/evaluation artifacts.

The repository includes this scenario under `examples/adoption-lab/` and a repeatable runner at `scripts/run_adoption_memo_experiment.py`.

## Evidence Source Boundaries

- `web_search`: broad public context, official docs, blogs, papers, benchmarks, ecosystem discussions, and current background.
- `mcp_tool` with GitHub MCP: repository files, source code, issues, pull requests, releases, and developer source-of-truth evidence.
- `vector_retrieval`: local documents and imported technical corpus through Agentic RAG.
- `run-artifact`: the system's own plan, routes, query rewrites, trace, and evaluation metadata for replay and interview explanation.

Each channel becomes `EvidenceItem` objects so the reporter, verifier, evaluator, API, and trace view can share the same citation contract.

## Repository Targeting

GitHub MCP is selected according to the research target, not used for every question:

- For a broad open-source topic, the researcher may use `search_repositories` or `search_code` to discover relevant projects before inspecting one of them.
- For an explicit GitHub URL or `owner/repo` target, the provider receives a repository hint and can call repository-aware tools such as `get_file_contents`, `search_code`, `list_issues`, `list_pull_requests`, or `get_latest_release` with structured arguments.
- For an engineering decision that needs broader context, GitHub evidence is combined with `web_search` and, when relevant, local RAG instead of replacing them.

This makes repository research a concrete workflow inside the runtime. It is not a guarantee that every technical question will call GitHub; the model still chooses the source that matches the evidence gap.

## Non-Goals

- It is not a private-data assistant or enterprise long-term personalization memory product.
- It is not a generic CRUD backend.
- It is not a complete agent platform or agent SDK.
- It is not a GitHub-only analyzer; GitHub MCP is one evidence channel inside a broader research workflow.
- It is not an MCP marketplace or a wrapper around another deep-research agent.
- It is not a distributed research platform.
- It is not trying to outperform Codex as a general-purpose assistant.

## Answering "Why Not Just Use Codex?"

Use this answer:

> Codex is a mature end-user product and is obviously stronger as a general assistant. My project is not trying to replace it. I built a smaller inspectable runtime to understand and demonstrate how this class of systems works internally: session state, memory, interactive planning, structured tool calls, bounded researcher loops, evidence contracts, local RAG, citation grounding, verification, evaluation, and trace replay.

This turns the comparison from product capability into engineering understanding.

## Interview Framing

The strongest framing is:

> I built a Conversational Research Agent Runtime inspired by Open Deep Research. The core work is a session and memory layer in front of a supervised LangGraph workflow: interactive planning, bounded tool routing, GitHub MCP developer evidence, local Agentic RAG, citation-locked synthesis, verification, evaluation, and trace replay.

This positioning keeps the project specific enough to be credible while still showing a complete large-model application stack.
