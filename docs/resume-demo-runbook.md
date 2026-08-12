# Resume Demo Runbook

## Goal

Prepare repeatable assets that show the repo as an Agentic Research Runtime, not as a generic chatbot or CRUD backend. A good demo should answer an engineering question by combining web search, optional GitHub MCP developer evidence, local RAG grounding, trace replay, and evaluation metrics.

Capture:

- grounded document corpus
- strict-provider run
- final report
- source index
- trace timeline
- retrieval routes
- evaluation metrics
- runtime config summary

Recommended primary demo:

> Analyze the architecture, implementation choices, issue risks, PR activity, and release changes of an open-source AI agent or RAG project.

## Preflight

Use strict providers when preparing real demo artifacts:

```powershell
$env:ARC_STRICT_PROVIDERS="true"
$env:ARC_MODEL_PROVIDER="openai_compatible"
$env:ARC_SEARCH_PROVIDER="tavily"
$env:ARC_QDRANT_URL="http://127.0.0.1:6333"
$env:ARC_RERANK_PROVIDER="dashscope"
```

Run:

```powershell
pytest
```

Then start the API:

```powershell
uvicorn agentic_research_copilot.server:create_app --factory --host 127.0.0.1 --port 8000
```

## Demo Topics

Use 2-3 topics that you can explain without pretending this is a commercial product.

### Topic A: Open-source Due Diligence

```text
Analyze https://github.com/langchain-ai/open_deep_research:
explain its research workflow, inspect relevant implementation files,
summarize issue risks, review recent pull request activity, and report release changes.
Use Tavily for ecosystem context and GitHub MCP for repository-level evidence.
```

### Topic B: Technical Decision Memo

```text
Compare LangGraph and AutoGen for building a bounded technical research agent.
Focus on workflow state, tool routing, human inspectability, failure handling,
and evaluation/replay support.
```

### Topic C: Local Corpus Research

```text
Using the uploaded architecture notes and RAG papers, explain why this project
uses child chunk retrieval, parent/neighbor expansion, BM25, graph signal fusion,
and rerank instead of plain top-k vector search.
```

## Corpus

Use a small corpus that you can explain:

- 2-5 papers, architecture notes, or technical docs
- stable URLs or local PDFs/Markdown files
- one topic around agentic research or RAG
- one topic around system architecture or evaluation

The corpus should make retrieval visible. Do not rely only on live web search.

## Run

Submit a research run:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/research/runs `
  -ContentType "application/json" `
  -Body '{"topic":"Compare Open Deep Research style agentic research with hybrid RAG grounding","depth":"standard"}'
```

Save the returned `run_id`.

Inspect:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/research/runs/<run_id>/trace
Invoke-RestMethod http://127.0.0.1:8000/v1/research/runs/<run_id>/evaluation
Invoke-RestMethod http://127.0.0.1:8000/v1/research/runs/<run_id>/result
```

## Optional GitHub MCP

For interview demos, prefer GitHub MCP as the external MCP. It gives concrete developer evidence that Tavily does not expose as precisely: repository files, code search, issues, pull requests, and releases.

```powershell
$env:ARC_MCP_ENABLED="true"
$env:ARC_MCP_SERVER_URL="https://api.githubcopilot.com/mcp/readonly"
$env:ARC_MCP_TOOLS="search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release"
$env:ARC_MCP_AUTH_REQUIRED="true"
$env:ARC_MCP_AUTH_TOKEN="<github-token>"
```

Use an explicit repository target when demonstrating this path. The provider extracts the repository target as structured `owner` and `repo` hints. The researcher can then select direct repository tools instead of treating the entire request as a generic web query. For a broad topic without a named repository, it may first use `search_repositories` or `search_code` to discover candidates.

If GitHub MCP auth or network access is unavailable during an interview, say so directly and show a saved run bundle or deterministic fallback instead of debugging live credentials.

## What To Capture

For each demo run, be ready to show:

- the original topic
- generated plan items
- `ConductResearch` tool calls
- selected tools per route
- web/internal/MCP evidence counts
- one retrieved document chunk with parent/neighbor context metadata
- final report citations
- evaluator notes and pass/fail metrics
- trace events showing handoffs and tool calls

## How To Explain It

Use this flow:

1. Start with the narrow experiment: "I implemented an inspectable research-agent runtime."
2. Show the plan and supervisor decisions.
3. Show one external evidence path and one local RAG path.
4. Show the final report citations.
5. Show verifier/evaluator output.
6. Show trace replay.
7. Explain what you intentionally removed: project memory and local self-calling MCP.

## Interview Talking Points

- The project is not plain RAG; RAG is one evidence channel inside a research graph.
- The supervisor decides what to delegate and what evidence tools to use.
- The retriever searches precise child chunks but returns expanded surrounding context for synthesis.
- GitHub MCP is useful for repository facts, not as a replacement for the whole agent.
- The report is generated from existing evidence, then verified and evaluated.
- Memory and the local workbench MCP were removed to keep the project honest and less toy-like.
- Codex is stronger as a product; this repo is valuable as an implementation and inspection of the underlying runtime ideas.
