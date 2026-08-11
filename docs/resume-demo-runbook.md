# Resume Demo Runbook

## Goal

Prepare repeatable assets that show the product as a technical research copilot, not as a generic chatbot or CRUD backend. A good demo should answer an engineering question by combining web search, GitHub MCP developer evidence, local RAG grounding, trace replay, and evaluation metrics.

Capture:

- grounded document corpus
- strict-provider run
- final report
- source index
- trace timeline
- retrieval routes
- evaluation metrics

Recommended demo shape:

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

## Corpus

Use a small corpus that you can explain:

- 2-5 papers or technical docs
- stable URLs or local PDFs
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

Use an explicit repository target when demonstrating this path:

```text
Analyze https://github.com/langchain-ai/open_deep_research:
explain its research workflow, inspect relevant implementation files,
summarize issue risks, review recent pull request activity, and report release changes.
Use Tavily for ecosystem context and GitHub MCP for repository-level evidence.
```

The provider extracts the repository target as structured `owner` and `repo` hints. The researcher can then select direct repository tools instead of treating the entire request as a generic web query. For a broad topic without a named repository, it may first use `search_repositories` or `search_code` to discover candidates.

```powershell
$env:ARC_MCP_ENABLED="true"
$env:ARC_MCP_SERVER_URL="https://api.githubcopilot.com/mcp/readonly"
$env:ARC_MCP_TOOLS="search_repositories,get_file_contents,search_code,list_issues,issue_read,search_issues,list_pull_requests,pull_request_read,get_latest_release"
$env:ARC_MCP_AUTH_REQUIRED="true"
$env:ARC_MCP_AUTH_TOKEN="<github-token>"
```

Use this for topics like: "Analyze the architecture, issue risks, PR activity, and release changes of an open-source AI agent or RAG project."

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

## Interview Talking Points

- The project is not plain RAG; RAG is one evidence channel inside a research graph.
- The supervisor decides what to delegate and what evidence tools to use.
- The retriever searches precise child chunks but returns expanded surrounding context for synthesis.
- The report is generated from existing evidence, then verified and evaluated.
- Memory and the local workbench MCP were removed to keep the project honest and less toy-like.
