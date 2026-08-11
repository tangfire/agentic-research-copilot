# Product Positioning

## One Sentence

AI Research Copilot is a technical research and engineering-intelligence assistant. It turns an open-ended technical question into a planned research run, chooses between web search, GitHub MCP, and local RAG evidence, and produces a citation-backed report with trace and evaluation artifacts.

## Target Use Cases

- Open-source project research: architecture, implementation details, issue risks, pull request activity, release changes, and maintainer signals.
- Engineering decision research: compare frameworks, libraries, retrieval designs, agent architectures, and deployment trade-offs.
- Resume/interview project analysis: produce inspectable reports, source indexes, traces, retrieval routes, and evaluation metrics for technical topics.
- Local-corpus grounding: combine imported papers, docs, notes, and architecture files with external technical evidence.

## Evidence Source Boundaries

- `web_search`: broad public context, official docs, blogs, papers, benchmarks, ecosystem discussions, and current background.
- `mcp_tool` with GitHub MCP: repository files, source code, issues, pull requests, releases, and developer source-of-truth evidence.
- `vector_retrieval`: local documents and imported technical corpus through Agentic RAG.
- `run-artifact`: the system's own plan, routes, trace, and evaluation metadata for replay and interview explanation.

## Repository Targeting

GitHub MCP is selected according to the research target, not used for every question:

- For a broad open-source topic, the researcher may use `search_repositories` or `search_code` to discover relevant projects before inspecting one of them.
- For an explicit GitHub URL or `owner/repo` target, the provider receives a repository hint and can call repository-aware tools such as `get_file_contents`, `search_code`, `list_issues`, `list_pull_requests`, or `get_latest_release` with structured arguments.
- For an engineering decision that needs broader context, GitHub evidence is combined with `web_search` and, when relevant, local RAG instead of replacing them.

This makes repository research a concrete workflow inside the copilot. It is not a guarantee that every technical question will call GitHub; the model still chooses the source that matches the evidence gap.

## Non-Goals

- It is not a private-data assistant or long-term personalization memory product.
- It is not a generic CRUD backend.
- It is not a complete agent platform or agent SDK.
- It is not a GitHub-only analyzer; GitHub MCP is one evidence channel inside a broader research workflow.
- It is not an MCP marketplace or a wrapper around another deep-research agent.

## Interview Framing

The strongest framing is:

> I built a technical research copilot inspired by Open Deep Research. The core work is a supervised LangGraph research runtime: structured planning, bounded tool routing, GitHub MCP developer evidence, local Agentic RAG, citation-locked synthesis, verification, evaluation, and trace replay.

This positioning keeps the project specific enough to be credible while still showing a complete large-model application stack.
