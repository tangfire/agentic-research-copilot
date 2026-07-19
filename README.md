# Agentic Research Copilot

Multi-agent research assistant built from MIT-licensed open-source references and original glue code.

It turns a user question into a citation-backed report by combining:

- task planning and decomposition
- parallel web research
- retrieval over private documents
- short-term and long-term memory
- agent handoff and verification
- observability, cost tracking, and replay

## Why this project

- It is a strong way to learn agent, RAG, memory, and tool calling.
- It is easier to explain in interviews than a generic chat bot.
- It can be packaged honestly as a derivative/assembly project with clear attribution.

## MVP Scope

1. User submits a research topic.
2. Planner agent breaks the topic into sub-questions.
3. Research agents gather evidence from the web.
4. Retriever searches uploaded documents and project notes.
5. Memory service stores user preferences and canonical facts.
6. Verifier agent checks citations, conflicts, and missing evidence.
7. Report generator writes a final markdown report with sources.
8. Run ledger stores traces, token usage, and replay inputs.

## Suggested Stack

- Backend: Python 3.11, FastAPI, LangGraph
- Search: Tavily or other web search tools
- Retrieval: PostgreSQL + pgvector or Chroma
- Memory: structured memory tables + summary memory
- UI: Next.js + chat/report view
- Infra: Docker Compose, structured logs, trace/cost dashboards

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
- The first target is a runnable API with an in-memory research pipeline.
- The next step is to add retrieval, memory, verification, and observability one by one.

## Resume-safe phrasing

Use wording like:

> Built a multi-agent research copilot by assembling and extending MIT-licensed open-source agent and research frameworks.

Do not present upstream code as your original work.

## Source Map

- `open_deep_research`: research planning, multi-agent research, citation-backed reports
- `praisonaiagents`: memory, handoff, RAG, tracing, workflow primitives
- this repo: product shape, API, storage model, UI, run ledger, evaluation, deployment
