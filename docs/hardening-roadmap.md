# Hardening Roadmap

## Current Assessment

The current architecture is coherent for an autumn-recruiting project when it is framed as a Conversational Research Agent Runtime rather than a commercial research assistant.

Already credible:

- ODR-style planning and supervision are implemented.
- Agent sessions, SQLite memory, plan confirmation, and the static Agent Workbench are implemented.
- The LangGraph workflow is explicit and test-covered.
- Agentic RAG is more than a thin top-k wrapper: it includes child chunks, parent/neighbor context, dense retrieval, BM25, graph signal fusion, and rerank.
- GitHub MCP can add developer source-of-truth evidence for repositories, code, issues, pull requests, and releases.
- Report generation is tied to existing evidence and checked by verifier/evaluator metrics.
- Trace, checkpoints, jobs, runs, and evaluation artifacts make the system explainable in interviews.

Kept out because they would weaken the story:

- local research-workbench MCP server

The reason is practical: the project now has local single-user agent memory, but it should not claim enterprise personalization or restore a self-calling MCP workbench. Those would look like broad framework features rather than evidence-backed project capabilities.

## What Needs To Be Hardened Now

The project should not add another large feature before the demo evidence is strong. The missing pieces are mostly proof, polish, and reproducibility.

### P0: Resume-Ready Proof

1. Build 2-3 repeatable demo topics:
   - open-source due diligence over a named repository
   - technical decision memo over two libraries or architectures
   - local corpus research over papers or project notes
2. Prepare a small real corpus for each local RAG demo.
3. Save report, trace, route, source-index, and evaluation artifacts for each topic.
4. Document one successful strict-provider run and one deterministic fallback run.
5. Keep the adoption memo lab current: team context pack, repo decision topic, generated report, trace, evaluation, and analysis.
6. Keep the conversational session demo current: saved team constraints, plan draft, confirmed job, completed run, memory, trace, and evaluation.
7. Keep `pytest` passing after the docs/demo cleanup.

### P1: Evaluation And Reliability

1. Expand `examples/eval-dataset.jsonl` with labeled expected sources and expected evidence types.
2. Add regression checks for citation coverage, source diversity, unsupported sections, and retrieval route visibility.
3. Add a GitHub MCP smoke test or runbook section that explains auth, allowlisted tools, expected evidence, and network fallback.
4. Add or document a run-bundle export path, for example a script that exports:
   - request
   - report markdown
   - source index
   - trace JSON
   - evaluation JSON
   - runtime config summary
5. Make failure states demo-friendly: provider missing, search missing, MCP auth missing, empty corpus, and reranker unavailable.
6. Add a small memory eval fixture so the extractor can be improved without silently polluting project memory.

### P2: Product Polish

1. Improve the Agent Workbench around the artifacts that matter: session memory, plan, evidence, citations, quality gates, trace, and replay.
2. Add example screenshots only after real demo artifacts exist.
3. Add a short "How to read a run" guide for interview preparation.
4. Consider exposing this app as an MCP Server later, but only as a facade over the stable API.

## Current Demo Standard

A convincing demo should contain:

- 2-3 stable research topics
- a small but real document corpus
- at least one successful strict-provider run per topic
- saved report, trace, route, evaluation, and source-index artifacts
- one external search path
- optional external GitHub MCP evidence for repository/code/issue/PR/release facts

An empty corpus plus strict model keys proves the runtime starts, but it does not prove the system is research-grade.

## MCP Decision

MCP should be used only when it extends evidence coverage.

Recommended:

- GitHub MCP remote read-only endpoint for developer source-of-truth evidence
- paper/search MCP only when the demo topic genuinely needs scholarly metadata beyond the configured search provider

Avoid:

- full research-agent MCPs that duplicate this repo's planner/supervisor/reporter
- local self-calling demo MCPs
- generic search MCPs that mostly duplicate Tavily without adding a new evidence type

Future:

- expose this app as an MCP Server with `run_research`, `search_local_corpus`, and `inspect_research_run`
- keep that as a separate API facade over the stable core

## What Not To Add Now

- enterprise-grade personalization memory
- multi-user account memory
- browser automation
- multi-agent theater with no separate responsibility
- distributed cluster claims
- a second local MCP server just for demos
- a full frontend rebuild before the research runtime is understood
- more provider integrations before the current provider path is demo-stable

## Interview Framing

Use this line:

> I kept the project narrow: a local conversational research agent with SQLite memory, a confirmation gate, supervised research graph, bounded tool use, hybrid retrieval, citation-locked synthesis, and replayable quality evaluation.

This is stronger than pretending the app is a complete enterprise research platform.
