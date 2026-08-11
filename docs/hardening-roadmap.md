# Hardening Roadmap

## Current Assessment

The current architecture is coherent for an autumn-recruiting project when it is framed as a technical research and engineering-intelligence copilot:

- ODR-style planning and supervision are real.
- The LangGraph workflow is explicit and test-covered.
- Agentic RAG is not a thin wrapper: it includes child chunks, parent/neighbor context, dense retrieval, BM25, graph signal fusion, and rerank.
- GitHub MCP adds developer source-of-truth evidence for repositories, code, issues, pull requests, and releases.
- Report generation is tied to existing evidence and checked by verifier/evaluator metrics.
- Trace, checkpoints, jobs, runs, and evaluation artifacts make the system explainable in interviews.

Two features were removed because they weakened the story:

- project memory as a core module
- local research-workbench MCP server

The reason is practical: without strong real corpus/history assets, those modules looked like broad framework features rather than industrial-quality product capabilities.

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

## Next Hardening Items

1. Build 2-3 repeatable demo corpora.
2. Save report/trace/evaluation artifacts for each demo topic.
3. Keep the GitHub MCP smoke demo stable and documented.
4. Expand pipeline integration tests for GitHub MCP evidence and trace visibility.
5. Improve graph extraction quality with stricter entity/relation schemas and duplicate normalization.
6. Add a small labeled eval set for retrieval and citation quality.
7. Add a command that exports a run bundle for resume/interview review.

## What Not To Add Now

- generic chat memory
- user preference storage
- browser automation
- multi-agent theater with no separate responsibility
- distributed cluster claims
- a second local MCP server just for demos
- a full frontend rebuild before the research runtime is understood

## Interview Framing

Use this line:

> I intentionally removed modules that were broader than the evidence supported. The final project focuses on the hard part: a supervised research graph, bounded tool use, hybrid retrieval, citation-locked synthesis, and replayable quality evaluation.

This is stronger than pretending the app is a complete enterprise research platform.
