# Adoption Memo Experiment Analysis

## Simulated User Input

Please simulate Northstar Platform, a 5-engineer Python/FastAPI platform team, and evaluate whether the GitHub repository langchain-ai/langgraph should be piloted as the workflow runtime for an internal open-source adoption memo and technical decision research desk. Combine local team constraints, public documentation or GitHub/Web evidence, whether graph structure is actually necessary, risks, a pilot plan, and a rollback plan.

## Headline Metrics

- Status: `completed`
- Evaluation passed: `True`
- Source count: 5
- Context recall: 1.0
- Citation precision: 1.0
- Faithfulness proxy: 0.8015
- Expected term recall: 0.75
- Team constraint recall: 0.5
- Constraint coverage passed: `False`
- Expected source recall: 0.6667

## Routing And Trace

- Route counts: `{'hybrid': 4}`
- Tool counts: `{'web_search': 4, 'vector_retrieval': 4}`
- Evidence channels: `{'external': 12, 'document-chunk': 6, 'run-artifact': 1}`
- Trace events: 49
- Checkpoints: 12
- Handoffs: 11
- Actors: `['ArchitectureFitAgent', 'OpsRiskAgent', 'evaluator', 'multi_agent_harness', 'planner', 'reporter', 'research_supervisor', 'researcher', 'supervisor', 'verifier']`

## Graph Retrieval Check

- Document hits: 16
- Graph-enabled document hits: 0
- Graph signal hits: 0
- Matched graph entities sample: `[]`
- Matched graph relationships sample: `[]`

Interpretation: graph design is justified only when the retrieved team context contains workflow stages, dependencies, quality gates, or revision paths. If graph_signal_hits is 0, the system still used graph-enabled indexing, but this run did not prove that graph signal improved retrieval.

## Matched Expectations

- Terms: `['LangGraph', 'checkpoint', 'trace', 'revision', 'FastAPI', 'citation', 'evaluation', 'pilot', 'rollback']`
- Constraints: `['5-engineer', 'Python 3.11', 'FastAPI', 'replayable runs']`
- Source patterns: `['github.com/langchain-ai/langgraph', 'langgraph']`

## Evaluation Notes

- No evaluator notes.

## Product Findings

1. The strongest real use case is a repeatable adoption memo, not a generic chatbot. The local team context pack removes the need to paste the same constraints into every prompt.
2. The current runtime already has a real graph: planner, research supervisor, parallel research, reporter, verifier/evaluator, revision, and finalize. In this scenario the graph is conceptually appropriate because evidence sufficiency and citation gates can change the path.
3. GitHub MCP is a separate source-of-truth evidence channel. In real MCP mode the lab now forces the GitHub read-only endpoint and fails fast when auth is missing, so web-only evidence cannot be mistaken for MCP evidence.
4. The next product surface should be a first-class adoption memo preset: repo, decision question, team context pack, generated report, trace, and metrics in one bundle.

## Issues Found And Fixed

1. Real provider timeouts could abort the run during long reporter/verifier calls. The reporter input is now compacted and the real lab is the default run mode.
2. Budgeted indexing graph extraction was over-collecting structure words such as `The` and `Input`. A small stopword filter makes graph signal more trustworthy while keeping the real research/report path on real providers.
3. MCP configuration used to inherit stale local workbench tools. Real MCP lab runs now force the GitHub read-only endpoint and GitHub tool allowlist.

## Provider Snapshot

- Chat: `openai_compatible` / `qwen-plus`
- Embedding: `openai_compatible` / `qwen3.7-text-embedding`
- Search: `tavily`
- MCP enabled for this run: `False`
- MCP auth token configured: `False`
- Rerank: `dashscope`