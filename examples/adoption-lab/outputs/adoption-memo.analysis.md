# Adoption Memo Experiment Analysis

## Simulated User Input

Please simulate Northstar Platform, a 5-engineer Python/FastAPI platform team, and evaluate whether the GitHub repository langchain-ai/langgraph should be piloted as the workflow runtime for an internal open-source adoption memo and technical decision research desk. Combine local team constraints, public documentation or GitHub/Web evidence, whether graph structure is actually necessary, risks, a pilot plan, and a rollback plan.

## Headline Metrics

- Status: `completed`
- Evaluation passed: `True`
- Source count: 3
- Context recall: 1.0
- Citation precision: 1.0
- Faithfulness proxy: 0.897
- Expected term recall: 0.5
- Team constraint recall: 0.625
- Constraint coverage passed: `True`
- Expected source recall: 0.0

## Routing And Trace

- Route counts: `{'internal': 2}`
- Tool counts: `{'vector_retrieval': 2}`
- Evidence channels: `{'document-chunk': 4, 'run-artifact': 1}`
- Trace events: 61
- Checkpoints: 19
- Handoffs: 15
- Actors: `['evaluator', 'planner', 'reporter', 'research_supervisor', 'researcher', 'retriever', 'supervisor', 'verifier']`

## Graph Retrieval Check

- Document hits: 7
- Graph-enabled document hits: 7
- Graph signal hits: 7
- Matched graph entities sample: `[['FastAPI', 'Northstar Platform', 'Northstar Platform Team Constraints', 'Python'], ['GitHub', 'Python/FastAPI'], ['FastAPI', 'Northstar Platform', 'Northstar Platform Team Constraints', 'Python']]`
- Matched graph relationships sample: `[['Northstar Platform Team Constraints -[co_occurs_with]-> Team Shape', 'Team Shape -[co_occurs_with]-> Northstar Platform'], ['platform -[co_occurs_with]-> adoption'], ['Northstar Platform Team Constraints -[co_occurs_with]-> Team Shape', 'Team Shape -[co_occurs_with]-> Northstar Platform']]`

Interpretation: graph design is justified only when the retrieved team context contains workflow stages, dependencies, quality gates, or revision paths. If graph_signal_hits is 0, the system still used graph-enabled indexing, but this run did not prove that graph signal improved retrieval.

## Matched Expectations

- Terms: `['LangGraph', 'langchain-ai/langgraph', 'trace', 'FastAPI', 'pilot', 'rollback']`
- Constraints: `['5-engineer', 'Python 3.11', 'FastAPI', 'one machine', 'replayable runs']`
- Source patterns: `[]`

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

- Chat: `deterministic` / `heuristic-chat`
- Embedding: `deterministic` / `hashed-embedding`
- Search: `none`
- MCP enabled for this run: `False`
- MCP auth token configured: `False`
- Rerank: `rule`