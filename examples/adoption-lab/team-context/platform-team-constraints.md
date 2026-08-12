# Northstar Platform Team Constraints

This document is a fictional but realistic private context pack for testing the adoption memo workflow.

## Team Shape

- Team: Northstar Platform, a 5-engineer internal platform team.
- Stack: Python 3.11, FastAPI, Pydantic, Celery/Redis for background jobs, SQLite or Postgres for durable state, Qdrant for vector search.
- Deployment: one Linux VM or a small Docker Compose stack. Kubernetes is not available for the first version.
- Users: 18 product/backend engineers who ask repeated technical questions about dependencies, migrations, architecture choices, and operational risks.
- Operating mode: the team prefers boring infrastructure and observable workflows over opaque all-in-one agents.

## Product Need

The team wants an internal technical research desk. A useful run should accept a GitHub repository, a technical question, and these local constraints, then produce an adoption memo with citations, evidence gaps, trace, and evaluation metrics.

The memo must answer:

- Does the repository solve a real team problem?
- Which parts of the project map to our current Python/FastAPI stack?
- What extra operational burden would it introduce?
- What failure modes or lock-in risks matter?
- What should we pilot first, and what should remain out of scope?

## Hard Constraints

- First deployment must run on one machine.
- The first version must support replayable runs and persisted trace artifacts.
- Every recommendation must be backed by at least one citation or marked as an assumption.
- Private team constraints should not be pasted into every prompt by hand; they should be stored as local documents and retrieved automatically.
- The team accepts graph-based orchestration only when the task has real branching, dependency, or revision structure. Simple FAQ and CRUD workflows should not use graph design just to look sophisticated.

## Evaluation Expectations

For a run to be trusted, it should report:

- context recall, so we can see whether the local constraints were retrieved
- citation precision, so every section is source-backed
- faithfulness proxy, so generation stays close to evidence
- source diversity, so one web result does not dominate the memo
- trace events, so planner, supervisor, researcher, retriever, reporter, verifier, and evaluator behavior can be inspected
