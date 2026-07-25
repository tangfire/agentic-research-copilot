# Web App

Local research workspace served by the FastAPI root route.

This is intentionally dependency-light for the first release. It gives the project a
usable product surface without adding a Node build step before the AI core is proven.
The main interaction follows Open Deep Research's LangGraph Studio style: users
submit a research thread, then inspect the brief, plan, evidence-gathering stages,
final report, evaluation gates, and trace artifacts.

Views:

- Portal: submit asynchronous research jobs through a thread-style Chinese UI, inspect the research brief, stage progress, plan items, quality gates, source summary, and citation-backed report.
- Runs: inspect job state, completed runs, plan transitions, and clear local history.
- Sources: add, delete, and clear project documents and grounding context.
- Memory: add structured project and user memory.
- Traces: inspect telemetry events, run traces, checkpoints, and lower-level handoffs.
- Config: inspect runtime agents, tools, routes, providers, provider readiness, and quality gates.

The same API can later support a richer Next.js or Vue frontend if the product needs
authentication, streaming jobs, collaborative report editing, or heavier UI state.

The portal is laid out as an integrated research console instead of a wall of cards:
a top command bar for the question, a central workspace for brief/plan/report, and a
right inspector for quality gates, sources, and trace details. The first-use path
intentionally mirrors Open Deep Research: ask a question, follow the research thread,
then read the report. Provider details, source lists, section citations, and low-level
routes/traces remain available for inspection without turning the entry flow into a
configuration form.
