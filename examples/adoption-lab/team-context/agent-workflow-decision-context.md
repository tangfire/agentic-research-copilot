# Agent Workflow Decision Context

The team is considering whether to use LangGraph for bounded agentic workflows, especially technical research workflows that include planning, delegated evidence gathering, retrieval, report writing, verification, and possible revision.

## Why This Question Is Not A Simple Chatbot Question

The workflow has stateful stages:

- planner creates a research brief and plan items
- research supervisor decides which tools each plan item needs
- researcher gathers public web or GitHub-style evidence
- retriever grounds private team constraints and prior docs
- reporter synthesizes sections with citations
- verifier checks unsupported claims
- evaluator calculates quality metrics
- failed quality gates may route back into a revision pass

This is a real graph-shaped workflow because the next node depends on state: evidence sufficiency, citation coverage, revision budget, and whether private context was retrieved. A simple linear prompt chain would hide these decisions and make trace/replay harder.

## When Graph Design Would Be Overkill

Do not use graph orchestration for:

- one-shot summaries
- simple Q&A over a single document
- CRUD screens
- prompt templates that never branch or revise
- workflows where failures do not need a recoverable state boundary

## Pilot Candidate

Pilot LangGraph only for the adoption memo workflow:

Input: repository, technical question, local team constraints.

Output: decision memo with GitHub/web evidence, local constraint evidence, citations, trace, and evaluation metrics.

Success means the system can repeatedly answer adoption questions without asking engineers to paste the same team constraints into every prompt.
