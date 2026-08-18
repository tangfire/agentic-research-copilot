# Open Source Adoption Review

## Purpose

Use this skill when you need to decide whether a GitHub repo or open-source project is worth introducing to a small team.

## When to Use

- A repo URL or `owner/name` is provided.
- The user cares about team constraints, deployment cost, maintenance quality, or demo value.
- The output should be an adoption memo, not a generic summary.

## Required Inputs

- Target repo or project
- Team constraints
- Decision question

## Operating Procedure

1. Confirm the target repo and decision question.
2. Read workspace constraints first.
3. Run the preflight script if available.
4. Build a short plan before research.
5. Gather evidence from repo metadata, official docs, issue/release signals, and local notes.
6. Score the result against constraint coverage and cite the evidence.

## Tool Policy

- Preferred tools: web search, local knowledge base, GitHub MCP read-only tools.
- Avoid destructive tools.
- Treat missing GitHub MCP auth as an explicit unavailable state.

## Output Contract

- Recommendation
- Evidence index
- Constraint coverage table
- Risks and mitigation
- Trace and evaluation summary

## Notes for the Agent

- Do not skip the confirmation gate if the request is vague.
- Prefer concrete repo signals over generic LLM intuition.
- Reuse saved workspace constraints on later turns.
