# Open Source Adoption Checklist

Use this checklist when the team evaluates a repository or library for production use.

## Decision Gates

1. Problem fit: the repository must solve a current team problem, not a future-maybe problem.
2. Integration fit: the project should work with Python/FastAPI service boundaries or have a clear adapter path.
3. Operational fit: first deployment should avoid Kubernetes-only assumptions, complex distributed control planes, or hard-to-debug background behavior.
4. Maintainability: the repository should show active maintenance, readable documentation, and a healthy issue/release signal.
5. Evidence quality: the memo should prefer official docs, GitHub repository evidence, release notes, issue discussions, and implementation files over generic blog posts.
6. Exit path: the team must be able to reverse the decision or isolate the dependency behind a small interface.
7. Pilot scope: the first pilot should be smaller than full adoption and should include success metrics.

## Required Memo Sections

- Decision summary: adopt, pilot, defer, or reject.
- Fit against local constraints.
- GitHub and documentation evidence.
- Risks and unknowns.
- Pilot plan and rollback plan.
- Metrics from the research run: source count, context recall, citation precision, faithfulness proxy, and trace coverage.

## Evidence Red Flags

- Report sections without citations.
- Claims about repo health without repository or release evidence.
- Recommendations that ignore the local deployment boundary.
- A graph architecture proposal when the workflow has no cycles, branching, handoffs, or revision loop.
- High confidence with low source diversity.
