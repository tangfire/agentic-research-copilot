from agentic_research_copilot.multi_agent_harness import build_role_assignments, build_route_decisions, select_specialists
from agentic_research_copilot.schemas import PlanItem, ResearchRequest, RetrievalRoute


def test_repo_adoption_plans_route_to_repo_and_ops_specialists():
    request = ResearchRequest(
        topic=(
            "Evaluate langchain-ai/langgraph for a 5-engineer Python/FastAPI team. "
            "The first deployment must run on one machine and support replayable runs."
        ),
        metadata={
            "hard_constraints": [
                "5-engineer team",
                "Python 3.11",
                "FastAPI",
                "Docker Compose",
                "one machine",
                "replayable runs",
                "graph-based orchestration only when branching is real",
            ]
        },
    )
    plan = [
        PlanItem(
            id="q1-background",
            question="What does the repository README and issue tracker say about maintainer activity and evidence quality?",
            purpose="Inspect source-of-truth repo signals.",
            search_query="site:github.com langchain-ai/langgraph README issue release",
        ),
        PlanItem(
            id="q2-rollback",
            question="What rollout and rollback constraints matter for a single-machine deployment?",
            purpose="Check operational risk and rollback paths.",
            search_query="langgraph single machine rollback deployment",
        ),
    ]
    routes = [
        RetrievalRoute(
            plan_item_id=item.id,
            mode="hybrid",
            reason="Hybrid evidence needed.",
            selected_tools=["web_search", "mcp_tool"],
            web_queries=["example web query"],
            internal_queries=["example internal query"],
            min_evidence=1,
            min_sources=1,
        )
        for item in plan
    ]

    selected = select_specialists(request, plan=plan)
    assert "repo_signal" in selected
    assert "ops_risk" in selected

    assignments = build_role_assignments(request, plan, routes, [], run_id="run-1")
    decisions = build_route_decisions(request, plan, routes, [], assignments, run_id="run-1")

    assert any(decision.plan_item_id == "q1-background" and decision.agent_id == "repo_signal" for decision in decisions)
    assert any(decision.plan_item_id == "q2-rollback" and decision.agent_id == "ops_risk" for decision in decisions)
