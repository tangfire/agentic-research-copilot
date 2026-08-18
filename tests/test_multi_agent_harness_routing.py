from pathlib import Path

from agentic_research_copilot.multi_agent_harness import select_specialists
from agentic_research_copilot.schemas import BenchmarkTask, ResearchRequest


def test_route_selection_does_not_match_repo_on_substrings():
    request = ResearchRequest(
        topic="Evaluate memory extraction quality for project constraints, including precision and recall proxy.",
        metadata={"benchmark_scenario": "demo_readiness_risk_review"},
    )

    selected = set(select_specialists(request))

    assert "repo_signal" not in selected
    assert selected == {"architecture_fit"}


def test_route_selection_uses_hard_constraints_for_ops_risk():
    request = ResearchRequest(
        topic="Evaluate whether hard team constraints are consistently covered in adoption memos.",
        metadata={
            "benchmark_scenario": "open_source_adoption_review",
            "hard_constraints": [
                "constraints must appear in report",
                "coverage below threshold must warn",
            ],
        },
    )

    selected = set(select_specialists(request))

    assert "ops_risk" in selected
    assert "repo_signal" not in selected


def test_route_selection_covers_labeled_benchmark_cases():
    dataset = Path("examples/research-desk-v4-benchmark.jsonl")
    cases = [
        BenchmarkTask.model_validate_json(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(cases) >= 24
    for case in cases:
        request = ResearchRequest(
            topic=case.topic,
            depth=case.depth,
            metadata={
                "benchmark_scenario": case.scenario,
                "hard_constraints": case.hard_constraints,
            },
        )

        assert set(select_specialists(request)) == set(case.expected_agent_ids), case.task_id
