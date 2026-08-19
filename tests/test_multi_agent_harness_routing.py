from pathlib import Path

from agentic_research_copilot.multi_agent_harness import select_specialists, summarize_run
from agentic_research_copilot.schemas import BenchmarkTask, ResearchRequest
from agentic_research_copilot.schemas import (
    AgentRoleAssignment,
    EvidenceItem,
    EvidenceLedger,
    RAGEvaluation,
    ResearchReport,
    ResearchRun,
    RunTraceEvent,
)


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


def test_benchmark_contract_requires_completed_tools_and_expected_evidence():
    task = BenchmarkTask(
        task_id="strict-tool-coverage",
        topic="Evaluate a repo with expected web and local evidence.",
        expected_agent_ids=["architecture_fit"],
        expected_tools=["web_search", "vector_retrieval"],
        expected_evidence_kinds=["web", "document-chunk"],
        min_source_count=2,
    )
    request = ResearchRequest(topic=task.topic)
    run = ResearchRun(
        run_id="run-strict",
        request=request,
        status="completed",
        trace=[
            RunTraceEvent(
                kind="tool_call",
                actor="ArchitectureFitAgent",
                tool_name="web_search",
                status="started",
                message="Started web search but did not complete.",
            ),
            RunTraceEvent(
                kind="tool_call",
                actor="ArchitectureFitAgent",
                tool_name="vector_retrieval",
                status="completed",
                message="Retrieved local evidence.",
            ),
        ],
        evidence=[
            EvidenceItem(
                title="Local note",
                source="local-kb",
                kind="document-chunk",
                snippet="Local grounding evidence.",
                metadata={"plan_item_id": "item_1"},
            )
        ],
        report=ResearchReport(title="Memo", summary="Summary", source_count=1),
        evaluation=RAGEvaluation(citation_precision=1.0, passed=True),
    )
    assignments = [
        AgentRoleAssignment(
            assignment_id="assignment-1",
            agent_id="architecture_fit",
            agent_name="ArchitectureFitAgent",
            status="completed",
            selected_tools=["web_search", "vector_retrieval"],
            evidence_count=1,
        )
    ]
    summary = summarize_run(
        run,
        task=task,
        assignments=assignments,
        route_decisions=[],
        conflicts=[],
        evidence_ledger=EvidenceLedger(
            run_id=run.run_id,
            total_evidence_count=1,
            utilization_rate=1.0,
        ),
    )

    assert summary.tool_completed_success_rate == 0.5
    assert summary.expected_tool_coverage == 0.5
    assert summary.expected_evidence_coverage == 0.5
    assert summary.passed is False
    assert any("missing web_search" in note for note in summary.notes)
