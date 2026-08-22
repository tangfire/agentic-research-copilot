from agentic_research_copilot.constraint_evaluation import (
    extract_constraint_coverage_from_run,
    extract_constraint_texts,
)
from agentic_research_copilot.schemas import EvidenceItem, ResearchReport, ResearchRequest, ResearchRun, ReportSection


def test_constraint_extraction_ignores_generated_coverage_and_parent_metadata():
    text = """
    - 已覆盖：First deployment must run on one machine.
    Parent document: Northstar Platform Team Constraints
    Matched child chunk: 1/2
    - Stack: Python/FastAPI service boundaries.
    """

    constraints = extract_constraint_texts(text)

    assert constraints == ["Stack: Python/FastAPI service boundaries."]


def test_run_constraint_coverage_does_not_use_report_output_as_input():
    run = ResearchRun(
        run_id="run-report-output-only",
        request=ResearchRequest(topic="Evaluate langchain-ai/langgraph"),
        report=ResearchReport(
            title="report",
            summary="summary",
            sections=[
                ReportSection(
                    heading="团队约束覆盖",
                    content="- 已覆盖：First deployment must run on one machine.",
                    citations=[EvidenceItem(title="Team constraints", source="local", snippet="one machine")],
                )
            ],
        ),
        evidence=[EvidenceItem(title="Team constraints", source="local", snippet="one machine")],
    )

    assert extract_constraint_coverage_from_run(run) == []
