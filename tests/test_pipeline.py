from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.schemas import ResearchRequest


def test_pipeline_returns_report():
    copilot = ResearchCopilot()
    result = copilot.run(ResearchRequest(topic="multi-agent memory"))

    assert result.status == "completed"
    assert result.report is not None
    assert "multi-agent" in result.report.title.lower()

