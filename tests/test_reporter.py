from agentic_research_copilot.agents import ReporterAgent
from agentic_research_copilot.provider_base import ModelUsage
from agentic_research_copilot.providers import StructuredOutputError
from agentic_research_copilot.schemas import (
    EvidenceItem,
    ReporterContract,
    ReporterSectionDraft,
    ReportSection,
)


class SynthesizingProvider:
    name = "fake_synthesizer"
    embedding_dimensions = 256

    def compose_report(self, topic, sections, evidence, confidence):
        return (
            ReporterContract(
                title=f"{topic} synthesized",
                summary="LLM synthesized summary.",
                sections=[
                    ReporterSectionDraft(
                        heading="Synthesized Analysis",
                        content="The final writer merged compressed findings into a citation-backed section.",
                        citation_indexes=[2, 1],
                    )
                ],
                source_index=["[1] First", "[2] Second"],
                confidence=confidence,
            ),
            ModelUsage(provider=self.name, model="fake", prompt_tokens=1, completion_tokens=1),
        )


class EmptyReporterProvider:
    name = "empty_reporter"
    embedding_dimensions = 256

    def compose_report(self, topic, sections, evidence, confidence):
        raise StructuredOutputError(
            response_model="ReporterContract",
            attempts=3,
            reason="provider returned empty message content",
            diagnostics={"finish_reason": "stop", "content_chars": 0},
        )


def test_reporter_uses_synthesized_sections_with_existing_citations():
    fallback = [
        ReportSection(
            heading="Template",
            content="Template content.",
            citations=[
                EvidenceItem(title="First", source="source-a", snippet="A"),
                EvidenceItem(title="Second", source="source-b", snippet="B"),
            ],
        )
    ]
    citations = fallback[0].citations

    report = ReporterAgent(model_provider=SynthesizingProvider()).build_report(
        "agentic research",
        fallback,
        citations,
        0.8,
    )

    assert report.sections[0].heading == "Synthesized Analysis"
    assert "citation-backed" in report.sections[0].content
    assert [item.title for item in report.sections[0].citations] == ["Second", "First"]


def test_reporter_preserves_team_constraint_coverage_section():
    fallback = [
        ReportSection(
            heading="Analysis",
            content="Template analysis.",
            citations=[EvidenceItem(title="First", source="source-a", snippet="A")],
        ),
        ReportSection(
            heading="团队约束覆盖",
            content="- 已覆盖：First deployment must run on one machine.",
            citations=[EvidenceItem(title="Constraints", source="team-context", snippet="One machine.")],
        ),
    ]
    citations = [item for section in fallback for item in section.citations]

    report = ReporterAgent(model_provider=SynthesizingProvider()).build_report(
        "agentic research",
        fallback,
        citations,
        0.8,
    )

    assert [section.heading for section in report.sections] == [
        "Synthesized Analysis",
        "团队约束覆盖",
    ]
    assert "one machine" in report.sections[1].content


def test_reporter_preserves_draft_when_structured_output_is_empty():
    fallback = [
        ReportSection(
            heading="证据分析",
            content="研究阶段已经完成了引用绑定的分析草稿。",
            citations=[EvidenceItem(title="First", source="source-a", snippet="A")],
        )
    ]
    reporter = ReporterAgent(model_provider=EmptyReporterProvider())

    report = reporter.build_report("agentic research", fallback, fallback[0].citations, 0.7)

    assert report.sections[0].heading == "证据分析"
    assert report.sections[0].citations[0].title == "First"
    assert reporter.last_degraded_reason
    assert "empty message content" in reporter.last_degraded_reason
