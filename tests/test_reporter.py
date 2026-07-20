from agentic_research_copilot.agents import ReporterAgent
from agentic_research_copilot.providers import ModelUsage
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
