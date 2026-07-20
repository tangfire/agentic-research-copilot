from agentic_research_copilot.agents.verifier import VerifierAgent
from agentic_research_copilot.schemas import ResearchReport, ReportSection


def test_verifier_flags_reports_without_evidence_or_confidence():
    report = ResearchReport(
        title="Unsupported Report",
        summary="This report has no citations.",
        sections=[ReportSection(heading="Claim", content="Unsupported claim.")],
        confidence=0.2,
    )

    issues = VerifierAgent().verify(report, evidence=[], plan=[])

    assert "No evidence attached to the report." in issues
    assert "Confidence is too low." in issues
    assert "Report sections exist but no citations were assembled." in issues
    assert "Sections missing citations: 1" in issues
