from __future__ import annotations

from ..schemas import EvidenceItem, ResearchReport, ReportSection


class ReporterAgent:
    def build_report(
        self,
        topic: str,
        sections: list[ReportSection],
        citations: list[EvidenceItem],
        confidence: float,
    ) -> ResearchReport:
        summary = f"A structured research brief on {topic}."
        return ResearchReport(
            title=f"{topic} Research Brief",
            summary=summary,
            sections=sections,
            citations=citations,
            confidence=confidence,
        )

