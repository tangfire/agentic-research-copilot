from __future__ import annotations

from ..schemas import EvidenceItem, ResearchReport


class VerifierAgent:
    def verify(self, report: ResearchReport, evidence: list[EvidenceItem]) -> list[str]:
        issues: list[str] = []
        if not evidence:
            issues.append("No evidence attached to the report.")
        if len(report.sections) < 2:
            issues.append("Report is too thin for interview-grade explanation.")
        if report.confidence < 0.5:
            issues.append("Confidence is too low.")
        return issues

