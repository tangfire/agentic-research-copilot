from __future__ import annotations

from ..provider_base import ResearchModelProvider
from ..providers import build_model_provider
from ..schemas import EvidenceItem, PlanItem, ResearchReport, VerificationContract
from ..settings import AppSettings, load_settings


class VerifierAgent:
    def __init__(
        self,
        model_provider: ResearchModelProvider | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_provider = model_provider or build_model_provider(self.settings)
        self.last_usage = None

    def assess(
        self,
        report: ResearchReport,
        evidence: list[EvidenceItem],
        plan: list[PlanItem] | None = None,
        *,
        revision_count: int = 0,
        max_revisions: int | None = None,
    ) -> VerificationContract:
        contract, usage = self.model_provider.assess_report(
            report,
            evidence,
            plan or [],
            revision_count=revision_count,
            max_revisions=max_revisions or self.settings.max_revisions,
        )
        self.last_usage = usage
        return contract

    def verify(
        self,
        report: ResearchReport,
        evidence: list[EvidenceItem],
        plan: list[PlanItem] | None = None,
        *,
        revision_count: int = 0,
        max_revisions: int | None = None,
    ) -> list[str]:
        return self.assess(
            report,
            evidence,
            plan,
            revision_count=revision_count,
            max_revisions=max_revisions,
        ).issues
