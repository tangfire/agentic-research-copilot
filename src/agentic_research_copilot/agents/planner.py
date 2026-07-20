from __future__ import annotations

from ..providers import ResearchModelProvider, build_model_provider
from ..schemas import CorpusProfile, PlannerContract, ResearchRequest
from ..settings import AppSettings, load_settings


class PlannerAgent:
    def __init__(
        self,
        model_provider: ResearchModelProvider | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_provider = model_provider or build_model_provider(self.settings)
        self.last_usage = None

    def draft(
        self,
        request: ResearchRequest,
        *,
        corpus_profile: CorpusProfile | None = None,
        memory_records=(),
        revision_count: int = 0,
        revision_notes=(),
    ) -> PlannerContract:
        contract, usage = self.model_provider.draft_plan(
            request,
            corpus_profile or CorpusProfile(),
            memory_records,
            revision_count=revision_count,
            revision_notes=revision_notes,
        )
        self.last_usage = usage
        return contract

    def create_research_brief(
        self,
        request: ResearchRequest,
        *,
        corpus_profile: CorpusProfile | None = None,
        memory_records=(),
        revision_count: int = 0,
        revision_notes=(),
    ) -> str:
        return self.draft(
            request,
            corpus_profile=corpus_profile,
            memory_records=memory_records,
            revision_count=revision_count,
            revision_notes=revision_notes,
        ).research_brief

    def create_plan(
        self,
        request: ResearchRequest,
        *,
        corpus_profile: CorpusProfile | None = None,
        memory_records=(),
        revision_count: int = 0,
        revision_notes=(),
    ):
        return self.draft(
            request,
            corpus_profile=corpus_profile,
            memory_records=memory_records,
            revision_count=revision_count,
            revision_notes=revision_notes,
        ).plan
